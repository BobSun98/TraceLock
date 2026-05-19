from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
from typing import Any


NAME_ERROR_RE = re.compile(r"NameError: name '([^']+)' is not defined")
TYPING_IMPORT_SYMBOLS = (
    "List",
    "Tuple",
    "Dict",
    "Set",
    "Optional",
    "Any",
    "Union",
    "Callable",
    "Iterable",
    "Iterator",
    "Sequence",
    "Mapping",
)
MODULE_IMPORT_NAMES = (
    "math",
    "re",
    "itertools",
    "functools",
    "collections",
    "heapq",
    "bisect",
    "random",
    "string",
)


def extract_python_code(answer: str) -> str:
    text = str(answer).replace("\r\n", "\n").replace("\r", "\n")
    if "```python" in text:
        return text.split("```python", 1)[-1].split("```", 1)[0].strip()
    if "```" in text:
        return text.split("```", 1)[-1].split("```", 1)[0].strip()
    return text.strip()


def _run_verifier(
    *,
    code: str,
    test: str,
    entry_point: str,
    timeout_seconds: float,
    python_executable: str | None,
) -> dict[str, Any]:
    payload = {
        "code": code,
        "test": test,
        "entry_point": entry_point,
    }
    verifier = """
import contextlib
import io
import json
import sys

payload = json.loads(sys.stdin.read())
namespace = {}
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    exec(payload["code"], namespace)
    exec(payload["test"], namespace)
    namespace["check"](namespace[payload["entry_point"]])
"""
    python_bin = python_executable or sys.executable
    with tempfile.TemporaryDirectory(prefix="tracelock_code_eval_", dir="/tmp") as tmp_dir:
        try:
            completed = subprocess.run(
                [python_bin, "-I", "-c", verifier],
                input=json.dumps(payload, ensure_ascii=False),
                text=True,
                cwd=tmp_dir,
                capture_output=True,
                timeout=float(timeout_seconds),
            )
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "error_type": "timeout",
                "error_message": f"TimeoutExpired after {float(timeout_seconds):.2f}s",
                "code": code,
            }

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        message = stderr.splitlines()[-1].strip() if stderr else "execution failed"
        return {
            "passed": False,
            "error_type": "runtime",
            "error_message": message,
            "code": code,
        }

    return {
        "passed": True,
        "error_type": "none",
        "error_message": "",
        "code": code,
    }


def _detect_missing_import_name(error_message: str) -> str | None:
    match = NAME_ERROR_RE.search(error_message)
    if match is None:
        return None
    return match.group(1)


def _build_missing_import_patch(code: str, missing_name: str) -> tuple[str | None, list[str]]:
    import_lines: list[str] = []

    if missing_name in TYPING_IMPORT_SYMBOLS:
        needed_symbols = [
            symbol
            for symbol in TYPING_IMPORT_SYMBOLS
            if re.search(rf"\b{re.escape(symbol)}\b", code)
        ]
        if not needed_symbols:
            return None, []
        if "from typing import" in code:
            return None, []
        import_lines.append(f"from typing import {', '.join(needed_symbols)}")
    elif missing_name in MODULE_IMPORT_NAMES:
        if f"{missing_name}." not in code:
            return None, []
        if re.search(rf"^\s*import\s+{re.escape(missing_name)}\b", code, flags=re.MULTILINE):
            return None, []
        import_lines.append(f"import {missing_name}")
    else:
        return None, []

    if not import_lines:
        return None, []
    repaired_code = "\n".join(import_lines) + "\n" + code
    return repaired_code, import_lines


def _maybe_retry_with_missing_imports(
    *,
    result: dict[str, Any],
    test: str,
    entry_point: str,
    timeout_seconds: float,
    python_executable: str | None,
) -> dict[str, Any]:
    if result.get("passed"):
        return result
    if result.get("error_type") != "runtime":
        return result

    error_message = str(result.get("error_message") or "")
    missing_name = _detect_missing_import_name(error_message)
    if missing_name is None:
        return result

    repaired_code, import_lines = _build_missing_import_patch(str(result["code"]), missing_name)
    if repaired_code is None:
        return result

    try:
        compile(repaired_code, "<solution>", "exec")
    except SyntaxError:
        return result

    retry_result = _run_verifier(
        code=repaired_code,
        test=test,
        entry_point=entry_point,
        timeout_seconds=timeout_seconds,
        python_executable=python_executable,
    )
    if not retry_result.get("passed"):
        return result

    retry_result["import_repair_applied"] = True
    retry_result["import_repair_lines"] = import_lines
    retry_result["import_repair_missing_name"] = missing_name
    return retry_result


def verify_code_answer(
    *,
    answer: str,
    test: str,
    entry_point: str,
    timeout_seconds: float = 30.0,
    python_executable: str | None = None,
) -> dict[str, Any]:
    code = extract_python_code(answer)
    if not code:
        return {
            "passed": False,
            "error_type": "no_code",
            "error_message": "No code block extracted from answer.",
            "code": "",
        }

    try:
        compile(code, "<solution>", "exec")
    except SyntaxError as exc:
        return {
            "passed": False,
            "error_type": "syntax",
            "error_message": f"{type(exc).__name__}: {exc}",
            "code": code,
        }

    try:
        compile(test, "<test>", "exec")
    except SyntaxError as exc:
        return {
            "passed": False,
            "error_type": "syntax",
            "error_message": f"{type(exc).__name__}: {exc}",
            "code": code,
        }
    result = _run_verifier(
        code=code,
        test=test,
        entry_point=entry_point,
        timeout_seconds=timeout_seconds,
        python_executable=python_executable,
    )
    return _maybe_retry_with_missing_imports(
        result=result,
        test=test,
        entry_point=entry_point,
        timeout_seconds=timeout_seconds,
        python_executable=python_executable,
    )
