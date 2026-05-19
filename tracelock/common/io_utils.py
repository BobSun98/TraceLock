from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

import torch


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _tmp_path_for_atomic_write(path: Path) -> Path:
    return path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')


def write_json(path: str | Path, payload: Any) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    tmp_path = _tmp_path_for_atomic_write(out_path)
    try:
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + '\n')


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    with out_path.open('w', encoding='utf-8') as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + '\n')


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    with Path(path).open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_pt(path: str | Path, payload: dict[str, Any]) -> None:
    out_path = Path(path)
    ensure_dir(out_path.parent)
    tmp_path = _tmp_path_for_atomic_write(out_path)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location='cpu', weights_only=False)
