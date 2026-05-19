from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from string import Template
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expand TraceLock config templates.")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[], help="Override key=value at the top level.")
    parser.add_argument("--gpus", nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--sets", nargs="+", default=None, help="Keep only the named top-level eval sets.")
    return parser.parse_args()


def rewrite_tracelock_aliases(value: Any) -> Any:
    if isinstance(value, list):
        return [rewrite_tracelock_aliases(item) for item in value]
    if not isinstance(value, dict):
        return value

    out: dict[str, Any] = {}
    for key, item in value.items():
        out[key] = rewrite_tracelock_aliases(item)
    return out


def parse_scalar(raw: str) -> Any:
    lowered = raw.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def main() -> None:
    args = parse_args()
    env = dict(os.environ)
    if "TRACELOCK_PYTHON" not in env:
        env["TRACELOCK_PYTHON"] = sys.executable
    rendered = Template(args.template.read_text()).safe_substitute(env)
    payload = json.loads(rendered)
    payload = rewrite_tracelock_aliases(payload)

    if args.gpus is not None and args.gpus:
        payload["gpus"] = list(args.gpus)
    if args.num_samples is not None:
        payload["num_samples"] = int(args.num_samples)
    if args.experiment_name:
        payload["experiment_name"] = args.experiment_name
    if args.sets is not None:
        requested = set(args.sets)
        payload["sets"] = [item for item in payload["sets"] if item.get("name") in requested]
        kept = {item["name"] for item in payload["sets"]}
        missing = sorted(requested - kept)
        if missing:
            raise ValueError(f"Unknown eval set(s): {', '.join(missing)}")

    for override in args.set:
        if "=" not in override:
            raise ValueError(f"Expected key=value override, got {override!r}")
        key, raw_value = override.split("=", 1)
        payload[key] = parse_scalar(raw_value)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
