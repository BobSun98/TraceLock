from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge TraceLock trace groups into one samples directory with symlinks.")
    parser.add_argument("--trace-run-dir", type=Path, required=True)
    parser.add_argument("--output-samples-dir", type=Path, required=True)
    parser.add_argument("--groups", nargs="+", default=["coding", "others"])
    parser.add_argument("--overwrite-links", action="store_true")
    return parser.parse_args()


def link_split(group: str, group_samples_dir: Path, output_samples_dir: Path, split: str, overwrite_links: bool) -> int:
    src_dir = group_samples_dir / split
    if not src_dir.exists():
        return 0
    dst_dir = output_samples_dir / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src in sorted(src_dir.glob("*.pt")):
        dst = dst_dir / f"{group}_{src.name}"
        if dst.exists() or dst.is_symlink():
            if not overwrite_links:
                continue
            dst.unlink()
        dst.symlink_to(src.resolve())
        count += 1
    return count


def main() -> None:
    args = parse_args()
    output = args.output_samples_dir.resolve()
    total = 0
    for group in args.groups:
        group_samples = args.trace_run_dir / group / "samples"
        total += link_split(group, group_samples, output, "train", args.overwrite_links)
        total += link_split(group, group_samples, output, "val", args.overwrite_links)
    print(f"Prepared {total} sample links under {output}")


if __name__ == "__main__":
    main()
