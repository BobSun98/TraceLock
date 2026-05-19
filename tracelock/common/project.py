from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_ROOT = Path(__file__).resolve().parents[3] / "workspace"


def _add_path(path: Path) -> None:
    resolved = str(path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def setup_project_root(project_root: Path | None = None) -> Path:
    root = (project_root or PROJECT_ROOT).resolve()
    _add_path(root)
    return root
