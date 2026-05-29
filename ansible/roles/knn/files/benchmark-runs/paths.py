import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_PYTHON = ROOT / ".venv/bin/python"
RUN_SCRIPT = ROOT / "run.py"
WORKER_SCRIPT = ROOT / "benchmark_worker.py"
BIN_DIR = ROOT / "../bin"
DATASET_DIR = ROOT / "../datasets"
LES3_DIR = BIN_DIR / "les3"
LES3_GROUP_DIR = DATASET_DIR / "les3"
LOCAL_CONFIG = ROOT / "config.local.json"


def ensure_project_python() -> int | None:
    """Re-exec through the uv environment when scheduler dependencies are needed."""
    if Path(sys.prefix) == ROOT / ".venv":
        return None
    if VENV_PYTHON.exists():
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), *sys.argv])
    if not VENV_PYTHON.exists():
        print(f"Missing {VENV_PYTHON}. Run `uv sync` in {ROOT}.", file=sys.stderr)
        return 2
    return None
