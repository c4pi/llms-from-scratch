from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_env_data_dir = __import__("os").environ.get("LLMSFS_DATA_DIR")
DATA_DIR = Path(_env_data_dir) if _env_data_dir else (_PROJECT_ROOT / "data")
RAW_DIR = DATA_DIR / "raw"
CHECKPOINTS_DIR = DATA_DIR / "checkpoints"
PLOTS_DIR = DATA_DIR / "plots"


def ensure_dirs():
    for d in [DATA_DIR, RAW_DIR, CHECKPOINTS_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
