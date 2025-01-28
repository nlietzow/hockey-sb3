from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.resolve()
_base_dir = _project_root / "checkpoints"
SAC_DIR = _base_dir / "sac"
CROSS_Q_DIR = _base_dir / "cross_q"

assert SAC_DIR.exists(), f"{SAC_DIR} does not exist."
