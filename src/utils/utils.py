from pathlib import Path

_project_root = Path(__file__).parent.parent.parent.resolve()
CHECKPOINTS_DIR = _project_root / "checkpoints"
