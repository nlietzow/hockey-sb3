from pathlib import Path
from typing import Type, Union

from sbx import CrossQ, DDPG, SAC, TD3, TQC

_project_root = Path(__file__).parent.parent.parent.resolve()
CHECKPOINTS_DIR = _project_root / "checkpoints"

Algorithm = Union[
    Type[CrossQ],
    Type[DDPG],
    Type[SAC],
    Type[TD3],
    Type[TQC],
]
