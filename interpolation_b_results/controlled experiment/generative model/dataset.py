"""
与 train_baselines.py 同目录，转发至 diffusionTry/eval.dataset（实例级 Z-score 等在 BearingSignalDataset.__getitem__）。

仓库布局假定::
    diffusionTry_cond/
      controlled experiment/
        generative model/   ← 本文件
      diffusionTry/
        eval/
"""
from __future__ import annotations

import sys
from pathlib import Path

_this_dir = Path(__file__).resolve().parent
_workspace_root = _this_dir.parent.parent  # diffusionTry_cond
_difftry_root = _workspace_root / "diffusionTry"
if str(_difftry_root) not in sys.path:
    sys.path.insert(0, str(_difftry_root))

from eval.dataset import BearingSignalDataset, build_dataloader  # noqa: E402

__all__ = ["BearingSignalDataset", "build_dataloader"]
