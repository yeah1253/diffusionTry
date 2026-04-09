#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承 / HIL 诊断训练 —— 以 Transformer 为特征提取骨干（BearingTransformer）。

数据划分、滑窗、混合/双模型策略、早停、checkpoint 字段等与 ``train_bearing_cnn1d.py`` 完全一致；
仅将 ``MODEL_TYPE`` 固定为 ``"transformer"``，对应 ``model.py`` 中的 ``BearingTransformer``
（4 层 TransformerEncoder + Patch 嵌入，见 ``ARCH_TRANSFORMER``）。

输出文件名（脚本同目录，与 CNN 脚本区分）：
  best_model_transformer.pth
  model_mixed_transformer.pth
  model_real_only_transformer.pth

依赖: pip install torch numpy
      （可选 scipy，用于 .mat；与主训练脚本相同）

用法（与 train_bearing_cnn1d.py 相同）::
  python train_bearing_transformer1d.py              # 默认：双模型（混合 + 纯真实）
  python train_bearing_transformer1d.py --single      # 单模型 best_model_transformer.pth
  python train_bearing_transformer1d.py --eval-only   # 仅评估上述单模型 checkpoint

数据路径、超参请在 ``train_bearing_cnn1d.py`` 顶部全局变量中修改（本文件通过 import 复用）。
"""

from __future__ import annotations

import argparse
import sys

# 复用完整训练管线；在导入后覆盖架构与输出名
import train_bearing_cnn1d as _tb

_tb.MODEL_TYPE = "transformer"
_tb.OUTPUT_MODEL_PATH = f"best_model_{_tb.MODEL_TYPE}.pth"
_tb.OUTPUT_MODEL_PATH_MIXED = f"model_mixed_{_tb.MODEL_TYPE}.pth"
_tb.OUTPUT_MODEL_PATH_REAL_ONLY = f"model_real_only_{_tb.MODEL_TYPE}.pth"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="轴承 1D Transformer 诊断训练（逻辑同 train_bearing_cnn1d.py，架构=BearingTransformer）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式说明：
  默认              双模型 train_two_models() — model_mixed_transformer.pth + model_real_only_transformer.pth
  --single          单模型 train()            — best_model_transformer.pth
  --eval-only       仅评估 best_model_transformer.pth（不训练）

数据根目录、GEN_DATA_ROOT、窗口与超参：编辑 train_bearing_cnn1d.py 顶部常量。
        """,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="不训练，仅加载 best_model_transformer.pth 在测试集上评估",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="单模型训练（生成 best_model_transformer.pth，不使用 GEN_DATA_ROOT 混合逻辑）",
    )
    args = parser.parse_args()

    if args.eval_only:
        return _tb.eval_only()
    if args.single:
        return _tb.train()
    return _tb.train_two_models()


if __name__ == "__main__":
    raise SystemExit(main())
