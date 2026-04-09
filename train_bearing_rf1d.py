#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承 / HIL 诊断训练 —— 随机森林（手工时频特征 + sklearn RF，见 ``model.BearingRFWrapper``）。

数据划分、混合/双模型策略等与 ``train_bearing_cnn1d.py`` 完全一致；
``MODEL_TYPE`` 固定为 ``"rf"``：单模型与双模型均走 ``train_rf_model``（无 PyTorch 早停，
在全量训练索引上 fit；测试集为共用纯真实划分）。

特征：每个 1024 点窗口经 ``extract_rf_features`` 得到 26 维向量（与 ``model.py`` 一致）。

依赖: pip install torch numpy scikit-learn
      （可选 scipy，用于 .mat）

输出文件名：
  best_model_rf.pth
  model_mixed_rf.pth
  model_real_only_rf.pth

用法::
  python train_bearing_rf1d.py              # 默认：双模型
  python train_bearing_rf1d.py --single     # 单模型 best_model_rf.pth
  python train_bearing_rf1d.py --eval-only  # 仅评估 best_model_rf.pth（RF 专用加载逻辑）

数据路径、超参请在 ``train_bearing_cnn1d.py`` 顶部全局变量中修改。
"""

from __future__ import annotations

import argparse
from pathlib import Path

# 复用完整训练管线；在导入后覆盖架构与输出名
import train_bearing_cnn1d as _tb

_tb.MODEL_TYPE = "rf"
_tb.OUTPUT_MODEL_PATH = f"best_model_{_tb.MODEL_TYPE}.pth"
_tb.OUTPUT_MODEL_PATH_MIXED = f"model_mixed_{_tb.MODEL_TYPE}.pth"
_tb.OUTPUT_MODEL_PATH_REAL_ONLY = f"model_real_only_{_tb.MODEL_TYPE}.pth"


def eval_rf_only() -> int:
    """
    加载 RF checkpoint（rf_wrapper），在与训练相同划分下评估测试集。
    ``train_bearing_cnn1d.eval_only`` 面向 PyTorch state_dict，不适用于 RF。
    """
    import sys

    import torch
    from torch.utils.data import DataLoader

    from model import ARCH_RF

    _tb.set_seed(_tb.SEED)

    root = Path(_tb.DATA_ROOT)
    index, class_names = _tb.build_window_index_wrapped(root)
    device = torch.device(_tb.DEVICE if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)
    cache: dict[str, _tb.np.ndarray] = {}

    train_list, val_list, test_list = _tb.stratified_train_val_test(
        index, _tb.TEST_RATIO, _tb.TRAIN_IN_TRAINVAL, _tb.SEED
    )
    print(
        f"[信息] 样本数 — 训练: {len(train_list)} | 验证: {len(val_list)} | "
        f"测试: {len(test_list)} | 类别数: {num_classes}"
    )

    if len(test_list) == 0:
        print("[错误] 测试集为空，无法评估。", file=sys.stderr)
        return 1

    ds_test = _tb.BearingNpyWindowDataset(test_list, cache, _tb.NORMALIZE_PER_WINDOW)

    def collate(batch):
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    dl_test = DataLoader(
        ds_test,
        batch_size=_tb.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    out_path = _tb.SCRIPT_DIR / _tb.OUTPUT_MODEL_PATH
    if not out_path.is_file():
        print(f"[错误] 未找到模型文件: {out_path.resolve()}", file=sys.stderr)
        return 1

    try:
        ck = torch.load(str(out_path), map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(str(out_path), map_location=device)

    if ck.get("architecture") != ARCH_RF or "rf_wrapper" not in ck:
        print(
            "[错误] checkpoint 不是 RF（缺少 architecture=rf_bearing_v1 或 rf_wrapper）。"
            "若评估 CNN/LSTM/Transformer，请使用 train_bearing_cnn1d.py --eval-only。",
            file=sys.stderr,
        )
        return 1

    ck_names = ck.get("class_names")
    if ck_names is not None and list(ck_names) != list(class_names):
        print(
            "[警告] 当前 DATA_ROOT 下类别顺序与 checkpoint 中 class_names 不一致。"
        )

    nc_ck = int(ck.get("num_classes", num_classes))
    if nc_ck != num_classes:
        print(
            f"[错误] checkpoint num_classes={nc_ck} 与当前数据类别数 {num_classes} 不一致。",
            file=sys.stderr,
        )
        return 1

    model = ck["rf_wrapper"]
    test_acc = _tb.evaluate_accuracy(model, dl_test, device)
    cm = _tb.evaluate_confusion_matrix(model, dl_test, device, num_classes)
    print(f"\n[仅评估·RF] 测试集准确率: {test_acc:.4f}  |  文件: {out_path.name}\n")
    print("[测试集] 混淆矩阵（行=真实标签，列=预测标签）:")
    _tb.print_confusion_matrix(cm, class_names)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="轴承随机森林诊断训练（逻辑同 train_bearing_cnn1d.py，架构=RF+手工特征）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式说明：
  默认              双模型 train_two_models() — model_mixed_rf.pth + model_real_only_rf.pth
  --single          单模型 train()            — best_model_rf.pth
  --eval-only       仅评估 best_model_rf.pth（需 scikit-learn；与 CNN 脚本 eval 不通用）

数据根目录、GEN_DATA_ROOT、窗口与超参：编辑 train_bearing_cnn1d.py 顶部常量。
        """,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="不训练，仅加载 best_model_rf.pth 在测试集上评估",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="单模型训练（生成 best_model_rf.pth）",
    )
    args = parser.parse_args()

    if args.eval_only:
        return eval_rf_only()
    if args.single:
        return _tb.train()
    return _tb.train_two_models()


if __name__ == "__main__":
    raise SystemExit(main())
