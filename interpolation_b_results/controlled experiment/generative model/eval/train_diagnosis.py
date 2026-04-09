"""
train_diagnosis.py — 下游故障诊断分类训练与 TRTR/TSTR 对比实验模块

功能：
  - 训练 1D-CNN 分类器（Adam + CrossEntropyLoss）
  - 评估分类准确率
  - 实验 A (TRTR)：真实训练集 → 真实测试集
  - 实验 B (TSTR)：生成训练集 → 真实测试集
  - 实验 C (TRTR-Augment)：真实+生成 ConcatDataset 混合训练 → 真实测试；验证集仅 real_val
  - 终端打印对比结果

依赖：torch, scikit-learn (classification_report)
"""

from __future__ import annotations

import copy
from typing import Optional, Dict, Tuple, Any, Union

import numpy as np

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader, Subset

from sklearn.metrics import classification_report, f1_score

from .dataset import BearingSignalDataset, build_dataloader, strict_balance_dataset
from .models import build_cnn

TrainDatasetLike = Union[BearingSignalDataset, Subset, ConcatDataset]


# ---------------------------------------------------------------------------
# 训练 / 评估函数
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    训练一个 epoch，返回平均损失。
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for signals, labels in dataloader:
        signals: Tensor = signals.to(device)     # (B, 1, L)
        labels: Tensor = labels.to(device)       # (B,)

        optimizer.zero_grad()
        logits = model(signals)                  # (B, num_classes)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * signals.size(0)
        total_samples += signals.size(0)

    return total_loss / max(total_samples, 1)


def split_train_val(
    dataset: BearingSignalDataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[BearingSignalDataset, BearingSignalDataset]:
    """按类别分层划分训练/验证集；划分后在各自子集内做强制类别平衡（与 dataset.strict_balance_dataset 一致）。"""
    labels = dataset.labels.numpy()
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []

    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val].tolist())
        train_idx.extend(idx[n_val:].tolist())

    tr_ix = np.array(train_idx, dtype=np.int64)
    va_ix = np.array(val_idx, dtype=np.int64)
    gr = dataset.group_labels
    train_ds = BearingSignalDataset(
        dataset.signals[tr_ix],
        dataset.labels[tr_ix],
        group_labels=gr[tr_ix] if gr is not None else None,
    )
    val_ds = BearingSignalDataset(
        dataset.signals[va_ix],
        dataset.labels[va_ix],
        group_labels=gr[va_ix] if gr is not None else None,
    )
    train_ds = strict_balance_dataset(train_ds, seed=seed + 101)
    val_ds = strict_balance_dataset(val_ds, seed=seed + 103)
    return train_ds, val_ds


def _bearing_dataset_fixed_n(
    dataset: BearingSignalDataset,
    n_target: int,
    seed: int,
) -> BearingSignalDataset:
    """将数据集重采样为恰好 n_target 条；足够则无放回截断，不足则有放回抽取。"""
    if n_target < 1:
        raise ValueError("_bearing_dataset_fixed_n: n_target 必须 >= 1")
    rng = np.random.default_rng(seed)
    n = len(dataset)
    if n == 0:
        raise ValueError("_bearing_dataset_fixed_n: 空数据集无法重采样")
    idx = np.arange(n, dtype=np.int64)
    if n >= n_target:
        rng.shuffle(idx)
        sel = idx[:n_target]
    else:
        sel = rng.choice(idx, size=n_target, replace=True)
    sig = dataset.signals.numpy()[sel]
    lbl = dataset.labels.numpy()[sel]
    gl = dataset.group_labels
    g2 = gl[sel] if gl is not None else None
    return BearingSignalDataset(
        torch.from_numpy(sig.copy()),
        torch.from_numpy(lbl.copy()),
        group_labels=g2.copy() if g2 is not None else None,
    )


def _match_gen_tr_to_real_tr(
    real_tr: BearingSignalDataset,
    gen_tr: BearingSignalDataset,
    seed: int = 42,
) -> BearingSignalDataset:
    """
    实验 B (TSTR) 公平性：按类别使生成训练集与真实训练集条数一致（各类 n 与 real_tr 相同，总样本数相同）。
    某类生成样本不足时有放回抽样补齐。
    """
    rng = np.random.default_rng(seed)
    rl = real_tr.labels.numpy()
    gl = gen_tr.labels.numpy()
    classes = np.unique(rl)

    idx_parts: list[np.ndarray] = []
    for c in classes:
        n_need = int(np.sum(rl == c))
        if n_need == 0:
            continue
        g_idx = np.where(gl == c)[0]
        if len(g_idx) == 0:
            raise ValueError(
                f"TSTR 规模对齐失败：生成训练集中无类别 {int(c)}，无法与真实训练集一致"
            )
        if len(g_idx) >= n_need:
            rng.shuffle(g_idx)
            chosen = g_idx[:n_need]
        else:
            chosen = rng.choice(g_idx, size=n_need, replace=True)
        idx_parts.append(chosen)
    sel = np.concatenate(idx_parts)
    rng.shuffle(sel)

    sig = gen_tr.signals.numpy()[sel]
    lbl = gl[sel]
    ggrp = gen_tr.group_labels
    ng = ggrp[sel] if ggrp is not None else None
    return BearingSignalDataset(
        torch.from_numpy(sig.copy()),
        torch.from_numpy(lbl.copy()),
        group_labels=ng.copy() if ng is not None else None,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, int, int]:
    """
    在给定数据集上评估模型，返回 (accuracy, macro_f1, correct, total)。
    """
    model.eval()
    correct = 0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)
        logits = model(signals)
        preds = logits.argmax(dim=-1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    accuracy = correct / max(total, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return accuracy, float(macro_f1), correct, total


@torch.no_grad()
def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[list, list]:
    """
    获取模型在数据集上的全部预测结果与真实标签。

    返回
    ----
    all_preds : list[int]
    all_labels : list[int]
    """
    model.eval()
    all_preds = []
    all_labels = []

    for signals, labels in dataloader:
        signals = signals.to(device)
        logits = model(signals)
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    return all_preds, all_labels


# ---------------------------------------------------------------------------
# 完整训练流程
# ---------------------------------------------------------------------------

def train_classifier(
    train_dataset: TrainDatasetLike,
    test_dataset: TrainDatasetLike,
    num_classes: int,
    num_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    tag: str = "",
    val_dataset: Optional[TrainDatasetLike] = None,
) -> Tuple[nn.Module, float, float, float]:
    """
    完整训练流程：构建模型 → 训练 → 评估 → 返回模型与最终准确率。

    参数
    ----
    train_dataset : BearingSignalDataset
        训练数据集。
    test_dataset : BearingSignalDataset
        测试数据集。
    num_classes : int
        故障类别数。
    num_epochs : int
        训练轮数。
    batch_size : int
        批大小。
    lr : float
        学习率。
    device : torch.device | None
        设备，默认自动检测。
    verbose : bool
        是否打印训练过程。
    tag : str
        实验标签，用于打印区分。

    返回
    ----
    model : BearingCNN1D
        训练好的模型。
    best_acc : float
        测试集上的最佳准确率。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构建模型
    model = build_cnn(num_classes=num_classes, device=device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # 数据加载器
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, shuffle=False) if val_dataset is not None else None

    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        eval_loader = val_loader if val_loader is not None else test_loader
        acc, macro_f1, correct, total = evaluate(model, eval_loader, device)

        if acc > best_acc:
            best_acc = acc
            # 学术严谨性关键：保存验证集最优权重，避免最后 epoch 过拟合污染测试结果
            best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
            eval_name = "val" if val_loader is not None else "test"
            print(f"  [{tag}] Epoch {epoch:3d}/{num_epochs}  "
                  f"loss={avg_loss:.4f}  {eval_name}_acc={acc:.4f} ({correct}/{total})  "
                  f"{eval_name}_macro_f1={macro_f1:.4f}  "
                  f"best={best_acc:.4f}")

    # 加载最佳验证权重后再评估测试集，确保 TRTR/TSTR 公平可复现
    model.load_state_dict(best_model_wts)
    test_acc, test_f1, _, _ = evaluate(model, test_loader, device)
    return model, best_acc, test_acc, test_f1


# ---------------------------------------------------------------------------
# TRTR / TSTR 对比实验
# ---------------------------------------------------------------------------

def run_trtr_tstr(
    real_train: BearingSignalDataset,
    gen_train: BearingSignalDataset,
    real_test: BearingSignalDataset,
    num_classes: int,
    num_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    mix_aug_gen_samples: int = 8640,
) -> Dict[str, Any]:
    """
    运行 TRTR / TSTR / TRTR-Augment 对比实验。

    - 实验 A (TRTR)：真实数据训练 → 真实数据测试
    - 实验 B (TSTR)：生成数据训练 → 真实数据测试（训练集按类与 TRTR 的 real_tr 同规模）（训练阶段 ``gen_tr`` 按类重采样，与 ``real_tr`` 规模完全一致）
    - 实验 C (TRTR-Augment)：真实+生成 ConcatDataset 混合训练 → 真实测试；
      验证集固定为真实验证集 real_val（早停与最佳权重不接触生成样本）。
      混合集中生成分支默认重采样为 ``mix_aug_gen_samples`` 条（默认 8640），自 ``gen_tr`` 无放回或有放回抽取。

    参数
    ----
    real_train : BearingSignalDataset
        真实数据（训练集）。
    gen_train : BearingSignalDataset
        生成数据（训练集）。
    real_test : BearingSignalDataset
        真实数据（测试集）。
    num_classes : int
        故障类别数量。
    mix_aug_gen_samples : int
        实验 C 中参与混合训练的生成样本条数（默认 8640）。

    返回
    ----
    results : dict
        TRTR/TSTR/MixAug 准确率与 F1，以及用于 t-SNE 的 TRTR 模型 ``model_trtr``。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  模块二：下游故障诊断分类 — TRTR / TSTR / TRTR-Augment 对比实验")
    print("=" * 60)

    print(f"\n  设备: {device}")
    print(f"  类别数: {num_classes}")
    print(f"  训练轮数: {num_epochs}")
    print(f"  真实训练集大小: {len(real_train)}")
    print(f"  生成训练集大小: {len(gen_train)}")
    print(f"  真实测试集大小: {len(real_test)}")

    # ---------- 实验 A: TRTR ----------
    print(f"\n{'─'*50}")
    print("  实验 A (TRTR): 真实数据训练 → 真实数据测试")
    print(f"{'─'*50}")
    real_tr, real_val = split_train_val(real_train, val_ratio=0.1, seed=42)
    gen_tr, gen_val = split_train_val(gen_train, val_ratio=0.1, seed=42)

    print(f"  真实训练/验证/测试: {len(real_tr)}/{len(real_val)}/{len(real_test)}")
    print(f"  生成训练/验证/测试: {len(gen_tr)}/{len(gen_val)}/{len(real_test)}")

    # 实验 B 专用：生成训练集与真实训练集按类对齐条数（总样本数 = len(real_tr)）
    gen_tr_for_tstr = _match_gen_tr_to_real_tr(real_tr, gen_tr, seed=4243)
    assert len(gen_tr_for_tstr) == len(real_tr), "TSTR 训练规模应与 TRTR 一致"
    print(
        f"  [TSTR] 生成训练集已对齐至与真实训练集相同规模: {len(gen_tr_for_tstr)} = {len(real_tr)}（按类匹配）"
    )

    model_a, val_acc_a, test_acc_a, test_f1_a = train_classifier(
        real_tr, real_test,
        num_classes=num_classes,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        tag="TRTR",
        val_dataset=real_val,
    )

    # ---------- 实验 B: TSTR ----------
    print(f"\n{'─'*50}")
    print("  实验 B (TSTR): 生成数据训练 → 真实数据测试")
    print(f"{'─'*50}")
    model_b, val_acc_b, test_acc_b, test_f1_b = train_classifier(
        gen_tr_for_tstr, real_test,
        num_classes=num_classes,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        tag="TSTR",
        val_dataset=gen_val,
    )

    # ---------- 实验 C: TRTR-Augment (混合数据增强) ----------
    print(f"\n{'─'*50}")
    print("  实验 C (TRTR-Augment): 真实+生成混合训练 → 真实数据测试")
    print(f"{'─'*50}")
    gen_tr_mix = _bearing_dataset_fixed_n(gen_tr, mix_aug_gen_samples, seed=4242)
    mix_tr = ConcatDataset([real_tr, gen_tr_mix])
    print(f"  混合训练集: 真实 {len(real_tr)} + 生成(固定) {len(gen_tr_mix)} → 合计 {len(mix_tr)}")
    # 核心严谨性：val_dataset 必须只用 real_val，避免生成样本污染验证指标与早停
    model_c, val_acc_c, test_acc_c, test_f1_c = train_classifier(
        mix_tr, real_test,
        num_classes=num_classes,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        tag="MixAug",
        val_dataset=real_val,
    )

    # ---------- 结果汇总 ----------
    print(f"\n{'═'*60}")
    print("  TRTR / TSTR / TRTR-Augment 对比结果")
    print(f"{'═'*60}")
    print(f"  实验 A (TRTR) 最佳验证准确率: {val_acc_a:.4f}  | 测试准确率: {test_acc_a:.4f} ({test_acc_a*100:.2f}%)  | Macro-F1: {test_f1_a:.4f}")
    print(f"  实验 B (TSTR) 最佳验证准确率: {val_acc_b:.4f}  | 测试准确率: {test_acc_b:.4f} ({test_acc_b*100:.2f}%)  | Macro-F1: {test_f1_b:.4f}")
    print(f"  实验 C (TRTR-Augment) 最佳验证准确率: {val_acc_c:.4f}  | 测试准确率: {test_acc_c:.4f} ({test_acc_c*100:.2f}%)  | Macro-F1: {test_f1_c:.4f}  [val=仅真实]")
    diff = test_acc_a - test_acc_b
    print(f"  差距 (TRTR - TSTR): {diff:+.4f}")
    aug_vs_trtr = test_acc_c - test_acc_a
    print(f"  差距 (TRTR-Augment - TRTR): {aug_vs_trtr:+.4f}  (小样本增强价值指标)")
    if abs(diff) < 0.05:
        print("  → 生成数据质量优秀，TSTR 接近 TRTR 基准！✓")
    elif diff > 0:
        print("  → 生成数据质量有提升空间，TSTR 略低于 TRTR。")
    else:
        print("  → TSTR 超过 TRTR，可能存在数据泄露或过拟合，需注意。")
    print()

    # 可选：打印详细分类报告
    test_loader = build_dataloader(real_test, batch_size=batch_size, shuffle=False)

    print("  [TRTR] 详细分类报告:")
    preds_a, labels_a = get_predictions(model_a, test_loader, device)
    print(classification_report(labels_a, preds_a, digits=4))

    print("  [TSTR] 详细分类报告:")
    preds_b, labels_b = get_predictions(model_b, test_loader, device)
    print(classification_report(labels_b, preds_b, digits=4))

    print("  [TRTR-Augment] 详细分类报告:")
    preds_c, labels_c = get_predictions(model_c, test_loader, device)
    print(classification_report(labels_c, preds_c, digits=4))

    return {
        "TRTR_acc": test_acc_a,
        "TSTR_acc": test_acc_b,
        "MixAug_acc": test_acc_c,
        "TRTR_f1": test_f1_a,
        "TSTR_f1": test_f1_b,
        "MixAug_f1": test_f1_c,
        "model_trtr": model_a,
    }


# 别名：与 run_trtr_tstr 相同，含实验 A/B/C
run_trtr_tstr_aug = run_trtr_tstr

