"""
train_diagnosis.py — 下游故障诊断分类训练与 TRTR/TSTR 对比实验模块

功能：
  - 训练 1D-CNN 分类器（Adam + CrossEntropyLoss）
  - 评估分类准确率
  - 实验 A (TRTR)：真实训练集 → 真实测试集
  - 实验 B (TSTR)：生成训练集 → 真实测试集
  - 终端打印对比结果

依赖：torch, scikit-learn (classification_report)
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from .dataset import BearingSignalDataset, build_dataloader
from .models import BearingCNN1D, build_cnn


# ---------------------------------------------------------------------------
# 训练 / 评估函数
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: BearingCNN1D,
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


@torch.no_grad()
def evaluate(
    model: BearingCNN1D,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, int, int]:
    """
    在给定数据集上评估模型，返回 (accuracy, correct, total)。
    """
    model.eval()
    correct = 0
    total = 0

    for signals, labels in dataloader:
        signals = signals.to(device)
        labels = labels.to(device)
        logits = model(signals)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = correct / max(total, 1)
    return accuracy, correct, total


@torch.no_grad()
def get_predictions(
    model: BearingCNN1D,
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
    train_dataset: BearingSignalDataset,
    test_dataset: BearingSignalDataset,
    num_classes: int,
    num_epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: Optional[torch.device] = None,
    verbose: bool = True,
    tag: str = "",
) -> Tuple[BearingCNN1D, float]:
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
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 数据加载器
    train_loader = build_dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = build_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc, correct, total = evaluate(model, test_loader, device)

        if acc > best_acc:
            best_acc = acc

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == num_epochs):
            print(f"  [{tag}] Epoch {epoch:3d}/{num_epochs}  "
                  f"loss={avg_loss:.4f}  acc={acc:.4f} ({correct}/{total})  "
                  f"best={best_acc:.4f}")

    return model, best_acc


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
) -> Dict[str, float]:
    """
    运行 TRTR / TSTR 对比实验。

    - 实验 A (TRTR)：真实数据训练 → 真实数据测试
    - 实验 B (TSTR)：生成数据训练 → 真实数据测试

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

    返回
    ----
    results : dict
        {"TRTR_acc": float, "TSTR_acc": float}
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  模块二：下游故障诊断分类 — TRTR / TSTR 对比实验")
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
    model_a, acc_a = train_classifier(
        real_train, real_test,
        num_classes=num_classes,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        tag="TRTR",
    )

    # ---------- 实验 B: TSTR ----------
    print(f"\n{'─'*50}")
    print("  实验 B (TSTR): 生成数据训练 → 真实数据测试")
    print(f"{'─'*50}")
    model_b, acc_b = train_classifier(
        gen_train, real_test,
        num_classes=num_classes,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        tag="TSTR",
    )

    # ---------- 结果汇总 ----------
    print(f"\n{'═'*60}")
    print("  TRTR / TSTR 对比结果")
    print(f"{'═'*60}")
    print(f"  实验 A (TRTR) 最佳测试准确率: {acc_a:.4f}  ({acc_a*100:.2f}%)")
    print(f"  实验 B (TSTR) 最佳测试准确率: {acc_b:.4f}  ({acc_b*100:.2f}%)")
    diff = acc_a - acc_b
    print(f"  差距 (TRTR - TSTR): {diff:+.4f}")
    if abs(diff) < 0.05:
        print("  → 生成数据质量优秀，TSTR 接近 TRTR 基准！✓")
    elif diff > 0:
        print("  → 生成数据质量有提升空间，TSTR 略低于 TRTR。")
    else:
        print("  → TSTR 超过 TRTR，可能存在数据泄露或过拟合，需注意。")
    print()

    # 可选：打印详细分类报告
    try:
        from sklearn.metrics import classification_report
        test_loader = build_dataloader(real_test, batch_size=batch_size, shuffle=False)

        print("  [TRTR] 详细分类报告:")
        preds_a, labels_a = get_predictions(model_a, test_loader, device)
        print(classification_report(labels_a, preds_a, digits=4))

        print("  [TSTR] 详细分类报告:")
        preds_b, labels_b = get_predictions(model_b, test_loader, device)
        print(classification_report(labels_b, preds_b, digits=4))
    except ImportError:
        print("  (sklearn 未安装，跳过详细分类报告)")

    return {"TRTR_acc": acc_a, "TSTR_acc": acc_b}

