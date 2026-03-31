#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承 / HIL 1D CNN 多分类训练脚本（NumPy 嵌套目录数据集）。

数据结构（与资源管理器一致）:
  D:\\HIL_train\\
    IF0.4\\              ← 一级文件夹名 = 类别名（标签按名称排序映射为 0..K-1）
      load_0\\
        rpm_1000\\
          filtered_0.npy
          filtered_1.npy
          ...
    <其它类别>\\  ...

递归收集每类下所有 .npy，读成一维序列后按 WINDOW_SIZE / STRIDE 滑窗；
每个 .npy 文件最多取 MAX_WINDOWS_PER_FILE 个窗，减轻过拟合。

划分比例：先按类分层划出测试集 TEST_RATIO，余下部分按 TRAIN_IN_TRAINVAL（8:2）分为训练/验证。
早停：以验证集准确率监控；测试集仅在训练结束后评估一次。

依赖: pip install torch numpy
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# =============================================================================
# 全局配置（直接修改；IDE 点击 Run 即可）
# =============================================================================

DATA_ROOT = r"D:\HIL_train"
WINDOW_SIZE = 1024  # 与 Speedgoat / receive_udp_hil 推理窗长一致
STRIDE = 1024  # 滑窗步长；1024 与在线非重叠分片一致；可改小以增加窗数

# 输出到脚本同目录，供 receive_udp_hil.py 加载（按「验证集最优」保存，部署更合理）
OUTPUT_MODEL_PATH = "best_model.pth"
EXPECTED_NUM_CLASSES = 10  # 一级类别文件夹数不等于此时仅警告，仍以实际为准

# 每个 .npy 文件最多使用的滑窗样本数（避免单文件扫满导致过拟合）
MAX_WINDOWS_PER_FILE = 20

# 分层划分：测试集占比；剩余样本中训练集占比（训练:验证 = 8:2 即 0.8:0.2）
TEST_RATIO = 0.2
TRAIN_IN_TRAINVAL = 0.8  # 验证集占剩余部分的比例为 1 - TRAIN_IN_TRAINVAL

# 早停：监控**验证集**准确率，连续若干 epoch 未超过历史最佳则停止（测试集不参与早停）
EARLY_STOP_PATIENCE = 10

BATCH_SIZE = 32
EPOCHS = 200  # 上限；通常会因早停提前结束
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42
DEVICE = "cuda"  # 无 GPU 时自动改 cpu

# 训练时是否对每个窗口做标准化（零均值单位方差），推理端 receive 需同样处理时再对齐
NORMALIZE_PER_WINDOW = True

# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_signal_from_npy(npy_path: Path) -> np.ndarray:
    """
    从 .npy 加载一维振动序列（支持任意嵌套路径下的 filtered_*.npy）。
    数值转为 float64；多维数组 squeeze 后展平为 1D。
    """
    try:
        arr = np.load(str(npy_path), allow_pickle=False)
    except Exception as e:
        raise ValueError(f"np.load 失败: {e}") from e

    if arr.dtype == object:
        raise ValueError("不支持 object 数组（请保存为数值型 .npy）")

    arr = np.asarray(arr, dtype=np.float64)
    arr = np.squeeze(arr)
    if arr.ndim == 0:
        raise ValueError(f"标量数组，shape={arr.shape}")
    if arr.ndim > 1:
        arr = arr.reshape(-1)
    return arr.reshape(-1)


def collect_class_folders(root: Path) -> list[tuple[str, int]]:
    """返回 [(文件夹名, 类别索引), ...]，按文件夹名字符串排序以保证标签稳定。"""
    if not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {root}")

    subs = [p for p in root.iterdir() if p.is_dir()]
    subs.sort(key=lambda p: p.name)
    if len(subs) == 0:
        raise FileNotFoundError(f"{root} 下没有子文件夹（每个子文件夹应为一类）")

    if len(subs) != EXPECTED_NUM_CLASSES:
        print(
            f"[警告] 子文件夹数量={len(subs)}，与 EXPECTED_NUM_CLASSES={EXPECTED_NUM_CLASSES} 不一致，将按实际 {len(subs)} 类训练。"
        )

    return [(p.name, i) for i, p in enumerate(subs)]


class BearingNpyWindowDataset:
    """简易数据集：__getitem__ 读 .npy 切片（带文件级缓存）。"""

    def __init__(
        self,
        index: list[tuple[Path, int, int]],
        cache: dict[str, np.ndarray],
        normalize: bool,
    ):
        self.index = index
        self._cache = cache
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.index)

    def _get_signal(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key not in self._cache:
            self._cache[key] = load_signal_from_npy(path)
        return self._cache[key]

    def __getitem__(self, i: int):
        path, start, label = self.index[i]
        sig = self._get_signal(path)
        w = sig[start : start + WINDOW_SIZE].astype(np.float32).copy()
        if self.normalize:
            m = float(w.mean())
            s = float(w.std()) + 1e-6
            w = (w - m) / s
        # [1, L] for conv1d
        import torch

        x = torch.from_numpy(w).unsqueeze(0)
        y = label
        return x, y


def _split_counts_one_class(n: int, test_ratio: float, train_frac_of_rest: float) -> tuple[int, int, int]:
    """
    将单类样本数 n 拆成 (n_train, n_val, n_test)，满足 n_train+n_val+n_test=n。
    尽量满足：测试约 test_ratio；余下训练:验证 ≈ train_frac : (1-train_frac)。
    小样本：n<3 不设独立测试；n==2 仅 train+val；n==1 全部进训练。
    当 n>=3 且比例舍入为 0 但 test_ratio>0 时，每类至少 1 个测试样本，避免全局测试集过小。
    """
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_test_tgt = int(round(n * test_ratio))
    n_test = min(max(0, n_test_tgt), n - 2)
    if n_test == 0 and test_ratio > 1e-9:
        n_test = min(1, n - 2)
    rest = n - n_test
    if rest < 2:
        n_test = max(0, n - 2)
        rest = n - n_test

    n_val_tgt = int(round(rest * (1.0 - train_frac_of_rest)))
    n_val = max(1, min(rest - 1, n_val_tgt))
    n_train = rest - n_val
    if n_train < 1:
        n_train, n_val = 1, rest - 1
    assert n_train + n_val + n_test == n
    return n_train, n_val, n_test


def stratified_train_val_test(
    index: list[tuple[Path, int, int]],
    test_ratio: float,
    train_frac_of_rest: float,
    seed: int,
) -> tuple[list, list, list]:
    """
    按类别分层划分：
    1) 每类先划出约 test_ratio 的测试样本（n>=3 时至少 1 个测试；n<3 该类不进测试集）；
    2) 剩余部分按 train_frac_of_rest : (1-train_frac_of_rest) 分为训练 / 验证。
    """
    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for item in index:
        by_label.setdefault(item[2], []).append(item)

    train_idx: list = []
    val_idx: list = []
    test_idx: list = []

    for _label in sorted(by_label.keys()):
        items = by_label[_label][:]
        rng.shuffle(items)
        n = len(items)
        n_train, n_val, n_test = _split_counts_one_class(n, test_ratio, train_frac_of_rest)

        test_part = items[:n_test]
        mid = items[n_test : n_test + n_val]
        train_part = items[n_test + n_val :]
        assert len(test_part) == n_test and len(mid) == n_val and len(train_part) == n_train

        train_idx.extend(train_part)
        val_idx.extend(mid)
        test_idx.extend(test_part)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def evaluate_accuracy(model, data_loader, device) -> float:
    """返回该 DataLoader 上的分类准确率。"""
    import torch

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).argmax(dim=-1)
            correct += int((pred == yb).sum().item())
            total += yb.numel()
    return correct / max(total, 1)


def evaluate_confusion_matrix(
    model, data_loader, device, num_classes: int
) -> np.ndarray:
    """在 data_loader 上统计混淆矩阵，形状 [num_classes, num_classes]，行=真实标签，列=预测标签。"""
    import torch

    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).argmax(dim=-1)
            yt = yb.cpu().numpy().astype(np.int64)
            pr = pred.cpu().numpy().astype(np.int64)
            for t, p in zip(yt.flat, pr.flat):
                if 0 <= t < num_classes and 0 <= p < num_classes:
                    cm[t, p] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """在控制台打印混淆矩阵（行=真实，列=预测）。"""
    n = cm.shape[0]
    if cm.shape != (n, n) or len(class_names) != n:
        print(f"[警告] 混淆矩阵形状 {cm.shape} 与类别数 {len(class_names)} 不一致，跳过格式化打印。")
        print(cm)
        return

    col_w = max(5, len(str(int(cm.max()))) + 1)
    label_w = min(max(len(nm) for nm in class_names), 14)
    head = "真实 \\ 预测".ljust(label_w + 5) + "".join(f"{j:>{col_w}}" for j in range(n))
    print(head)
    for i in range(n):
        lbl = (class_names[i][:label_w] + " " * label_w)[:label_w]
        print(f"{lbl} {i:2d}  " + "".join(f"{cm[i, j]:>{col_w}}" for j in range(n)))
    print("\n列 j 的类别名:", ", ".join(f"{j}→{class_names[j]}" for j in range(n)))

    total = int(cm.sum())
    if total > 0:
        diag = int(np.trace(cm))
        print(f"\n对角线之和 / 总样本 = {diag} / {total} = {diag / total:.4f}")
        print("各类召回率 (行归一化):")
        for i in range(n):
            row_sum = int(cm[i].sum())
            r = cm[i, i] / row_sum if row_sum > 0 else 0.0
            print(f"  [{i}] {class_names[i]}: {r:.4f}  (n={row_sum})")


def train() -> int:
    set_seed(SEED)
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("[错误] 需要 PyTorch: pip install torch", file=sys.stderr)
        return 1

    from bearing_models import BEARING_CNN_ARCH, BearingCNN1D, count_conv1d_layers

    root = Path(DATA_ROOT)
    index, class_names = build_window_index_wrapped(root)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and DEVICE == "cuda":
        print("[警告] CUDA 不可用，使用 CPU 训练。")

    num_classes = len(class_names)
    cache: dict[str, np.ndarray] = {}

    train_list, val_list, test_list = stratified_train_val_test(
        index, TEST_RATIO, TRAIN_IN_TRAINVAL, SEED
    )
    print(
        f"[信息] 样本数 — 训练: {len(train_list)} | 验证: {len(val_list)} | 测试: {len(test_list)} | 类别数: {num_classes}"
    )
    print(
        f"[信息] 划分说明: 每类先取约 {TEST_RATIO:.0%} 作测试集；"
        f"余下按 {TRAIN_IN_TRAINVAL:.0%}:{1 - TRAIN_IN_TRAINVAL:.0%} 分为训练/验证。"
    )
    if len(val_list) == 0:
        print(
            "[错误] 验证集为空（每类仅 1 个窗口时会出现）。请增大 MAX_WINDOWS_PER_FILE、减小 STRIDE 或合并类别。",
            file=sys.stderr,
        )
        return 1
    if len(test_list) == 0:
        print(
            "[错误] 测试集为空（每类样本 <3 时无法分层出测试）。请增大 MAX_WINDOWS_PER_FILE、减小 STRIDE 或降低 TEST_RATIO。",
            file=sys.stderr,
        )
        return 1

    ds_train = BearingNpyWindowDataset(train_list, cache, NORMALIZE_PER_WINDOW)
    ds_val = BearingNpyWindowDataset(val_list, cache, NORMALIZE_PER_WINDOW)
    ds_test = BearingNpyWindowDataset(test_list, cache, NORMALIZE_PER_WINDOW)

    # DataLoader 需要 collate
    def collate(batch):
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    dl_train = DataLoader(
        ds_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    model = BearingCNN1D(num_classes=num_classes).to(device)
    n_conv = count_conv1d_layers(model)
    print(
        f"[模型] BearingCNN1D 中 **Conv1d 卷积层数 = {n_conv}**（实时性：层数/通道越少通常越快；当前为 4 层卷积 + 分类头 Linear）。"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
    crit = nn.CrossEntropyLoss()

    # 以验证集准确率为早停与保存依据；测试集不参与训练期决策，仅在最后评估一次
    best_val_acc = -1.0
    patience_cnt = 0
    out_path = SCRIPT_DIR / OUTPUT_MODEL_PATH

    stopped_early = False
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n_seen += xb.size(0)
        sched.step()

        train_loss = total_loss / max(n_seen, 1)
        val_acc = evaluate_accuracy(model, dl_val, device)

        improved = val_acc > best_val_acc + 1e-7
        if improved:
            best_val_acc = val_acc
            patience_cnt = 0
            ckpt = {
                "architecture": BEARING_CNN_ARCH,
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "window_size": WINDOW_SIZE,
                "normalize_per_window": NORMALIZE_PER_WINDOW,
                "best_val_acc": best_val_acc,
                "epoch": epoch + 1,
            }
            torch.save(ckpt, str(out_path))
            print(
                f"Epoch {epoch+1:03d}/{EPOCHS} | train_loss={train_loss:.4f} | "
                f"val_acc={val_acc:.4f} | 【保存】验证集最优 -> {out_path.name}"
            )
        else:
            patience_cnt += 1
            print(
                f"Epoch {epoch+1:03d}/{EPOCHS} | train_loss={train_loss:.4f} | "
                f"val_acc={val_acc:.4f} | 早停计数 {patience_cnt}/{EARLY_STOP_PATIENCE}"
            )

        if patience_cnt >= EARLY_STOP_PATIENCE:
            print(f"[早停] 验证集准确率已连续 {EARLY_STOP_PATIENCE} 个 epoch 未提升，停止训练。")
            stopped_early = True
            break

    if not stopped_early:
        print(f"[信息] 已达最大 epoch={EPOCHS}，未触发早停。")

    # 加载「验证集最优」checkpoint；测试集仅此时评估一次（无偏泛化估计）
    if out_path.is_file():
        try:
            ck = torch.load(str(out_path), map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(str(out_path), map_location=device)
        model.load_state_dict(ck["state_dict"])
    final_test_acc = evaluate_accuracy(model, dl_test, device)
    cm = evaluate_confusion_matrix(model, dl_test, device, num_classes)
    print(
        f"\n[最终评估] 使用「验证集最优」权重在**测试集**上的准确率（未参与早停）: {final_test_acc:.4f}\n"
        f"[信息] 训练过程中验证集最佳 val_acc={best_val_acc:.4f}\n"
        f"[完成] 模型已保存: {out_path.resolve()}"
    )
    print("\n[测试集] 混淆矩阵（行=真实标签，列=预测标签）:")
    print_confusion_matrix(cm, class_names)
    return 0


def build_window_index_wrapped(root: Path) -> tuple[list[tuple[Path, int, int]], list[str]]:
    """
    扫描 DATA_ROOT 下各类一级子文件夹，递归收集所有 .npy，生成窗口索引 (npy路径, 起点, 标签)。
    每个 .npy 最多保留 MAX_WINDOWS_PER_FILE 个滑窗（随机下采样，可复现）。
    """
    class_list = collect_class_folders(root)
    class_names = [name for name, _ in class_list]

    rng = random.Random(SEED)
    index: list[tuple[Path, int, int]] = []
    for folder_name, label in class_list:
        folder = root / folder_name
        npy_paths = sorted(folder.rglob("*.npy"))
        for npy_path in npy_paths:
            try:
                sig = load_signal_from_npy(npy_path)
            except Exception as e:
                print(f"[跳过] {npy_path}: {e}", file=sys.stderr)
                continue
            n = int(sig.shape[0])
            if n < WINDOW_SIZE:
                print(f"[跳过] {npy_path}: 长度 {n} < WINDOW_SIZE={WINDOW_SIZE}", file=sys.stderr)
                continue
            starts = list(range(0, n - WINDOW_SIZE + 1, STRIDE))
            if len(starts) > MAX_WINDOWS_PER_FILE:
                starts = rng.sample(starts, MAX_WINDOWS_PER_FILE)
            for start in starts:
                index.append((npy_path, start, label))

    if not index:
        raise RuntimeError("没有可用窗口（请检查 DATA_ROOT 下是否有足够长的 .npy）")

    print(
        f"[信息] 共 {len(class_names)} 类: {class_names} | "
        f"窗口总数: {len(index)}（每 .npy 最多 {MAX_WINDOWS_PER_FILE} 窗 | WINDOW={WINDOW_SIZE} STRIDE={STRIDE}）"
    )
    return index, class_names


def eval_only() -> int:
    """
    仅加载 OUTPUT_MODEL_PATH，在与训练相同划分（SEED / TEST_RATIO / …）下评估测试集并打印混淆矩阵。
    """
    set_seed(SEED)
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("[错误] 需要 PyTorch: pip install torch", file=sys.stderr)
        return 1

    from bearing_models import BearingCNN1D

    root = Path(DATA_ROOT)
    index, class_names = build_window_index_wrapped(root)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)
    cache: dict[str, np.ndarray] = {}

    train_list, val_list, test_list = stratified_train_val_test(
        index, TEST_RATIO, TRAIN_IN_TRAINVAL, SEED
    )
    print(
        f"[信息] 样本数 — 训练: {len(train_list)} | 验证: {len(val_list)} | 测试: {len(test_list)} | 类别数: {num_classes}"
    )

    if len(test_list) == 0:
        print("[错误] 测试集为空，无法评估。", file=sys.stderr)
        return 1

    ds_test = BearingNpyWindowDataset(test_list, cache, NORMALIZE_PER_WINDOW)

    def collate(batch):
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    dl_test = DataLoader(
        ds_test,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )

    out_path = SCRIPT_DIR / OUTPUT_MODEL_PATH
    if not out_path.is_file():
        print(f"[错误] 未找到模型文件: {out_path.resolve()}", file=sys.stderr)
        return 1

    try:
        ck = torch.load(str(out_path), map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(str(out_path), map_location=device)

    ck_names = ck.get("class_names")
    if ck_names is not None and list(ck_names) != list(class_names):
        print(
            "[警告] 当前 DATA_ROOT 下类别顺序与 checkpoint 中 class_names 不一致，"
            "标签语义可能与训练时不一致；请确认未增删/重命名一级文件夹。"
        )

    nc_ck = int(ck.get("num_classes", num_classes))
    if nc_ck != num_classes:
        print(
            f"[错误] checkpoint num_classes={nc_ck} 与当前数据类别数 {num_classes} 不一致。",
            file=sys.stderr,
        )
        return 1

    model = BearingCNN1D(num_classes=num_classes).to(device)
    model.load_state_dict(ck["state_dict"])

    test_acc = evaluate_accuracy(model, dl_test, device)
    cm = evaluate_confusion_matrix(model, dl_test, device, num_classes)
    print(f"\n[仅评估] 测试集准确率: {test_acc:.4f}  |  权重: {out_path.name}\n")
    print("[测试集] 混淆矩阵（行=真实标签，列=预测标签）:")
    print_confusion_matrix(cm, class_names)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="轴承 1D CNN 训练 / 仅测试集评估")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="不训练，仅加载 best_model.pth（OUTPUT_MODEL_PATH）在测试集上评估并打印混淆矩阵",
    )
    args = parser.parse_args()
    raise SystemExit(eval_only() if args.eval_only else train())
