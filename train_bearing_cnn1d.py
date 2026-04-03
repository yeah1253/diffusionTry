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

生成数据根目录结构与真实数据相同（GEN_DATA_ROOT）：
  D:\\HIL_gen\\
    IF0.4\\
      load_0\\
        rpm_1000\\
          filtered_0.npy ...

默认运行模式（train_two_models）生成两个 .pth 文件供 receive_udp_hil.py 调用：
  model_mixed.pth     — 模型1：按工况混合真实+生成数据（核心区 1:1，边缘区 1:9）
  model_real_only.pth — 模型2：仅真实数据，数量与模型1中真实数据一致

使用 --single 可回退到原始单模型训练（保存 best_model.pth）。

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

DATA_ROOT = r"D:\data\轴承数据集"
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
# 双模型训练配置（train_two_models 模式）
# =============================================================================

# 生成数据根目录（结构与 DATA_ROOT 一致：GEN_DATA_ROOT/类别/load_X/rpm_Y/filtered_*.npy）
GEN_DATA_ROOT = str(Path(__file__).resolve().parent / "generated_samples_cvae")

# 两个输出模型文件名（保存在脚本同目录）
OUTPUT_MODEL_PATH_MIXED = "model_mixed.pth"       # 模型1：真实+生成混合
OUTPUT_MODEL_PATH_REAL_ONLY = "model_real_only.pth"  # 模型2：仅真实数据

# 工况区域定义（用于计算混合比例）
# 核心区：CORE_RPM ± 任意，负载 [CORE_LOAD_LO, CORE_LOAD_HI]
CORE_RPM = 1800
CORE_LOAD_LO = 20
CORE_LOAD_HI = 40
# 边缘区：RPM ≤ EDGE_RPM_LO 或 RPM ≥ EDGE_RPM_HI；负载 ≤ EDGE_LOAD_LO 或 ≥ EDGE_LOAD_HI
EDGE_RPM_LO = 1000
EDGE_RPM_HI = 2500
EDGE_LOAD_LO = 0
EDGE_LOAD_HI = 60

# 真实数据占比：
#   核心区 1:1  → 0.5（64 真实 + 64 生成 = 128）
#   边缘区 0:10 → 0.0（  0 真实 + 128 生成 = 128）
#   过渡区按核心度线性插值（真实数量从 64 衰减到 0）
REAL_FRAC_CORE = 0.5
REAL_FRAC_EDGE = 0.0

# 每个工况的固定总样本数（128 = 核心区 64 真实 + 64 生成）
TOTAL_WINDOWS_PER_CONDITION = 128

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


def _load_signal_from_mat(mat_path: Path) -> np.ndarray:
    """
    从 .mat 文件加载一维振动信号，与 train_1d_vibration.py line 74 行为一致：
        signal = data['Signal']['y_values'][0, 0]['values'].item()[:, 0]

    Signal.y_values.values 为 (N, C) double，取第一通道（列 0）。
    自动兼容两种 scipy 返回形式：
      - (1, 1) object array  → .item() 解包后得 (N, C) ndarray
      - 直接  (N, C) ndarray  → 直接使用
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy 未安装，请运行: pip install scipy")

    data = loadmat(str(mat_path))
    arr = data['Signal']['y_values'][0, 0]['values']

    # scipy 有时将矩阵包在 (1,1) object array 中，需要 .item() 解包
    if arr.dtype == object and arr.shape == (1, 1):
        mat = arr.item()
    else:
        mat = arr

    if mat.ndim == 2:
        signal = mat[:, 0].astype(np.float64)
    else:
        signal = mat.ravel().astype(np.float64)

    if signal.size == 0:
        raise ValueError(f"信号长度为 0: {mat_path.name}")
    return signal


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
            if path.suffix.lower() == ".mat":
                self._cache[key] = _load_signal_from_mat(path)
            else:
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


# =============================================================================
# 双模型训练核心函数
# =============================================================================


def _rpm_load_from_two_ints(a: int, b: int) -> tuple[int, int]:
    """按数值大小区分 rpm（大）与 load（小），返回 (rpm, load)。"""
    return (a, b) if a >= b else (b, a)


def _parse_condition_from_path(p: Path) -> tuple[int | None, int | None]:
    """
    从路径中解析 (rpm, load)，按优先级依次尝试三种格式：

    格式 A（目录前缀）：    .../load_20/rpm_1800/...
    格式 C（.mat 文件名）： .../IF0.2/IF0.2 1000 0.mat  → 取末尾两个整数
    格式 B（两整数目录名）：.../class_0/20 1000/...      → 目录名恰好两整数

    rpm/load 均用数值大小区分（rpm ≥ 1000，load ≤ 60）。
    """
    rpm: int | None = None
    load: int | None = None

    # 格式 A
    for part in p.parts:
        pl = part.lower()
        if pl.startswith("load_"):
            try:
                load = int(part[5:])
            except ValueError:
                pass
        elif pl.startswith("rpm_"):
            try:
                rpm = int(part[4:])
            except ValueError:
                pass
    if rpm is not None and load is not None:
        return rpm, load

    # 格式 C：.mat 文件名 "<class> <rpm> <load>.mat"，取末尾两整数
    if p.suffix.lower() == ".mat":
        ints = [int(t) for t in p.stem.split() if t.lstrip("-").isdigit()]
        if len(ints) >= 2:
            return _rpm_load_from_two_ints(ints[-2], ints[-1])

    # 格式 B：目录名恰好为两个整数（生成数据 "20 1000" 格式）
    for part in p.parts:
        tokens = part.strip().split()
        if len(tokens) == 2:
            try:
                return _rpm_load_from_two_ints(int(tokens[0]), int(tokens[1]))
            except ValueError:
                pass

    return rpm, load


def _compute_real_count(rpm: int, load: int) -> int:
    """
    基于工况 (rpm, load) 计算该工况下应采样的真实数据条数。

    规则：
      核心区（CORE_RPM, load ∈ [CORE_LOAD_LO, CORE_LOAD_HI]）
          → TOTAL_WINDOWS_PER_CONDITION × REAL_FRAC_CORE = 64 条真实
      边缘区（rpm = EDGE_RPM_LO/EDGE_RPM_HI 或 load = EDGE_LOAD_LO/EDGE_LOAD_HI）
          → TOTAL_WINDOWS_PER_CONDITION × REAL_FRAC_EDGE = 0 条真实（纯生成）
      过渡区
          → 按归一化"核心度"线性插值：真实条数 = round(TOTAL × coreness × REAL_FRAC_CORE)
            保证结果在 [0, round(TOTAL × REAL_FRAC_CORE)] 之间，且每个工况总数保持 TOTAL 不变。

    生成数据条数 = TOTAL_WINDOWS_PER_CONDITION - real_count
    """
    # RPM 核心度：CORE_RPM 处为 1.0，EDGE_RPM_LO/EDGE_RPM_HI 处为 0.0，线性过渡
    if rpm <= CORE_RPM:
        rpm_core = (rpm - EDGE_RPM_LO) / max(CORE_RPM - EDGE_RPM_LO, 1)
    else:
        rpm_core = (EDGE_RPM_HI - rpm) / max(EDGE_RPM_HI - CORE_RPM, 1)
    rpm_core = max(0.0, min(1.0, rpm_core))

    # 负载核心度：[CORE_LOAD_LO, CORE_LOAD_HI] 内为 1.0，EDGE_LOAD_LO/HI 处为 0.0
    if CORE_LOAD_LO <= load <= CORE_LOAD_HI:
        load_core = 1.0
    elif load < CORE_LOAD_LO:
        load_core = (load - EDGE_LOAD_LO) / max(CORE_LOAD_LO - EDGE_LOAD_LO, 1)
    else:
        load_core = (EDGE_LOAD_HI - load) / max(EDGE_LOAD_HI - CORE_LOAD_HI, 1)
    load_core = max(0.0, min(1.0, load_core))

    # 综合核心度：两个维度算术平均，再映射到 [0, REAL_FRAC_CORE] 后乘以 TOTAL
    coreness = (rpm_core + load_core) / 2.0
    real_frac = REAL_FRAC_EDGE + (REAL_FRAC_CORE - REAL_FRAC_EDGE) * coreness
    real_count = round(TOTAL_WINDOWS_PER_CONDITION * real_frac)
    # 严格约束在 [0, TOTAL] 内
    return max(0, min(TOTAL_WINDOWS_PER_CONDITION, real_count))


def _collect_windows_by_condition(
    class_dir: Path,
    label: int,
    rng: random.Random,
) -> dict[tuple[int, int], list[tuple[Path, int, int]]]:
    """
    扫描一个类别目录下所有 .npy 和 .mat，按 (load, rpm) 工况键分组。
    每个文件最多取 MAX_WINDOWS_PER_FILE 个滑窗。

    .mat  — SDUST/Simulink 长序列，按 STRIDE 滑窗（真实数据）
    .npy  — 单样本，通常给出 1 个 1024 点窗（生成数据）

    返回 {(load, rpm): [(file_path, start, label), ...]}
    """
    by_cond: dict[tuple[int, int], list] = {}
    n_loaded = 0
    n_skipped = 0

    all_files = sorted(
        list(class_dir.rglob("*.npy")) + list(class_dir.rglob("*.mat"))
    )

    for file_path in all_files:
        rpm, load = _parse_condition_from_path(file_path)
        cond_key = (load if load is not None else -1, rpm if rpm is not None else -1)

        try:
            if file_path.suffix.lower() == ".mat":
                sig = _load_signal_from_mat(file_path)
            else:
                sig = load_signal_from_npy(file_path)
        except Exception as e:
            print(f"  [跳过加载] {file_path.name}: {e}")
            n_skipped += 1
            continue

        n = int(sig.shape[0])
        if n < WINDOW_SIZE:
            print(f"  [跳过长度] {file_path.name}: {n} < {WINDOW_SIZE}")
            n_skipped += 1
            continue

        starts = list(range(0, n - WINDOW_SIZE + 1, STRIDE))
        if len(starts) > MAX_WINDOWS_PER_FILE:
            starts = rng.sample(starts, MAX_WINDOWS_PER_FILE)

        by_cond.setdefault(cond_key, [])
        for start in starts:
            by_cond[cond_key].append((file_path, start, label))
        n_loaded += 1

    if n_skipped > 0 and n_loaded == 0:
        print(f"  [警告] {class_dir.name}: {n_skipped} 个文件全部加载失败！")
    return by_cond


def _sample_pool(pool: list, n: int, rng: random.Random) -> list:
    """
    从 pool 中取 n 个样本：
    - pool 为空或 n<=0 返回 []
    - pool 够用则直接 sample（无放回）
    - pool 不足则有放回重复补足
    """
    if not pool or n <= 0:
        return []
    if len(pool) >= n:
        return rng.sample(pool, n)
    # 有放回补足
    result = list(pool)
    while len(result) < n:
        result.extend(rng.choices(pool, k=min(len(pool), n - len(result))))
    return result[:n]


def build_mixed_and_real_only_indices(
    root_real: Path,
    root_gen: Path | None,
    rng: random.Random,
) -> tuple[list, list, list, list[str]]:
    """
    构建两个训练集的窗口索引以及共用测试集索引，每工况总样本数固定为
    TOTAL_WINDOWS_PER_CONDITION（128）。

    测试集（共用）
    --------------
    从每个工况的**真实数据池**中先按 TEST_RATIO 分层划出测试窗口，剩余真实数据
    才进入训练采样池。两个模型使用完全相同的测试集（纯真实数据）。

    模型1（混合）训练集规则
    -----------------------
    - 核心区（rpm≈1800, load∈[20,40]）：64 真实 + 64 生成
    - 边缘区（rpm=1000/2500, load=0/60）：0 真实 + 128 生成（纯生成）
    - 过渡区：真实数量按核心度线性衰减（从 64→0），生成数量补足至 128
    真实/生成样本不足时：有放回重复抽样补足。

    模型2（纯真实）训练集规则
    -------------------------
    直接从模型1 的训练索引中过滤出 .mat 真实文件条目。
    保证与模型1 中的真实训练样本完全一致（同批次、同滑窗起点）。
    边缘区真实数量为 0 时，模型2 该工况无训练数据。

    Returns
    -------
    mixed_idx, real_only_idx, shared_test_idx, class_names
    """
    class_list = collect_class_folders(root_real)
    class_names = [name for name, _ in class_list]

    gen_available = root_gen is not None and root_gen.is_dir()
    if not gen_available:
        print(
            f"[警告] 生成数据目录不存在: {root_gen}，模型1 将退化为全真实数据。",
            file=sys.stderr,
        )

    mixed_idx: list = []
    shared_test_idx: list = []

    for folder_name, label in class_list:
        real_class_dir = root_real / folder_name
        # 生成数据目录按类别索引命名：class_0, class_1, ...
        # 与真实数据按排序顺序对应（IF0.2→class_0, NC→class_3, ...）
        gen_class_dir = (root_gen / f"class_{label}") if gen_available else None

        real_by_cond = _collect_windows_by_condition(real_class_dir, label, rng)
        gen_by_cond = (
            _collect_windows_by_condition(gen_class_dir, label, rng)
            if gen_class_dir is not None and gen_class_dir.is_dir()
            else {}
        )

        all_conds = set(real_by_cond.keys()) | set(gen_by_cond.keys())
        total_real_cls = 0
        total_gen_cls = 0
        total_test_cls = 0

        for cond_key in sorted(all_conds):
            load_val, rpm_val = cond_key
            real_pool = real_by_cond.get(cond_key, [])
            gen_pool = gen_by_cond.get(cond_key, [])

            # ── 第一步：从真实池中按 TEST_RATIO 划出共用测试集 ──
            rng.shuffle(real_pool)
            n_test = max(1, round(len(real_pool) * TEST_RATIO)) if len(real_pool) >= 3 else 0
            test_part = real_pool[:n_test]
            train_real_pool = real_pool[n_test:]   # 剩余真实数据才用于训练

            shared_test_idx.extend(test_part)
            total_test_cls += len(test_part)

            if rpm_val == -1 or load_val == -1:
                # 无法解析工况 → 全量剩余真实数据加入训练
                mixed_idx.extend(train_real_pool)
                total_real_cls += len(train_real_pool)
                continue

            # ── 第二步：按距离衰减计算本工况目标真实训练数量 ──
            n_real = _compute_real_count(rpm_val, load_val)
            n_gen = TOTAL_WINDOWS_PER_CONDITION - n_real

            sampled_real = _sample_pool(train_real_pool, n_real, rng)
            sampled_gen = _sample_pool(gen_pool, n_gen, rng)

            mixed_idx.extend(sampled_real)
            mixed_idx.extend(sampled_gen)

            total_real_cls += len(sampled_real)
            total_gen_cls += len(sampled_gen)

        print(
            f"[信息] 类别 {folder_name}: "
            f"模型1 真实={total_real_cls} 生成={total_gen_cls} "
            f"总={total_real_cls + total_gen_cls} | "
            f"模型2(真实过滤后)={total_real_cls} | "
            f"共用测试集={total_test_cls}"
        )

    if not mixed_idx:
        raise RuntimeError(
            "混合数据集为空。请检查 DATA_ROOT 路径及 .npy 文件是否存在。"
        )
    if not shared_test_idx:
        raise RuntimeError(
            "共用测试集为空。请检查真实数据目录中是否有足够的 .mat 文件。"
        )

    # 模型2：直接从模型1 训练索引中过滤 .mat 真实文件条目
    # 保证与模型1 中的真实训练样本完全相同（同批次、同滑窗起点）
    real_only_idx = [
        entry for entry in mixed_idx
        if Path(entry[0]).suffix.lower() == ".mat"
    ]

    print(
        f"\n[数据集统计] "
        f"模型1(混合)训练: {len(mixed_idx)} | "
        f"模型2(真实)训练: {len(real_only_idx)} | "
        f"共用测试集(真实): {len(shared_test_idx)}\n"
    )
    print(
        f"[混合策略] 每工况总样本={TOTAL_WINDOWS_PER_CONDITION} | "
        f"核心区真实={round(TOTAL_WINDOWS_PER_CONDITION * REAL_FRAC_CORE)} | "
        f"边缘区真实={round(TOTAL_WINDOWS_PER_CONDITION * REAL_FRAC_EDGE)}（纯生成）\n"
    )
    return mixed_idx, real_only_idx, shared_test_idx, class_names


def train_one_model(
    index: list,
    class_names: list[str],
    out_path: Path,
    model_label: str,
    device,
    cache: dict,
    shared_test_index: list | None = None,
) -> float:
    """
    从给定窗口索引训练 BearingCNN1D，以验证集最优权重保存到 out_path。

    参数
    ----
    shared_test_index : list | None
        若提供，直接使用该列表作为测试集（两个模型共用纯真实测试集）；
        若为 None，则从 index 中按 TEST_RATIO 自行划分测试集（向后兼容）。

    返回最终测试集准确率。
    """
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    from bearing_models import BEARING_CNN_ARCH, BearingCNN1D, count_conv1d_layers

    num_classes = len(class_names)

    if shared_test_index is not None:
        # 使用外部共用测试集：从 index 中只划分 train/val
        train_list, val_list, _ = stratified_train_val_test(
            index, test_ratio=0.0, train_frac_of_rest=TRAIN_IN_TRAINVAL, seed=SEED
        )
        # test_ratio=0.0 时 stratified 可能仍尝试切 test，直接把 _ 丢弃
        test_list = shared_test_index
    else:
        train_list, val_list, test_list = stratified_train_val_test(
            index, TEST_RATIO, TRAIN_IN_TRAINVAL, SEED
        )
    print(
        f"[{model_label}] 样本数 — 训练: {len(train_list)} | "
        f"验证: {len(val_list)} | 测试: {len(test_list)} "
        f"{'(共用真实测试集)' if shared_test_index is not None else ''} | "
        f"类别数: {num_classes}"
    )

    if len(val_list) == 0:
        print(f"[错误] {model_label} 验证集为空，跳过训练。", file=sys.stderr)
        return 0.0
    if len(test_list) == 0:
        print(f"[错误] {model_label} 测试集为空，跳过训练。", file=sys.stderr)
        return 0.0

    ds_train = BearingNpyWindowDataset(train_list, cache, NORMALIZE_PER_WINDOW)
    ds_val = BearingNpyWindowDataset(val_list, cache, NORMALIZE_PER_WINDOW)
    ds_test = BearingNpyWindowDataset(test_list, cache, NORMALIZE_PER_WINDOW)

    def collate(batch):
        xs = torch.stack([b[0] for b in batch], dim=0)
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    dl_train = DataLoader(
        ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate
    )
    dl_val = DataLoader(
        ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate
    )
    dl_test = DataLoader(
        ds_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate
    )

    model = BearingCNN1D(num_classes=num_classes).to(device)
    n_conv = count_conv1d_layers(model)
    print(
        f"[{model_label}] BearingCNN1D: Conv1d×{n_conv} | "
        f"设备: {device} | epochs上限: {EPOCHS} | 早停: {EARLY_STOP_PATIENCE}"
    )

    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
    crit = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    patience_cnt = 0
    stopped_early = False

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
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
                "model_type": model_label,
            }
            torch.save(ckpt, str(out_path))
            print(
                f"  [{model_label}] Epoch {epoch+1:03d}/{EPOCHS} | "
                f"loss={train_loss:.4f} | val_acc={val_acc:.4f} | 【保存】-> {out_path.name}"
            )
        else:
            patience_cnt += 1
            if (epoch + 1) % 20 == 0 or patience_cnt >= EARLY_STOP_PATIENCE:
                print(
                    f"  [{model_label}] Epoch {epoch+1:03d}/{EPOCHS} | "
                    f"loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
                    f"早停计数 {patience_cnt}/{EARLY_STOP_PATIENCE}"
                )

        if patience_cnt >= EARLY_STOP_PATIENCE:
            print(f"  [{model_label}] 早停触发（连续 {EARLY_STOP_PATIENCE} epoch 无提升）。")
            stopped_early = True
            break

    if not stopped_early:
        print(f"  [{model_label}] 已达最大 epoch={EPOCHS}，未触发早停。")

    # 加载验证集最优权重，最终在测试集评估一次
    if out_path.is_file():
        try:
            ck = torch.load(str(out_path), map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(str(out_path), map_location=device)
        model.load_state_dict(ck["state_dict"])

    test_acc = evaluate_accuracy(model, dl_test, device)
    cm = evaluate_confusion_matrix(model, dl_test, device, num_classes)
    print(
        f"\n  [{model_label}] 测试集准确率（验证集最优权重）: {test_acc:.4f} | "
        f"val_best={best_val_acc:.4f}"
    )
    print(f"  [{model_label}] 模型已保存: {out_path.resolve()}")
    print(f"  [{model_label}] 混淆矩阵（行=真实，列=预测）:")
    print_confusion_matrix(cm, class_names)
    return test_acc


def train_two_models() -> int:
    """
    主训练入口（默认）：构造混合数据集与纯真实数据集，分别训练并保存两个 .pth 文件。

    模型1 (model_mixed.pth)：
      - 核心区（CORE_RPM=1800, load 20-40）：64 真实 + 64 生成 = 128（1:1）
      - 边缘区（rpm 1000/2500, load 0/60）：   0 真实 + 128 生成 = 128（纯生成）
      - 过渡区：真实数量从 64 线性衰减到 0，生成数量补足至 128
      - 每工况总样本数固定为 TOTAL_WINDOWS_PER_CONDITION（128）

    模型2 (model_real_only.pth)：
      - 直接从模型1 索引中过滤出真实数据条目（.mat 文件）
      - 与模型1 中的真实样本完全一致（同批次、同滑窗起点）
    """
    set_seed(SEED)
    try:
        import torch
    except ImportError:
        print("[错误] 需要 PyTorch: pip install torch", file=sys.stderr)
        return 1

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and DEVICE == "cuda":
        print("[警告] CUDA 不可用，使用 CPU 训练。")

    root_real = Path(DATA_ROOT)
    root_gen = Path(GEN_DATA_ROOT)

    print(f"{'='*60}")
    print(f"  双模型训练流程")
    print(f"  真实数据: {root_real}")
    print(f"  生成数据: {root_gen}")
    print(f"  工况设置: 核心区 RPM={CORE_RPM} Load=[{CORE_LOAD_LO},{CORE_LOAD_HI}]")
    print(f"           边缘区 RPM=[{EDGE_RPM_LO},{EDGE_RPM_HI}] Load=[{EDGE_LOAD_LO},{EDGE_LOAD_HI}]")
    core_n = round(TOTAL_WINDOWS_PER_CONDITION * REAL_FRAC_CORE)
    print(
        f"  混合策略: 每工况总={TOTAL_WINDOWS_PER_CONDITION} | "
        f"核心区 真实{core_n}:生成{TOTAL_WINDOWS_PER_CONDITION - core_n} | "
        f"边缘区 纯生成{TOTAL_WINDOWS_PER_CONDITION}（真实占比 0）"
    )
    print(f"{'='*60}\n")

    rng_build = random.Random(SEED)
    mixed_idx, real_only_idx, shared_test_idx, class_names = build_mixed_and_real_only_indices(
        root_real, root_gen, rng_build
    )

    cache: dict[str, np.ndarray] = {}
    out_mixed = SCRIPT_DIR / OUTPUT_MODEL_PATH_MIXED
    out_real = SCRIPT_DIR / OUTPUT_MODEL_PATH_REAL_ONLY

    print(f"\n{'='*60}")
    print(f"[模型1] 真实+生成混合模型 -> {out_mixed.name}")
    print(f"{'='*60}")
    set_seed(SEED)
    acc_mixed = train_one_model(
        mixed_idx, class_names, out_mixed, "模型1-混合", device, cache,
        shared_test_index=shared_test_idx,
    )

    print(f"\n{'='*60}")
    print(f"[模型2] 纯真实数据模型 -> {out_real.name}")
    print(f"{'='*60}")
    set_seed(SEED)
    acc_real = train_one_model(
        real_only_idx, class_names, out_real, "模型2-真实", device, cache,
        shared_test_index=shared_test_idx,
    )

    print(f"\n{'='*60}")
    print(f"  双模型训练完成（两模型使用相同的纯真实测试集，共 {len(shared_test_idx)} 条）")
    print(f"{'='*60}")
    print(f"  模型1（混合）测试准确率: {acc_mixed:.4f}  ({acc_mixed*100:.2f}%)")
    print(f"           路径: {out_mixed.resolve()}")
    print(f"  模型2（真实）测试准确率: {acc_real:.4f}  ({acc_real*100:.2f}%)")
    print(f"           路径: {out_real.resolve()}")
    print(
        f"\n  在 receive_udp_hil.py 中修改 MODEL_PATH 即可切换使用两个模型："
        f"\n    MODEL_PATH = \"{OUTPUT_MODEL_PATH_MIXED}\"    # 混合模型"
        f"\n    MODEL_PATH = \"{OUTPUT_MODEL_PATH_REAL_ONLY}\"  # 纯真实模型"
    )
    print(f"{'='*60}")
    return 0


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
    parser = argparse.ArgumentParser(
        description="轴承 1D CNN 训练（默认：双模型混合训练）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式说明：
  默认              train_two_models()  — 生成 model_mixed.pth + model_real_only.pth
  --single          train()             — 原始单模型训练，生成 best_model.pth
  --eval-only       eval_only()         — 仅评估 best_model.pth（不训练）
        """,
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="不训练，仅加载 best_model.pth 在测试集上评估并打印混淆矩阵",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="原始单模型训练模式（生成 best_model.pth，忽略 GEN_DATA_ROOT）",
    )
    args = parser.parse_args()

    if args.eval_only:
        raise SystemExit(eval_only())
    elif args.single:
        raise SystemExit(train())
    else:
        raise SystemExit(train_two_models())
