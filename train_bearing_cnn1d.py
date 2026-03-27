#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承数据集 1D CNN 十分类（或多子文件夹类别）训练脚本。

数据结构（与资源管理器一致）:
  D:\\data\\轴承数据集\\
    IF0.2\\
      IF0.2 1000 0.mat
      IF0.2 1500 20.mat
      ...
    <其它故障文件夹>\\  ...

每个一级子文件夹名对应一个类别；默认期望 10 个子文件夹，否则按实际数量训练。

每个工况文件（如 IF0.2 3000 60.mat）最多取 MAX_WINDOWS_PER_MAT 个滑窗样本，减轻过拟合。
划分比例：先按类分层划出测试集 TEST_RATIO，余下部分按 TRAIN_IN_TRAINVAL（8:2）分为训练/验证。
早停：以**验证集**准确率为监控（避免对测试集调参导致指标偏乐观）；连续 EARLY_STOP_PATIENCE 个 epoch
未刷新验证最佳则停止。保存验证集最优权重；**测试集仅在训练结束后评估一次**，作为无偏泛化估计。

依赖: pip install torch scipy

说明: 经典 .mat 用 scipy.io.loadmat。振动数据路径与 train_1d_vibration 一致:
      Signal.y_values.values 二维数组的第 0 列。若为 MATLAB v7.3 (HDF5)，需 h5py 另行读取。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

# =============================================================================
# 全局配置（直接修改；IDE 点击 Run 即可）
# =============================================================================

DATA_ROOT = r"D:\data\轴承数据集"
WINDOW_SIZE = 1024  # 与 Speedgoat / receive_udp_hil 推理窗长一致
STRIDE = 1024       # 滑窗步长；改为 1024 则窗口不重叠

# 输出到脚本同目录，供 receive_udp_hil.py 加载（按「验证集最优」保存，部署更合理）
OUTPUT_MODEL_PATH = "best_model.pth"
EXPECTED_NUM_CLASSES = 10  # 子文件夹数不等于此时仅警告，仍以实际为准

# 每个 .mat 工况文件最多使用的滑窗样本数（避免读满全长导致过拟合）
MAX_WINDOWS_PER_MAT = 20

# 分层划分：测试集占比；剩余样本中训练集占比（训练:验证 = 8:2 即 0.8:0.2）
TEST_RATIO = 0.2
TRAIN_IN_TRAINVAL = 0.8  # 验证集占剩余部分的比例为 1 - TRAIN_IN_TRAINVAL

# 早停：监控**验证集**准确率，连续若干 epoch 未超过历史最佳则停止（测试集不参与早停）
EARLY_STOP_PATIENCE = 5

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


def extract_signal_sdust_signal_column0(data: dict) -> np.ndarray:
    """
    SDUST 轴承 .mat 标准结构（与 MATLAB 变量编辑器一致）:
      Signal -> y_values -> values 为二维表，振动通道取第 0 列。

    与 train_1d_vibration.py 一致:
      signal = data['Signal']['y_values'][0, 0]['values'].item()[:, 0]
    """
    sig_root = data["Signal"]
    yv = sig_root["y_values"]
    cell = yv[0, 0]
    values = cell["values"]
    # scipy 常把嵌套矩阵装在 0 维 object 里，需 .item() 取出真实 ndarray
    if isinstance(values, np.ndarray) and values.dtype == object and values.size == 1:
        mat = values.item()
    else:
        mat = np.asarray(values)
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[1] < 1:
        raise ValueError(f"Signal.y_values.values 期望 2D 且至少 1 列，当前 shape={mat.shape}")
    signal = np.asarray(mat[:, 0], dtype=np.float64).reshape(-1)
    return signal


def extract_signal_from_mat(mat_path: Path) -> np.ndarray:
    """
    从 .mat 中提取一维振动序列。

    1) 优先: SDUST 结构 Signal.y_values.values 的第 0 列（与 train_1d_vibration.py 相同）。
    2) 回退: 旧版启发式（顶层常见键名或最长 1D 数值数组）。
    """
    from scipy.io import loadmat

    path_str = str(mat_path)

    # --- 主路径：不用 simplify_cells，避免破坏 MATLAB struct 嵌套结构 ---
    try:
        data = loadmat(path_str)
        signal = extract_signal_sdust_signal_column0(data)
        if signal.size >= WINDOW_SIZE:
            return signal
        raise ValueError(f"Signal 第 0 列长度 {signal.size} < WINDOW_SIZE={WINDOW_SIZE}")
    except (KeyError, TypeError, IndexError, AttributeError, ValueError) as e:
        err_sdust = e

    # --- 回退：simplify_cells / 扁平键名 ---
    try:
        try:
            d = loadmat(path_str, simplify_cells=True)
        except TypeError:
            d = loadmat(path_str)
    except Exception as e:
        raise ValueError(f"loadmat 失败 {mat_path}: {e}") from err_sdust

    preferred_keys = (
        "sig_tensor",
        "signal",
        "data",
        "Data",
        "x",
        "X",
        "vibration",
        "sig",
        "FE",
        "de",
        "DE",
    )

    def _as_1d_float(arr: np.ndarray) -> np.ndarray | None:
        if arr.dtype == object:
            return None
        v = np.asarray(arr, dtype=np.float64)
        v = np.squeeze(v)
        if v.ndim != 1:
            v = v.reshape(-1)
        if v.size < WINDOW_SIZE:
            return None
        return v

    for key in preferred_keys:
        if key in d:
            v = _as_1d_float(np.asarray(d[key]))
            if v is not None:
                return v

    best = None
    best_len = 0
    for k, v in d.items():
        if k.startswith("__"):
            continue
        if not isinstance(v, np.ndarray):
            continue
        v = _as_1d_float(v)
        if v is not None and v.size > best_len:
            best_len = v.size
            best = v

    if best is None:
        raise ValueError(
            f"无法在 {mat_path} 中解析振动序列。"
            f" 已尝试 SDUST 路径 Signal.y_values.values[:,0]，失败原因: {err_sdust}"
        )

    return best


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


class BearingMatWindowDataset:
    """简易数据集：__getitem__ 读 mat 切片（带文件级缓存）。"""

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
            self._cache[key] = extract_signal_from_mat(path)
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


def train() -> int:
    set_seed(SEED)
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("[错误] 需要 PyTorch: pip install torch", file=sys.stderr)
        return 1

    try:
        from scipy.io import loadmat  # noqa: F401
    except ImportError:
        print("[错误] 需要 scipy: pip install scipy", file=sys.stderr)
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
            "[错误] 验证集为空（每类仅 1 个窗口时会出现）。请增大 MAX_WINDOWS_PER_MAT 或合并类别。",
            file=sys.stderr,
        )
        return 1
    if len(test_list) == 0:
        print(
            "[错误] 测试集为空（每类样本 <3 时无法分层出测试）。请增大 MAX_WINDOWS_PER_MAT 或降低 TEST_RATIO。",
            file=sys.stderr,
        )
        return 1

    ds_train = BearingMatWindowDataset(train_list, cache, NORMALIZE_PER_WINDOW)
    ds_val = BearingMatWindowDataset(val_list, cache, NORMALIZE_PER_WINDOW)
    ds_test = BearingMatWindowDataset(test_list, cache, NORMALIZE_PER_WINDOW)

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
    print(
        f"\n[最终评估] 使用「验证集最优」权重在**测试集**上的准确率（未参与早停）: {final_test_acc:.4f}\n"
        f"[信息] 训练过程中验证集最佳 val_acc={best_val_acc:.4f}\n"
        f"[完成] 模型已保存: {out_path.resolve()}"
    )
    return 0


def build_window_index_wrapped(root: Path) -> tuple[list[tuple[Path, int, int]], list[str]]:
    """
    扫描子文件夹与 .mat，生成窗口索引 (路径, 起点, 标签)。
    每个工况文件（单个 .mat）最多保留 MAX_WINDOWS_PER_MAT 个滑窗（随机下采样可复现）。
    """
    class_list = collect_class_folders(root)
    class_names = [name for name, _ in class_list]

    rng = random.Random(SEED)
    index: list[tuple[Path, int, int]] = []
    for folder_name, label in class_list:
        folder = root / folder_name
        for mat_path in sorted(folder.glob("*.mat")):
            try:
                sig = extract_signal_from_mat(mat_path)
            except Exception as e:
                print(f"[跳过] {mat_path}: {e}", file=sys.stderr)
                continue
            n = sig.shape[0]
            if n < WINDOW_SIZE:
                print(f"[跳过] {mat_path}: 长度 {n} < WINDOW_SIZE", file=sys.stderr)
                continue
            starts = list(range(0, n - WINDOW_SIZE + 1, STRIDE))
            if len(starts) > MAX_WINDOWS_PER_MAT:
                starts = rng.sample(starts, MAX_WINDOWS_PER_MAT)
            for start in starts:
                index.append((mat_path, start, label))

    if not index:
        raise RuntimeError("没有可用窗口")

    print(
        f"[信息] 共 {len(class_names)} 类: {class_names} | "
        f"窗口总数: {len(index)}（每文件最多 {MAX_WINDOWS_PER_MAT} 窗）"
    )
    return index, class_names


if __name__ == "__main__":
    raise SystemExit(train())
