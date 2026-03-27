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

依赖: pip install torch scipy

说明: 经典 .mat 用 scipy.io.loadmat；若为 MATLAB v7.3 (HDF5)，需 pip install h5py 并自行改写读取逻辑。
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
STRIDE = 512        # 滑窗步长；改为 1024 则窗口不重叠

# 输出到脚本同目录，供 receive_udp_hil.py 加载
OUTPUT_MODEL_PATH = "best_model.pth"
EXPECTED_NUM_CLASSES = 10  # 子文件夹数不等于此时仅警告，仍以实际为准

BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VAL_RATIO = 0.15
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


def extract_signal_from_mat(mat_path: Path) -> np.ndarray:
    """
    从 .mat 中提取一维振动序列。优先常见键名，否则取第一个足够长的数值 ndarray。
    """
    from scipy.io import loadmat

    try:
        d = loadmat(str(mat_path), simplify_cells=True)
    except TypeError:
        d = loadmat(str(mat_path))

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
        raise ValueError(f"无法在 {mat_path} 中找到长度 >= {WINDOW_SIZE} 的 1D 数值序列")

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


def split_train_val(
    index: list[tuple[Path, int, int]], val_ratio: float, seed: int
) -> tuple[list, list]:
    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for item in index:
        by_label.setdefault(item[2], []).append(item)

    train_idx: list = []
    val_idx: list = []
    for _, items in sorted(by_label.items()):
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_ratio)) if len(items) > 1 else 0
        if n_val >= len(items):
            n_val = len(items) // 5
        val_idx.extend(items[:n_val])
        train_idx.extend(items[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


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

    from bearing_models import BEARING_CNN_ARCH, BearingCNN1D

    root = Path(DATA_ROOT)
    index, class_names = build_window_index_wrapped(root)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and DEVICE == "cuda":
        print("[警告] CUDA 不可用，使用 CPU 训练。")

    num_classes = len(class_names)
    cache: dict[str, np.ndarray] = {}

    train_list, val_list = split_train_val(index, VAL_RATIO, SEED)
    print(f"[信息] 训练窗: {len(train_list)} | 验证窗: {len(val_list)} | 类别数: {num_classes}")

    ds_train = BearingMatWindowDataset(train_list, cache, NORMALIZE_PER_WINDOW)
    ds_val = BearingMatWindowDataset(val_list, cache, NORMALIZE_PER_WINDOW)

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

    model = BearingCNN1D(num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    out_path = SCRIPT_DIR / OUTPUT_MODEL_PATH

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

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).argmax(dim=-1)
                correct += int((pred == yb).sum().item())
                total += yb.numel()
        val_acc = correct / max(total, 1)
        train_loss = total_loss / max(n_seen, 1)

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            ckpt = {
                "architecture": BEARING_CNN_ARCH,
                "state_dict": model.state_dict(),
                "num_classes": num_classes,
                "class_names": class_names,
                "window_size": WINDOW_SIZE,
                "normalize_per_window": NORMALIZE_PER_WINDOW,
            }
            torch.save(ckpt, str(out_path))
            print(f"  -> 保存最佳权重: {out_path.resolve()} (val_acc={val_acc:.4f})")

    print(f"[完成] 最佳验证准确率: {best_acc:.4f} | 模型: {out_path.resolve()}")
    return 0


def build_window_index_wrapped(root: Path) -> tuple[list[tuple[Path, int, int]], list[str]]:
    """扫描子文件夹与 .mat，生成窗口索引 (路径, 起点, 标签) 与类别名列表。"""
    class_list = collect_class_folders(root)
    class_names = [name for name, _ in class_list]

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
            for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
                index.append((mat_path, start, label))

    if not index:
        raise RuntimeError("没有可用窗口")

    print(f"[信息] 共 {len(class_names)} 类: {class_names} | 窗口总数: {len(index)}")
    return index, class_names


if __name__ == "__main__":
    raise SystemExit(train())
