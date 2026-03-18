"""
dataset.py — 数据集加载与管理模块

提供统一的轴承故障振动信号数据集接口，支持从 .npy 文件加载数据，
并按照 TRTR / TSTR 实验范式划分真实数据（训练/测试）与生成数据（训练）。

数据格式约定：
  - .npy 文件，形状 (N, 1, 1024)
  - 值域：float32，归一化或原始幅值均可
  - 标签以整数表示故障类别（0, 1, 2, ...）

依赖：torch, numpy
"""

from __future__ import annotations

import glob
import os
import re
from typing import Tuple, Optional, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# 核心数据集类
# ---------------------------------------------------------------------------

class BearingSignalDataset(Dataset):
    """
    轴承故障振动信号数据集。

    每个样本为 (signal, label)：
      - signal: Tensor, shape (1, L)   — 单通道 1D 振动信号
      - label:  Tensor, shape ()       — 故障类别标签 (int64)

    支持两种构造方式：
      1. 直接传入 Tensor：BearingSignalDataset(signals, labels)
      2. 从 .npy 文件加载：BearingSignalDataset.from_npy(signal_path, label_path)
    """

    def __init__(self, signals: Tensor, labels: Tensor) -> None:
        """
        参数
        ----
        signals : Tensor, shape (N, 1, L)
            N 条振动信号。
        labels : Tensor, shape (N,)
            每条信号对应的故障类别标签。
        """
        assert signals.ndim == 3 and signals.shape[1] == 1, \
            f"signals 应为 (N, 1, L)，实际为 {tuple(signals.shape)}"
        assert labels.ndim == 1 and len(labels) == len(signals), \
            f"labels 长度 ({len(labels)}) 与 signals 长度 ({len(signals)}) 不匹配"

        self.signals = signals.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.signals[idx], self.labels[idx]

    # ----- 工厂方法 -----

    @classmethod
    def from_npy(
        cls,
        signal_path: str,
        label_path: str,
    ) -> "BearingSignalDataset":
        """
        从 .npy 文件构造数据集。

        参数
        ----
        signal_path : str
            信号文件路径，形状 (N, 1, L)。
        label_path : str
            标签文件路径，形状 (N,)。
        """
        signals = np.load(signal_path).astype(np.float32)
        labels = np.load(label_path).astype(np.int64)

        # 兼容 (N, L) 形状：自动 unsqueeze
        if signals.ndim == 2:
            signals = signals[:, np.newaxis, :]

        return cls(
            signals=torch.from_numpy(signals),
            labels=torch.from_numpy(labels),
        )

    @classmethod
    def from_class_folders(
        cls,
        root_dir: str,
        class_names: Optional[List[str]] = None,
    ) -> "BearingSignalDataset":
        """
        从按类别分文件夹的 .npy 文件加载。

        目录结构示例::

            root_dir/
              class_0/
                sample_0.npy   # shape (1, L)  or (L,)
                sample_1.npy
              class_1/
                ...

        参数
        ----
        root_dir : str
            根目录路径。
        class_names : list[str] | None
            类别名称列表，若为 None 则按文件夹名排序自动推断。
        """
        if class_names is None:
            class_names = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )

        all_signals: list[np.ndarray] = []
        all_labels: list[int] = []

        for label_idx, cname in enumerate(class_names):
            folder = os.path.join(root_dir, cname)
            for fname in sorted(os.listdir(folder)):
                if not fname.endswith(".npy"):
                    continue
                sig = np.load(os.path.join(folder, fname)).astype(np.float32)
                if sig.ndim == 1:
                    sig = sig[np.newaxis, :]       # (L,) -> (1, L)
                elif sig.ndim == 2 and sig.shape[0] != 1:
                    sig = sig[:1, :]               # 取第一通道
                all_signals.append(sig)
                all_labels.append(label_idx)

        signals = np.stack(all_signals, axis=0)    # (N, 1, L)
        labels = np.array(all_labels, dtype=np.int64)

        return cls(
            signals=torch.from_numpy(signals),
            labels=torch.from_numpy(labels),
        )

    @classmethod
    def from_class_folders_with_subsample(
        cls,
        root_dir: str,
        class_names: Optional[List[str]] = None,
        max_per_class: Optional[int] = None,
        max_per_group: Optional[int] = None,
        seed: int = 42,
    ) -> "BearingSignalDataset":
        """
        从按类别分文件夹加载，支持按类或按组抽样以控制样本量。

        文件名若含 "RPM Load" 模式（如 "xxx 1000 0.npy"），则按组抽样；
        否则仅按类抽样。

        参数
        ----
        root_dir : str
            根目录路径。
        class_names : list[str] | None
            类别名称列表。
        max_per_class : int | None
            每类最多保留样本数，None 表示不限制。
        max_per_group : int | None
            每个 (RPM, Load) 组最多保留样本数，None 表示不限制。
        seed : int
            随机种子。
        """
        rng = np.random.default_rng(seed)
        if class_names is None:
            class_names = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )

        # 匹配文件名中的 "数字 数字" 作为 (rpm, load)，如 "IF0.2 1000 0.npy"
        group_pattern = re.compile(r"(\d+)\s+(\d+)(?:\D|$)")

        all_signals: list[np.ndarray] = []
        all_labels: list[int] = []

        for label_idx, cname in enumerate(class_names):
            folder = os.path.join(root_dir, cname)
            files_by_group: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = {}
            ungrouped: list[tuple[str, np.ndarray]] = []

            for fname in sorted(os.listdir(folder)):
                if not fname.endswith(".npy"):
                    continue
                fp = os.path.join(folder, fname)
                sig = np.load(fp).astype(np.float32)
                if sig.ndim == 1:
                    sig = sig[np.newaxis, :]
                elif sig.ndim == 2 and sig.shape[0] != 1:
                    sig = sig[:1, :]
                m = group_pattern.search(fname)
                if m and max_per_group is not None:
                    rpm, load = int(m.group(1)), int(m.group(2))
                    key = (rpm, load)
                    if key not in files_by_group:
                        files_by_group[key] = []
                    files_by_group[key].append((fname, sig))
                else:
                    ungrouped.append((fname, sig))

            # 抽样：优先按组，再按类
            collected: list[np.ndarray] = []
            if max_per_group is not None and files_by_group:
                for lst in files_by_group.values():
                    rng.shuffle(lst)
                    for _, sig in lst[:max_per_group]:
                        collected.append(sig)
                for _, sig in ungrouped:
                    collected.append(sig)
            else:
                for grp in files_by_group.values():
                    for _, sig in grp:
                        collected.append(sig)
                for _, sig in ungrouped:
                    collected.append(sig)
            if max_per_class is not None and len(collected) > max_per_class:
                rng.shuffle(collected)
                collected = collected[:max_per_class]
            for sig in collected:
                all_signals.append(sig)
                all_labels.append(label_idx)

        if not all_signals:
            raise FileNotFoundError(f"No .npy files found under {root_dir}")

        signals = np.stack(all_signals, axis=0)
        labels = np.array(all_labels, dtype=np.int64)
        return cls(
            signals=torch.from_numpy(signals),
            labels=torch.from_numpy(labels),
        )


# ---------------------------------------------------------------------------
# 从路径加载真实/生成数据（用于 infer 生成结果的评估）
# ---------------------------------------------------------------------------

def load_from_flat_folder(
    folder: str,
    label: int,
    pattern: str = "*.npy",
) -> "BearingSignalDataset":
    """
    从扁平目录加载所有 .npy 文件，统一指定一个类别标签。

    用于加载 generated_samples_infer 等单类别生成数据。

    参数
    ----
    folder : str
        目录路径（如 ./generated_samples_infer）。
    label : int
        所有样本的类别标签。
    pattern : str
        文件名匹配模式，默认 *.npy。

    返回
    ----
    BearingSignalDataset
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise FileNotFoundError(f"No .npy files found in {folder}")

    all_signals: list[np.ndarray] = []
    for fp in paths:
        sig = np.load(fp).astype(np.float32)
        if sig.ndim == 1:
            sig = sig[np.newaxis, :]       # (L,) -> (1, L)
        elif sig.ndim == 2 and sig.shape[0] != 1:
            sig = sig[:1, :]
        all_signals.append(sig)

    signals = np.stack(all_signals, axis=0)   # (N, 1, L)
    labels = np.full(len(all_signals), label, dtype=np.int64)
    return BearingSignalDataset(
        signals=torch.from_numpy(signals),
        labels=torch.from_numpy(labels),
    )


def _load_gen_from_root_or_flat(
    gen_folder: str,
    class_names: List[str],
    gen_label: int,
    max_per_class: Optional[int] = None,
    seed: int = 42,
) -> "BearingSignalDataset":
    """
    从生成数据目录加载。若 gen_folder 下有与 class_names 同名的子文件夹，
    则按类加载；否则视为扁平目录，所有样本标为 gen_label。
    """
    rng = np.random.default_rng(seed)
    subdirs = [d for d in os.listdir(gen_folder) if os.path.isdir(os.path.join(gen_folder, d))]
    npy_in_root = len(glob.glob(os.path.join(gen_folder, "*.npy")))

    all_sigs, all_lbls = [], []
    if subdirs and npy_in_root == 0:
        for label_idx, cname in enumerate(class_names):
            subpath = os.path.join(gen_folder, cname)
            if not os.path.isdir(subpath):
                continue
            paths = sorted(glob.glob(os.path.join(subpath, "*.npy")))
            sigs_here = []
            for fp in paths:
                sig = np.load(fp).astype(np.float32)
                if sig.ndim == 1:
                    sig = sig[np.newaxis, :]
                elif sig.ndim == 2 and sig.shape[0] != 1:
                    sig = sig[:1, :]
                sigs_here.append(sig)
            if max_per_class is not None and len(sigs_here) > max_per_class:
                rng.shuffle(sigs_here)
                sigs_here = sigs_here[:max_per_class]
            all_sigs.extend(sigs_here)
            all_lbls.extend([label_idx] * len(sigs_here))
    if all_sigs:
        signals = np.stack(all_sigs, axis=0)
        labels = np.array(all_lbls, dtype=np.int64)
        return BearingSignalDataset(torch.from_numpy(signals), torch.from_numpy(labels))
    # Fallback: flat folder, single label
    return load_from_flat_folder(gen_folder, gen_label)


def load_real_and_gen_for_eval(
    real_data_path: str,
    gen_data_folder: str,
    gen_label: int,
    train_ratio: float = 0.7,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
    max_per_class: Optional[int] = None,
    max_per_group: Optional[int] = None,
) -> Tuple["BearingSignalDataset", "BearingSignalDataset", "BearingSignalDataset", int, List[str]]:
    """
    加载真实数据与生成数据，并按 train_ratio 划分真实数据为 train/test。

    真实数据支持两种结构：
      1) 按类别分文件夹：real_data_path/IF0.2/*.npy, real_data_path/OF0.2/*.npy ...
      2) 扁平单文件夹：real_data_path/*.npy（全部视为同一类别）

    生成数据：gen_data_folder 下所有 .npy，统一标签为 gen_label。

    返回
    ----
    real_train, gen_train, real_test : BearingSignalDataset
    num_classes : int
        总类别数（由真实数据推断，gen_label 必须在 [0, num_classes-1] 内）
    class_names : List[str]
        类别名称列表
    """
    rng = np.random.default_rng(seed)

    # 判断真实数据结构
    subdirs = [
        d for d in os.listdir(real_data_path)
        if os.path.isdir(os.path.join(real_data_path, d))
    ]
    npy_in_root = len(glob.glob(os.path.join(real_data_path, "*.npy")))

    if subdirs and not npy_in_root:
        # 按类别分文件夹（支持抽样）
        if class_names is None:
            class_names = sorted(subdirs)
        if max_per_class is not None or max_per_group is not None:
            real_full = BearingSignalDataset.from_class_folders_with_subsample(
                real_data_path,
                class_names=class_names,
                max_per_class=max_per_class,
                max_per_group=max_per_group,
                seed=seed,
            )
        else:
            real_full = BearingSignalDataset.from_class_folders(
                real_data_path, class_names=class_names
            )
        num_classes = len(class_names)
    else:
        # 扁平单文件夹，单类别
        all_sigs = []
        for fp in sorted(glob.glob(os.path.join(real_data_path, "*.npy"))):
            sig = np.load(fp).astype(np.float32)
            if sig.ndim == 1:
                sig = sig[np.newaxis, :]
            elif sig.ndim == 2 and sig.shape[0] != 1:
                sig = sig[:1, :]
            all_sigs.append(sig)
        if not all_sigs:
            raise FileNotFoundError(f"No .npy files in {real_data_path}")
        if max_per_class is not None and len(all_sigs) > max_per_class:
            rng.shuffle(all_sigs)
            all_sigs = all_sigs[:max_per_class]
        signals = np.stack(all_sigs, axis=0)
        labels = np.zeros(len(all_sigs), dtype=np.int64)
        real_full = BearingSignalDataset(torch.from_numpy(signals), torch.from_numpy(labels))
        num_classes = 1
        class_names = ["类别0"] if class_names is None else class_names

    # 确保 gen_label 有效
    if gen_label < 0 or gen_label >= num_classes:
        gen_label = 0
        if num_classes > 1:
            print(f"Warning: gen_label 超出范围，已设为 0")

    # 分层划分 real -> train / test
    train_sigs, train_lbls = [], []
    test_sigs, test_lbls = [], []
    for c in range(num_classes):
        mask = real_full.labels.numpy() == c
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * train_ratio))
        n_test = len(idx) - n_train
        if n_test < 1:
            n_train -= 1
            n_test = 1
        train_sigs.append(real_full.signals.numpy()[idx[:n_train]])
        train_lbls.append(real_full.labels.numpy()[idx[:n_train]])
        test_sigs.append(real_full.signals.numpy()[idx[n_train:]])
        test_lbls.append(real_full.labels.numpy()[idx[n_train:]])

    real_train_sig = np.concatenate(train_sigs, axis=0)
    real_train_lbl = np.concatenate(train_lbls, axis=0)
    real_test_sig = np.concatenate(test_sigs, axis=0)
    real_test_lbl = np.concatenate(test_lbls, axis=0)

    real_train = BearingSignalDataset(
        torch.from_numpy(real_train_sig), torch.from_numpy(real_train_lbl)
    )
    real_test = BearingSignalDataset(
        torch.from_numpy(real_test_sig), torch.from_numpy(real_test_lbl)
    )

    # 加载生成数据（支持按类子目录或扁平目录）
    gen_train = _load_gen_from_root_or_flat(
        gen_data_folder, class_names, gen_label,
        max_per_class=max_per_class, seed=seed,
    )

    return real_train, gen_train, real_test, num_classes, class_names


def load_gen_only_for_eval(
    gen_data_folder: str,
    train_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple["BearingSignalDataset", "BearingSignalDataset", "BearingSignalDataset", int, List[str]]:
    """
    仅从生成数据目录加载，将其划分为「参考真实」(前一半) 与「生成」(后一半)，
    用于仅有 generated_samples_infer 时的时频对比与 t-SNE。
    TRTR/TSTR 将退化为单类（准确率恒为 1.0）。

    返回
    ----
    real_train, gen_train, real_test : 前一半作为 real，后一半作为 gen
    num_classes : 1
    class_names : ["生成信号"]
    """
    rng = np.random.default_rng(seed)
    gen_full = load_from_flat_folder(gen_data_folder, label=0)
    n = len(gen_full)
    if n < 4:
        raise ValueError(f"生成数据至少需 4 条，当前仅 {n} 条")
    idx = np.arange(n)
    rng.shuffle(idx)
    n_real = n // 2
    real_idx = idx[:n_real]
    gen_idx = idx[n_real:]

    real_sigs = gen_full.signals.numpy()[real_idx]
    real_lbls = gen_full.labels.numpy()[real_idx]
    gen_sigs = gen_full.signals.numpy()[gen_idx]
    gen_lbls = gen_full.labels.numpy()[gen_idx]

    n_real_train = max(1, int(n_real * train_ratio))
    real_train = BearingSignalDataset(
        torch.from_numpy(real_sigs[:n_real_train]),
        torch.from_numpy(real_lbls[:n_real_train]),
    )
    real_test = BearingSignalDataset(
        torch.from_numpy(real_sigs[n_real_train:]),
        torch.from_numpy(real_lbls[n_real_train:]),
    )
    gen_train = BearingSignalDataset(torch.from_numpy(gen_sigs), torch.from_numpy(gen_lbls))

    return real_train, gen_train, real_test, 1, ["生成信号"]


# ---------------------------------------------------------------------------
# 快速构造 Dummy 数据（用于全流程冒烟测试）
# ---------------------------------------------------------------------------

def make_dummy_datasets(
    num_classes: int = 4,
    samples_per_class: int = 64,
    seq_length: int = 1024,
    train_ratio: float = 0.7,
    seed: int = 42,
) -> Tuple[BearingSignalDataset, BearingSignalDataset, BearingSignalDataset]:
    """
    生成三份 dummy 数据集：真实训练集、生成训练集、真实测试集。

    每个类别生成 `samples_per_class` 条含不同基频的合成正弦 + 噪声信号，
    以确保不同类别在频域上可区分（便于验证分类器流程）。

    返回
    ----
    real_train : BearingSignalDataset
        真实数据 — 训练集
    gen_train  : BearingSignalDataset
        生成数据 — 训练集（模拟扩散模型输出）
    real_test  : BearingSignalDataset
        真实数据 — 测试集
    """
    rng = np.random.default_rng(seed)
    fs = 25600.0
    t = np.arange(seq_length, dtype=np.float32) / fs

    all_signals: list[np.ndarray] = []
    all_labels: list[int] = []

    for cls_idx in range(num_classes):
        # 每个类别使用不同的基频（模拟不同故障模式）
        base_freq = 50.0 + cls_idx * 120.0  # 50, 170, 290, 410 Hz ...
        for _ in range(samples_per_class):
            # 基波 + 2 次谐波 + 高斯噪声
            amp = rng.uniform(0.5, 1.5)
            phase = rng.uniform(0, 2 * np.pi)
            sig = (
                amp * np.sin(2 * np.pi * base_freq * t + phase)
                + 0.3 * amp * np.sin(2 * np.pi * 2 * base_freq * t + phase)
                + rng.normal(0, 0.1, size=t.shape)
            ).astype(np.float32)
            # 归一化到 [-1, 1]
            sig = sig / (np.abs(sig).max() + 1e-8)
            all_signals.append(sig[np.newaxis, :])   # (1, L)
            all_labels.append(cls_idx)

    signals = np.stack(all_signals, axis=0)          # (N_total, 1, L)
    labels = np.array(all_labels, dtype=np.int64)    # (N_total,)

    # 按类别分层划分 train / test
    train_sigs, train_lbls = [], []
    test_sigs, test_lbls = [], []
    for c in range(num_classes):
        mask = labels == c
        idx = np.where(mask)[0]
        rng.shuffle(idx)
        n_train = int(len(idx) * train_ratio)
        train_sigs.append(signals[idx[:n_train]])
        train_lbls.append(labels[idx[:n_train]])
        test_sigs.append(signals[idx[n_train:]])
        test_lbls.append(labels[idx[n_train:]])

    real_train_sig = np.concatenate(train_sigs, axis=0)
    real_train_lbl = np.concatenate(train_lbls, axis=0)
    real_test_sig = np.concatenate(test_sigs, axis=0)
    real_test_lbl = np.concatenate(test_lbls, axis=0)

    # 生成数据：在真实训练数据基础上叠加轻微噪声，模拟扩散模型输出
    gen_noise_scale = 0.15
    gen_train_sig = real_train_sig + rng.normal(0, gen_noise_scale, size=real_train_sig.shape).astype(np.float32)
    gen_train_sig = gen_train_sig / (np.abs(gen_train_sig).max(axis=-1, keepdims=True) + 1e-8)
    gen_train_lbl = real_train_lbl.copy()

    real_train = BearingSignalDataset(
        torch.from_numpy(real_train_sig), torch.from_numpy(real_train_lbl)
    )
    gen_train = BearingSignalDataset(
        torch.from_numpy(gen_train_sig), torch.from_numpy(gen_train_lbl)
    )
    real_test = BearingSignalDataset(
        torch.from_numpy(real_test_sig), torch.from_numpy(real_test_lbl)
    )

    return real_train, gen_train, real_test


# ---------------------------------------------------------------------------
# DataLoader 快捷构造
# ---------------------------------------------------------------------------

def build_dataloader(
    dataset: BearingSignalDataset,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """构造 DataLoader，默认参数适合小规模实验。"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


