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
from typing import Dict, Tuple, Optional, List

import numpy as np
import torch

try:
    from scipy.io import loadmat
except ImportError:
    loadmat = None  # type: ignore
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# .mat 加载辅助（与 train_1d_vibration 一致）
# ---------------------------------------------------------------------------

def _load_mat_as_samples(
    mat_path: str,
    seq_length: int = 1024,
    overlap: float = 0.5,
    channel: int = 0,
) -> list[np.ndarray]:
    """
    从 .mat 加载长序列并分割为 (1, seq_length) 样本。

    支持两种格式（与 train_1d_vibration 一致）：
      - SDUST/IF0.2: Signal['y_values'][0,0][0] -> (N, 6)，取第 channel 通道
      - 带 values 字段: Signal['y_values'][0,0]['values'].item()[:, 0]
    """
    if loadmat is None:
        raise ImportError("scipy 未安装，无法加载 .mat 文件。请运行: pip install scipy")
    data = loadmat(mat_path)
    sig = data["Signal"][0, 0]
    yv = sig["y_values"][0, 0]

    # 格式1: SDUST/IF0.2 — 与 train_1d_vibration._load_signal_from_mat 一致
    signal = None
    for accessor in [lambda: yv[0], lambda: yv[0, 0], lambda: np.asarray(yv, dtype=object).flat[0]]:
        try:
            data = accessor()
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 2:
                s = arr[:, channel].flatten()
            elif arr.ndim == 1:
                s = arr.flatten()
            else:
                continue
            if len(s) >= seq_length:
                signal = s
                break
        except (IndexError, TypeError, ValueError, KeyError):
            continue

    # 格式2: 带 "values" 字段的结构体
    if signal is None:
        try:
            signal = yv["values"].item()[:, 0].astype(np.float32).flatten()
        except (KeyError, TypeError, AttributeError) as e:
            raise ValueError(
                f"无法解析 .mat 文件 {mat_path}：既非 SDUST 格式，亦无 values 字段。错误: {e}"
            ) from e

    if signal is None or len(signal) < seq_length:
        return []
    step = int(seq_length * (1 - overlap))
    samples = []
    for i in range(0, len(signal) - seq_length + 1, step):
        samples.append(signal[i : i + seq_length][np.newaxis, :])  # (1, L)
    return samples


def _normalize_rpm_load(v1: int, v2: int) -> Tuple[int, int]:
    """智能解析转速与负载，解决真实/生成数据命名顺序倒挂与量纲不一致"""
    rpm = max(v1, v2)
    load = min(v1, v2)
    # 统一量纲：如果真实数据负载是个位数（如 2, 4, 6），放大10倍对齐生成数据的 20, 40, 60
    if 0 < load < 10:
        load *= 10
    return load, rpm


def _collect_group_keys_in_class_folder(folder: str) -> set[tuple[int, int]]:
    """收集单类别目录中可解析到的工况组键 (load, rpm)。"""
    group_pattern = re.compile(r"(\d+)\s+(\d+)(?:\D|$)")
    keys: set[tuple[int, int]] = set()

    if not os.path.isdir(folder):
        return keys

    for item in os.listdir(folder):
        subpath = os.path.join(folder, item)
        if os.path.isdir(subpath):
            m = re.match(r"^(\d+)\s+(\d+)$", item.strip())
            if m:
                has_files = bool(glob.glob(os.path.join(subpath, "*.npy")) or glob.glob(os.path.join(subpath, "*.mat")))
                if has_files:
                    load, rpm = _normalize_rpm_load(int(m.group(1)), int(m.group(2)))
                    keys.add((load, rpm))
        elif item.endswith(".npy") or item.endswith(".mat"):
            m = group_pattern.search(item)
            if m:
                load, rpm = _normalize_rpm_load(int(m.group(1)), int(m.group(2)))
                keys.add((load, rpm))

    return keys


# ---------------------------------------------------------------------------
# 核心数据集类
# ---------------------------------------------------------------------------

# 工况未知时的占位标识
UNKNOWN_GROUP = (-1, -1)


class BearingSignalDataset(Dataset):
    """
    轴承故障振动信号数据集。

    每个样本为 (signal, label)：
      - signal: Tensor, shape (1, L)   — 单通道 1D 振动信号
      - label:  Tensor, shape ()       — 故障类别标签 (int64)

    group_labels: 可选，shape (N, 2)，每行 [load, rpm]，用于按工况对齐反归一化。
                  未知工况用 (-1, -1)。
    """

    def __init__(
        self,
        signals: Tensor,
        labels: Tensor,
        group_labels: Optional[np.ndarray] = None,
    ) -> None:
        """
        参数
        ----
        signals : Tensor, shape (N, 1, L)
            N 条振动信号。
        labels : Tensor, shape (N,)
            每条信号对应的故障类别标签。
        group_labels : np.ndarray, shape (N, 2)，可选
            每行 [load, rpm]，工况标识。
        """
        assert signals.ndim == 3 and signals.shape[1] == 1, \
            f"signals 应为 (N, 1, L)，实际为 {tuple(signals.shape)}"
        assert labels.ndim == 1 and len(labels) == len(signals), \
            f"labels 长度 ({len(labels)}) 与 signals 长度 ({len(signals)}) 不匹配"

        self.signals = signals.float()
        self.labels = labels.long()
        self.group_labels = group_labels  # (N, 2) 或 None

    def __len__(self) -> int:
        return len(self.signals)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        # 实例级 Z-score 归一化：按整条信号零均值、单位方差，削弱绝对幅值记忆、突出波形形状
        sig = self.signals[idx].clone()
        sig = (sig - sig.mean()) / (sig.std() + 1e-8)
        return sig, self.labels[idx]

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
        balance_groups: bool = False,
        groups_per_class: Optional[int] = None,
        seed: int = 42,
        seq_length: int = 1024,
        overlap: float = 0.5,
    ) -> "BearingSignalDataset":
        """
        从按类别分文件夹加载，支持 .npy 与 .mat，按类或按组抽样。

        支持格式：.npy（单样本）或 .mat（长序列，自动分割为 seq_length 样本）。
        文件名若含 "RPM Load"（如 "xxx 1000 0.npy"），则按组抽样。

        参数补充
        -------
        balance_groups : bool
            是否对各类别使用相同的“有效工况组数”。仅当 max_per_group 不为 None 时生效。
        groups_per_class : int | None
            每个类别保留的工况组数；None 时在 balance_groups=True 下自动取各类别可解析组数的最小值。
        """
        rng = np.random.default_rng(seed)
        if class_names is None:
            class_names = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )

        target_groups_per_class = groups_per_class
        if balance_groups and max_per_group is not None and target_groups_per_class is None:
            valid_group_counts: list[int] = []
            for cname in class_names:
                folder = os.path.join(root_dir, cname)
                n_groups = len(_collect_group_keys_in_class_folder(folder))
                if n_groups > 0:
                    valid_group_counts.append(n_groups)
            if valid_group_counts:
                # 自动对齐到可解析工况组最少的类别，确保可严格均衡
                target_groups_per_class = int(min(valid_group_counts))

        # 匹配 "数字 数字" 作为 (load, rpm)
        group_pattern = re.compile(r"(\d+)\s+(\d+)(?:\D|$)")

        all_signals: list[np.ndarray] = []
        all_labels: list[int] = []
        all_group_labels: list[Tuple[int, int]] = []

        for label_idx, cname in enumerate(class_names):
            folder = os.path.join(root_dir, cname)
            if not os.path.isdir(folder):
                continue
            files_by_group: dict[tuple[int, int], list[tuple[str, np.ndarray]]] = {}
            ungrouped: list[tuple[str, np.ndarray]] = []

            for item in sorted(os.listdir(folder)):
                subpath = os.path.join(folder, item)
                if os.path.isdir(subpath):
                    # 子目录如 "1000 0"：解析为组，加载其内 *.npy 或 *.mat
                    m = re.match(r"^(\d+)\s+(\d+)$", item.strip())
                    grp_key = _normalize_rpm_load(int(m.group(1)), int(m.group(2))) if m and max_per_group is not None else None
                    for fp in sorted(glob.glob(os.path.join(subpath, "*.npy"))):
                        sig = np.load(fp).astype(np.float32)
                        if sig.ndim == 1:
                            sig = sig[np.newaxis, :]
                        elif sig.ndim == 2 and sig.shape[0] != 1:
                            sig = sig[:1, :]
                        if grp_key is not None:
                            if grp_key not in files_by_group:
                                files_by_group[grp_key] = []
                            files_by_group[grp_key].append((item, sig))
                        else:
                            ungrouped.append((item, sig))
                    for fp in sorted(glob.glob(os.path.join(subpath, "*.mat"))):
                        try:
                            for sig in _load_mat_as_samples(fp, seq_length=seq_length, overlap=overlap):
                                if grp_key is not None:
                                    if grp_key not in files_by_group:
                                        files_by_group[grp_key] = []
                                    files_by_group[grp_key].append((item, sig))
                                else:
                                    ungrouped.append((item, sig))
                        except Exception:
                            continue
                elif item.endswith(".npy"):
                    fp = subpath
                    sig = np.load(fp).astype(np.float32)
                    if sig.ndim == 1:
                        sig = sig[np.newaxis, :]
                    elif sig.ndim == 2 and sig.shape[0] != 1:
                        sig = sig[:1, :]
                    m = group_pattern.search(item)
                    if m and max_per_group is not None:
                        load, rpm = _normalize_rpm_load(int(m.group(1)), int(m.group(2)))
                        key = (load, rpm)
                        if key not in files_by_group:
                            files_by_group[key] = []
                        files_by_group[key].append((item, sig))
                    else:
                        ungrouped.append((item, sig))
                elif item.endswith(".mat"):
                    try:
                        samples = _load_mat_as_samples(subpath, seq_length=seq_length, overlap=overlap)
                    except Exception:
                        continue
                    m = group_pattern.search(item)
                    if m and max_per_group is not None:
                        load, rpm = _normalize_rpm_load(int(m.group(1)), int(m.group(2)))
                        key = (load, rpm)
                        if key not in files_by_group:
                            files_by_group[key] = []
                        for sig in samples:
                            files_by_group[key].append((item, sig))
                    else:
                        for sig in samples:
                            ungrouped.append((item, sig))

            # 抽样：优先按组，再按类；同时保留工况标识 (load, rpm) 用于后续按工况反归一化
            collected: list[Tuple[np.ndarray, Tuple[int, int]]] = []
            if max_per_group is not None and files_by_group:
                if balance_groups and target_groups_per_class is not None and target_groups_per_class > 0:
                    grp_keys = list(files_by_group.keys())
                    rng.shuffle(grp_keys)
                    grp_keys = grp_keys[:min(target_groups_per_class, len(grp_keys))]
                    files_by_group = {k: files_by_group[k] for k in grp_keys}
                for grp_key, lst in files_by_group.items():
                    rng.shuffle(lst)
                    for _, sig in lst[:max_per_group]:
                        collected.append((sig, grp_key))
                for _, sig in ungrouped:
                    collected.append((sig, UNKNOWN_GROUP))
            else:
                for grp_key, grp in files_by_group.items():
                    for _, sig in grp:
                        collected.append((sig, grp_key))
                for _, sig in ungrouped:
                    collected.append((sig, UNKNOWN_GROUP))

            if balance_groups and max_per_group is not None and target_groups_per_class is not None and target_groups_per_class > 0:
                # 对“仅文件级且无法解析工况”的类别同样生效，避免其样本数远大于其他类别
                target_samples = target_groups_per_class * max_per_group
                if len(collected) > target_samples:
                    rng.shuffle(collected)
                    collected = collected[:target_samples]

            if max_per_class is not None and len(collected) > max_per_class:
                rng.shuffle(collected)
                collected = collected[:max_per_class]
            for sig, grp_key in collected:
                all_signals.append(sig)
                all_labels.append(label_idx)
                all_group_labels.append(grp_key)

        if not all_signals:
            # 诊断：检查各子目录是否有文件
            hint_parts = []
            for cname in class_names:
                folder = os.path.join(root_dir, cname)
                if os.path.isdir(folder):
                    npy = len(glob.glob(os.path.join(folder, "*.npy")))
                    mat = len(glob.glob(os.path.join(folder, "*.mat")))
                    for sub in os.listdir(folder):
                        sp = os.path.join(folder, sub)
                        if os.path.isdir(sp):
                            npy += len(glob.glob(os.path.join(sp, "*.npy")))
                            mat += len(glob.glob(os.path.join(sp, "*.mat")))
                    if npy or mat:
                        hint_parts.append(f"{cname}: {npy}个.npy, {mat}个.mat（解析可能失败）")
                    else:
                        hint_parts.append(f"{cname}: 无 .npy/.mat 文件")
                else:
                    hint_parts.append(f"{cname}: 目录不存在")
            hint = "; ".join(hint_parts) if hint_parts else "未找到故障子目录"
            raise FileNotFoundError(
                f"No .npy or .mat files found under {root_dir}. "
                "真实数据需为 .npy 或 .mat（SDUST 格式），置于各故障子目录内。"
                f"\n诊断: {hint}"
            )

        signals = np.stack(all_signals, axis=0)
        labels = np.array(all_labels, dtype=np.int64)
        group_labels_arr = np.array(all_group_labels, dtype=np.int64)  # (N, 2)
        return cls(
            signals=torch.from_numpy(signals),
            labels=torch.from_numpy(labels),
            group_labels=group_labels_arr,
        )


def strict_balance_dataset(dataset: BearingSignalDataset, seed: int = 42) -> BearingSignalDataset:
    """
    强制类别平衡：对数据集中出现的每个故障类别，无放回随机抽样至相同数量（各类 = min_count）。

    min_count 取各类别样本数的最小值，保证训练/测试/生成子集内类别比例严格 1:1:1…
    """
    rng = np.random.default_rng(seed)
    labels = dataset.labels.numpy()
    classes = np.unique(labels)
    if len(classes) == 0:
        return dataset
    counts = {int(c): int(np.sum(labels == c)) for c in classes}
    min_count = min(counts.values())
    if min_count < 1:
        raise ValueError("strict_balance_dataset: 存在空类别或无效样本数，无法平衡")

    idx_keep: list[np.ndarray] = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        idx_keep.append(idx_c[:min_count])
    sel = np.concatenate(idx_keep)
    rng.shuffle(sel)

    new_signals = dataset.signals.numpy()[sel]
    new_labels = labels[sel]
    gl = dataset.group_labels
    new_grps = gl[sel] if gl is not None else None
    return BearingSignalDataset(
        signals=torch.from_numpy(new_signals.copy()),
        labels=torch.from_numpy(new_labels.copy()),
        group_labels=new_grps.copy() if new_grps is not None else None,
    )


def _align_generated_to_real_train(
    real_train: BearingSignalDataset,
    gen_train: BearingSignalDataset,
    seed: int = 42,
) -> BearingSignalDataset:
    """
    将生成训练集按“类别+工况”分布对齐到真实训练集。

    目标：
      - len(gen_train_aligned) == len(real_train)
      - 生成集工况种类与真实训练集一致（以 real_train 中出现的组为准）
      - 各 (class, group) 样本数与 real_train 一致

    若某个 (class, group) 在生成集中样本不足，则采用有放回抽样补齐，
    以保证规模与分布严格对齐。
    """
    rng = np.random.default_rng(seed)

    real_lbl = real_train.labels.numpy().astype(np.int64)
    gen_lbl = gen_train.labels.numpy().astype(np.int64)
    real_grp = real_train.group_labels
    gen_grp = gen_train.group_labels

    # 若缺失工况标签，退化为按类别对齐
    if real_grp is None:
        real_grp = np.full((len(real_train), 2), UNKNOWN_GROUP[0], dtype=np.int64)
    if gen_grp is None:
        gen_grp = np.full((len(gen_train), 2), UNKNOWN_GROUP[0], dtype=np.int64)

    target_counts: Dict[Tuple[int, int, int], int] = {}
    for i in range(len(real_lbl)):
        key = (int(real_lbl[i]), int(real_grp[i, 0]), int(real_grp[i, 1]))
        target_counts[key] = target_counts.get(key, 0) + 1

    src_indices: Dict[Tuple[int, int, int], np.ndarray] = {}
    for key in target_counts.keys():
        c, load, rpm = key
        mask = (
            (gen_lbl == c)
            & (gen_grp[:, 0] == load)
            & (gen_grp[:, 1] == rpm)
        )
        src_indices[key] = np.where(mask)[0]

    # 兜底：若某些目标组在生成集中完全缺失，直接报错提示目录/工况缺失
    missing = [k for k, v in src_indices.items() if len(v) == 0]
    if missing:
        miss_txt = ", ".join([f"(class={c}, load={l}, rpm={r})" for c, l, r in missing[:12]])
        if len(missing) > 12:
            miss_txt += f", ... 共{len(missing)}个"
        raise ValueError(
            "生成训练集无法对齐到真实训练集：以下类别-工况在生成数据中缺失: "
            f"{miss_txt}"
        )

    selected_idx_parts: list[np.ndarray] = []
    for key, n_target in target_counts.items():
        idx = src_indices[key]
        replace = len(idx) < n_target
        chosen = rng.choice(idx, size=n_target, replace=replace)
        selected_idx_parts.append(chosen.astype(np.int64))

    selected_idx = np.concatenate(selected_idx_parts, axis=0)
    rng.shuffle(selected_idx)

    sig = gen_train.signals.numpy()[selected_idx]
    lbl = gen_lbl[selected_idx]
    grp = gen_grp[selected_idx]
    return BearingSignalDataset(
        signals=torch.from_numpy(sig.copy()),
        labels=torch.from_numpy(lbl.copy()),
        group_labels=grp.copy(),
    )


# ---------------------------------------------------------------------------
# 按工况物理量纲还原（Condition-wise Physical Dimension Restoration）
# ---------------------------------------------------------------------------

def denormalize_gen_by_condition(
    real_signals: np.ndarray,
    real_labels: np.ndarray,
    real_group_labels: Optional[np.ndarray],
    gen_signals: np.ndarray,
    gen_labels: np.ndarray,
    gen_group_labels: Optional[np.ndarray],
    amplitude_metric: str = "mean_max_abs",
) -> np.ndarray:
    """
    按工况对齐的物理量纲还原：将生成信号（模型输出 [-1,1]）乘以对应工况的真实物理幅值。

    训练时采用逐段 max-abs 归一化（seg / max_abs），模型输出为无量纲波形。
    本函数按 (故障类 c, 工况 g=(load,rpm)) 计算真实幅值 A_real，并还原：X_denorm = X_gen * A_real。

    参数
    ----
    real_signals : (N, 1, L) 真实物理幅值
    real_labels  : (N,) 故障类别
    real_group_labels : (N, 2) 每行 [load, rpm]，-1 表示未知
    gen_signals  : (M, 1, L) 生成信号，值域约 [-1, 1]
    gen_labels   : (M,) 故障类别
    gen_group_labels : (M, 2) 工况标识
    amplitude_metric : "mean_max_abs"（每段 max|·| 的均值）或 "p99"（99 分位 |·|）

    返回
    ----
    gen_denorm : (M, 1, L) 反归一化后的生成信号
    """
    gen_denorm = np.empty_like(gen_signals, dtype=np.float32)
    n_gen = len(gen_signals)

    def _amp_real(sigs: np.ndarray) -> float:
        """计算幅值：贴合训练时的 max-abs 逻辑"""
        if amplitude_metric == "mean_max_abs":
            # 每段 max|·| 的均值，与训练归一化一致
            per_seg = np.max(np.abs(sigs.reshape(-1, sigs.shape[-1])), axis=1)
            return float(np.mean(per_seg) + 1e-8)
        if amplitude_metric == "p99":
            return float(np.percentile(np.abs(sigs), 99) + 1e-8)
        return float(np.mean(np.max(np.abs(sigs.reshape(-1, sigs.shape[-1])), axis=1)) + 1e-8)

    def _max_val(sigs: np.ndarray) -> float:
        """该组真实信号绝对值上界，用于物理截断"""
        return float(np.max(np.abs(sigs)) + 1e-8)

    # 工况未知或缺失时，退化为按类匹配
    if real_group_labels is None or gen_group_labels is None:
        real_group_labels = np.full((len(real_signals), 2), UNKNOWN_GROUP[0], dtype=np.int64)
        gen_group_labels = np.full((len(gen_signals), 2), UNKNOWN_GROUP[0], dtype=np.int64)

    # 预计算每个 (c, load, rpm) 的 A_real 和 max_val
    cache: dict[Tuple[int, int, int], Tuple[float, float]] = {}
    all_grps = np.vstack([real_group_labels, gen_group_labels])
    seen_grps = {tuple(row) for row in all_grps}
    for c in np.unique(np.concatenate([real_labels, gen_labels])).astype(int):
        for (load, rpm) in seen_grps:
            load, rpm = int(load), int(rpm)
            if (load, rpm) == UNKNOWN_GROUP:
                mr = real_labels == c
            else:
                mr = (
                    (real_labels == c)
                    & (real_group_labels[:, 0] == load)
                    & (real_group_labels[:, 1] == rpm)
                )
            if not np.any(mr):
                continue
            sigs_r = real_signals[mr]
            A = _amp_real(sigs_r)
            M = _max_val(sigs_r)
            cache[(c, load, rpm)] = (A, M)

    # 全局/类级兜底：当某 (c,g) 无真实样本时使用
    def _fallback_amp(label: int, load: int, rpm: int) -> Tuple[float, float]:
        c = int(label)
        if (load, rpm) != UNKNOWN_GROUP and (c, load, rpm) in cache:
            return cache[(c, load, rpm)]
        # 按类
        mr = real_labels == c
        if np.any(mr):
            sigs = real_signals[mr]
            return _amp_real(sigs), _max_val(sigs)
        # 全局
        return _amp_real(real_signals), _max_val(real_signals)

    for i in range(n_gen):
        sig = gen_signals[i].astype(np.float32)
        c = int(gen_labels[i])
        load = int(gen_group_labels[i, 0])
        rpm = int(gen_group_labels[i, 1])
        A, max_val = _fallback_amp(c, load, rpm)
        # 去均值（0 频置零）后按物理幅值缩放，再物理截断
        sig_centered = sig - np.mean(sig)
        out = sig_centered * A
        gen_denorm[i] = np.clip(out, -max_val, max_val)

    return gen_denorm


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


def load_real_and_gen_for_eval(
    real_data_path: str,
    gen_data_folder: str,
    train_ratio: float = 0.7,
    seed: int = 42,
    class_names: Optional[List[str]] = None,
    gen_class_names: Optional[List[str]] = None,
    max_per_class: Optional[int] = None,
    max_per_group: Optional[int] = None,
    balance_groups: bool = False,
    groups_per_class: Optional[int] = None,
    align_gen_to_real_train: bool = True,
    seq_length: int = 1024,
    overlap: float = 0.5,
) -> Tuple["BearingSignalDataset", "BearingSignalDataset", "BearingSignalDataset", int, List[str]]:
    """
    加载真实数据与生成数据，使用相同的 max_per_class、max_per_group 抽样逻辑。
    真实数据按 train_ratio 划分为 train/test；生成数据全部作为 gen_train。

    当 align_gen_to_real_train=True 时，会将 gen_train 进一步按“类别+工况”
    分布对齐到 real_train，使其样本总量和工况种类与真实训练集一致。

    gen_class_names : 生成数据根目录下的子文件夹名列表；例如 CVAE 基线为
    ``class_0`` … ``class_{K-1}``，须与真实类别数 K 一致、且与真实类顺序一一对应。

    目录结构：真实 root/类别名/*.npy 或 .mat；生成侧可为 ``class_i/*.npy``。
    支持 .mat（长序列自动分割为 seq_length 样本）与 .npy。
    """
    if not os.path.isdir(real_data_path):
        raise FileNotFoundError(f"真实数据目录不存在: {real_data_path}")
    if not os.path.isdir(gen_data_folder):
        raise FileNotFoundError(f"生成数据目录不存在: {gen_data_folder}")

    rng = np.random.default_rng(seed)
    subdirs = [d for d in os.listdir(real_data_path)
               if os.path.isdir(os.path.join(real_data_path, d))]
    npy_in_root = len(glob.glob(os.path.join(real_data_path, "*.npy")))

    if not subdirs and not npy_in_root:
        raise FileNotFoundError(f"真实数据目录为空或不存在: {real_data_path}")

    if class_names is None:
        class_names = sorted(subdirs) if subdirs else ["类别0"]
    num_classes = len(class_names)

    gnames = gen_class_names if gen_class_names is not None else class_names
    if len(gnames) != num_classes:
        raise ValueError(
            f"gen_class_names 长度 ({len(gnames)}) 须与真实数据类别数 ({num_classes}) 一致"
        )

    # 统一加载逻辑：from_class_folders_with_subsample（支持 .npy 与 .mat）
    def _load(root: str, cn: List[str]) -> "BearingSignalDataset":
        return BearingSignalDataset.from_class_folders_with_subsample(
            root, class_names=cn,
            max_per_class=max_per_class,
            max_per_group=max_per_group,
            balance_groups=balance_groups,
            groups_per_class=groups_per_class,
            seed=seed,
            seq_length=seq_length, overlap=overlap,
        )

    real_full = _load(real_data_path, class_names)
    gen_train = _load(gen_data_folder, gnames)

    # 分层划分 real -> train / test，并保留工况标签 group_labels
    train_sigs, train_lbls, train_grps = [], [], []
    test_sigs, test_lbls, test_grps = [], [], []
    for c in range(num_classes):
        mask = real_full.labels.numpy() == c
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
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
        if real_full.group_labels is not None:
            train_grps.append(real_full.group_labels[idx[:n_train]])
            test_grps.append(real_full.group_labels[idx[n_train:]])

    if not train_sigs:
        raise FileNotFoundError(f"真实数据加载后无样本: {real_data_path}")

    train_grps_cat = np.concatenate(train_grps, axis=0) if train_grps else None
    test_grps_cat = np.concatenate(test_grps, axis=0) if test_grps else None
    real_train = BearingSignalDataset(
        torch.from_numpy(np.concatenate(train_sigs, axis=0)),
        torch.from_numpy(np.concatenate(train_lbls, axis=0)),
        group_labels=train_grps_cat,
    )
    real_test = BearingSignalDataset(
        torch.from_numpy(np.concatenate(test_sigs, axis=0)),
        torch.from_numpy(np.concatenate(test_lbls, axis=0)),
        group_labels=test_grps_cat,
    )
    # 强制类别平衡：真实训练 / 真实测试各自内部按类对齐到 min_count（独立随机种子）
    real_train = strict_balance_dataset(real_train, seed=seed + 11)
    real_test = strict_balance_dataset(real_test, seed=seed + 13)

    # 生成训练集与真实训练集对齐：总样本数 + 类别工况分布一致
    if align_gen_to_real_train:
        gen_train = _align_generated_to_real_train(real_train, gen_train, seed=seed + 17)
    else:
        gen_train = strict_balance_dataset(gen_train, seed=seed + 17)
    return real_train, gen_train, real_test, num_classes, class_names


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


