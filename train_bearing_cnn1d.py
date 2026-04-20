#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
轴承 / HIL 1D 多架构诊断训练脚本（v2 — 文件级划分，消除时序泄露）。

核心改进（相对旧版）：
  1. 文件级 Train / Val / Test 划分（TEST_RATIO=0.20, VAL_RATIO=0.10）。
     同一 .mat 文件的全部窗口仅属于三者之一，彻底消除同文件时序泄露。
  2. 验证集与测试集 100% 纯真实数据（仅来自 .mat 文件），val_acc 真实
     反映泛化能力，不再因混入生成数据而虚高。
  3. 模型 1（混合）与模型 2（纯真实）共用同一组纯真实 val / test，
     保证公平评估基准。
  4. 混合训练集采用曼哈顿距离梯度配比：
       real_frac = max(0, 0.5 − dist × 0.125)
       dist 0（核心 1800rpm/40load）→ 50% 真实；dist 4（最边缘）→ 0% 真实。
  5. 模型 2 的训练集 = 模型 1 训练集的 .mat 精确子集（包括边缘工况的极少样本）。
  6. 评估时额外输出按曼哈顿距离分组的准确率（核心 / 过渡 / 边缘独立精度）。
  7. 支持 9 种神经网络架构（CNN / Transformer / TCN / MobileNet / ResNet /
     ShuffleNet / Conformer / ConvNeXt / RepVGG），由 model.py 中的
     SelectModel 变量统一控制；RepVGG checkpoint 会写入标准 architecture
     字段，便于 HIL / 推理脚本自动识别并部署。

数据假设：
  真实数据：DATA_ROOT/<类别>/**/*.mat
            （.mat 文件命名含 RPM 和 Load，如 "IF0.2 1800 40.mat"）
  生成数据：GEN_DATA_ROOT/<类别>/load_X/rpm_Y/filtered_*.npy

运行模式（__main__）：
  默认        train_two_models()   — model_mixed_<type>.pth + model_real_only_<type>.pth
  --single    train()              — best_model_<type>.pth（传统单模型，窗口级划分）
  --eval-only eval_only()          — 仅评估 best_model_<type>.pth
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

# =============================================================================
# 架构选择 — 修改 model.py 中的 SelectModel 变量来切换模型
# =============================================================================
import model as _model_module

MODEL_TYPE: str = _model_module.SelectModel   # 从 model.py 统一读取

SCRIPT_DIR = Path(__file__).resolve().parent

# =============================================================================
# 数据路径
# =============================================================================
DATA_ROOT     = r"D:\data\轴承数据集"
GEN_DATA_ROOT = str(SCRIPT_DIR / "generated_grid_train")

# =============================================================================
# 信号处理
# =============================================================================
WINDOW_SIZE          = 1024   # 与 receive_udp_hil 推理窗长一致
STRIDE               = 1024   # 非重叠滑窗；减小可增加样本但引入时序相关
MAX_WINDOWS_PER_FILE = 20     # 每个 .mat/.npy 文件最多保留的滑窗数
NORMALIZE_PER_WINDOW = True   # 每窗口 z-score 标准化

# =============================================================================
# 数据集划分（文件级，彻底消除时序泄露）
# =============================================================================
TEST_RATIO = 0.20   # 每工况 .mat 文件中 20% → 共用纯真实测试集
VAL_RATIO  = 0.10   # 每工况 .mat 文件中 10% → 共用纯真实验证集
# 剩余约 70% → 训练文件池（模型 1 + 模型 2 混合采样）

# =============================================================================
# 混合策略：曼哈顿距离梯度配比（模型 1）
# =============================================================================
TOTAL_WINDOWS_PER_CONDITION = 128   # 每个工况目标训练窗口总数

# 曼哈顿距离映射（d_rpm + d_load = total_dist，最大 4）
# 中心点：1800 rpm / 40 load
_D_RPM_MAP:  dict[int, int] = {1000: 2, 1500: 1, 1800: 0, 2000: 1, 2500: 2}
_D_LOAD_MAP: dict[int, int] = {0: 2, 20: 1, 40: 0, 60: 1}

# 边缘工况阈值：total_dist >= EDGE_DIST_THRESHOLD 视为"边缘"，单独汇报精度
EDGE_DIST_THRESHOLD = 2

# 文件级划分阈值：每工况 .mat 文件数 >= 此值时使用严格文件级划分；
# 否则（典型情况：每工况仅 1 个大 .mat 文件）自动退回时序分段划分
# （train=前70%窗口 | val=中10% | test=后20%，每个窗口仍仅属于一个集合）
_MIN_FILES_FOR_FILE_SPLIT = 5

# =============================================================================
# 训练超参数
# =============================================================================
BATCH_SIZE          = 32
EPOCHS              = 200
LEARNING_RATE       = 1e-3
WEIGHT_DECAY        = 1e-4
EARLY_STOP_PATIENCE = 10
SEED                = 42
DEVICE              = "cuda"
TRAIN_IN_TRAINVAL   = 0.8    # 仅用于 --single 单模型模式

# =============================================================================
# 输出文件名（含架构后缀，多架构实验并排保存）
# =============================================================================
EXPECTED_NUM_CLASSES        = 10
OUTPUT_MODEL_PATH           = f"best_model_{MODEL_TYPE}.pth"
OUTPUT_MODEL_PATH_MIXED     = f"model_mixed_{MODEL_TYPE}.pth"
OUTPUT_MODEL_PATH_REAL_ONLY = f"model_real_only_{MODEL_TYPE}.pth"

# 生成过程元数据文件，不进入训练集
_GEN_SKIP_FILES = {"kl_scores.npy", "selected_idx.npy", "all_generated.npy"}


# =============================================================================
# 通用工具
# =============================================================================

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
    从 .mat 文件加载一维振动信号。
    Signal.y_values.values 为 (N, C) double，取第一通道（列 0）。
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy 未安装，请运行: pip install scipy")
    data = loadmat(str(mat_path))
    arr  = data['Signal']['y_values'][0, 0]['values']
    mat  = arr.item() if (arr.dtype == object and arr.shape == (1, 1)) else arr
    signal = (mat[:, 0] if mat.ndim == 2 else mat.ravel()).astype(np.float64)
    if signal.size == 0:
        raise ValueError(f"信号长度为 0: {mat_path.name}")
    return signal


def load_signal_from_npy(npy_path: Path) -> np.ndarray:
    """从 .npy 加载一维振动序列（支持 generated_grid_train 中的 filtered_*.npy）。"""
    try:
        arr = np.load(str(npy_path), allow_pickle=False)
    except Exception as e:
        raise ValueError(f"np.load 失败: {e}") from e
    if arr.dtype == object:
        raise ValueError("不支持 object 数组")
    arr = np.squeeze(np.asarray(arr, dtype=np.float64))
    if arr.ndim == 0:
        raise ValueError(f"标量数组，shape={arr.shape}")
    return arr.reshape(-1)


def collect_class_folders(root: Path) -> list[tuple[str, int]]:
    """返回 [(文件夹名, 类别索引), ...]，按文件夹名排序（保证标签稳定）。"""
    if not root.is_dir():
        raise FileNotFoundError(f"数据根目录不存在: {root}")
    subs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not subs:
        raise FileNotFoundError(f"{root} 下没有子文件夹")
    if len(subs) != EXPECTED_NUM_CLASSES:
        print(f"[警告] 子文件夹数={len(subs)}，与 EXPECTED_NUM_CLASSES={EXPECTED_NUM_CLASSES} 不一致。")
    return [(p.name, i) for i, p in enumerate(subs)]


class BearingNpyWindowDataset:
    """简易数据集：__getitem__ 读 .npy / .mat 切片（带文件级缓存）。"""

    def __init__(self, index: list[tuple[Path, int, int]],
                 cache: dict[str, np.ndarray], normalize: bool):
        self.index     = index
        self._cache    = cache
        self.normalize = normalize

    def __len__(self) -> int:
        return len(self.index)

    def _get_signal(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key not in self._cache:
            self._cache[key] = (
                _load_signal_from_mat(path) if path.suffix.lower() == ".mat"
                else load_signal_from_npy(path)
            )
        return self._cache[key]

    def __getitem__(self, i: int):
        import torch
        path, start, label = self.index[i]
        sig = self._get_signal(path)
        w   = sig[start: start + WINDOW_SIZE].astype(np.float32).copy()
        if self.normalize:
            w = (w - w.mean()) / (w.std() + 1e-6)
        return torch.from_numpy(w).unsqueeze(0), label


# =============================================================================
# 路径解析工具
# =============================================================================

def _rpm_load_from_two_ints(a: int, b: int) -> tuple[int, int]:
    """按数值大小区分 rpm（大）与 load（小），返回 (rpm, load)。"""
    return (a, b) if a >= b else (b, a)


def _parse_condition_from_path(p: Path) -> tuple[int | None, int | None]:
    """
    从路径中解析 (rpm, load)，按优先级依次尝试三种格式：
      格式 A（目录前缀）:   .../load_20/rpm_1800/...
      格式 C（.mat 文件名）:.../IF0.2 1800 40.mat → 末尾两个整数
      格式 B（两整数目录名）:.../1800 40/...
    """
    rpm: int | None = None
    load: int | None = None
    for part in p.parts:
        pl = part.lower()
        if pl.startswith("load_"):
            try: load = int(part[5:])
            except ValueError: pass
        elif pl.startswith("rpm_"):
            try: rpm = int(part[4:])
            except ValueError: pass
    if rpm is not None and load is not None:
        return rpm, load

    if p.suffix.lower() == ".mat":
        ints = [int(t) for t in p.stem.split() if t.lstrip("-").isdigit()]
        if len(ints) >= 2:
            return _rpm_load_from_two_ints(ints[-2], ints[-1])

    for part in p.parts:
        tokens = part.strip().split()
        if len(tokens) == 2:
            try: return _rpm_load_from_two_ints(int(tokens[0]), int(tokens[1]))
            except ValueError: pass

    return rpm, load


# =============================================================================
# 曼哈顿距离梯度混合策略
# =============================================================================

def _nearest_dist(value: int, dist_map: dict[int, int]) -> int:
    """
    将 value 映射到 dist_map 中最近键所对应的距离级别。
    unknown 值取最近已知键的距离。
    """
    if value in dist_map:
        return dist_map[value]
    return dist_map[min(dist_map, key=lambda k: abs(k - value))]


def _compute_manhattan_dist(rpm: int, load: int) -> int:
    """计算工况 (rpm, load) 距中心 (1800rpm, 40load) 的曼哈顿距离（0–4）。"""
    return _nearest_dist(rpm, _D_RPM_MAP) + _nearest_dist(load, _D_LOAD_MAP)


def _compute_manhattan_real_frac(rpm: int, load: int) -> float:
    """
    曼哈顿距离梯度真实数据占比：
      real_frac = max(0.0, 0.5 − dist × 0.125)
      dist 0 (核心 1800/40) → 0.50   dist 1 → 0.375
      dist 2               → 0.25   dist 3 → 0.125
      dist 4 (最边缘)      → 0.00
    """
    return max(0.0, 0.5 - _compute_manhattan_dist(rpm, load) * 0.125)


# =============================================================================
# 文件级数据收集函数
# =============================================================================

def _collect_mat_files_by_condition(
    class_dir: Path,
) -> dict[tuple[int, int], list[Path]]:
    """
    扫描类别目录下所有 .mat 文件（真实数据），
    按解析到的 (load, rpm) 工况键分组。
    无法解析工况的文件归入键 (-1, -1)。
    """
    by_cond: dict[tuple[int, int], list[Path]] = {}
    for fp in sorted(class_dir.rglob("*.mat")):
        rpm, load = _parse_condition_from_path(fp)
        key = (load if load is not None else -1,
               rpm  if rpm  is not None else -1)
        by_cond.setdefault(key, []).append(fp)
    return by_cond


def _collect_gen_files_by_condition(
    class_dir: Path,
) -> dict[tuple[int, int], list[Path]]:
    """
    扫描类别目录下所有 .npy 文件（生成数据，跳过元数据），
    按工况键分组。
    """
    by_cond: dict[tuple[int, int], list[Path]] = {}
    for fp in sorted(class_dir.rglob("*.npy")):
        if fp.name in _GEN_SKIP_FILES:
            continue
        rpm, load = _parse_condition_from_path(fp)
        key = (load if load is not None else -1,
               rpm  if rpm  is not None else -1)
        by_cond.setdefault(key, []).append(fp)
    return by_cond


def _windows_from_file_list(
    file_list:  list[Path],
    label:      int,
    rng:        random.Random,
    sig_cache:  dict[str, np.ndarray] | None = None,
) -> list[tuple[Path, int, int]]:
    """
    从文件列表中提取所有有效滑窗，返回 [(path, start, label), ...]。
    使用 sig_cache 共享内存，同一文件在一次训练流程中只加载一次。
    每个文件最多保留 MAX_WINDOWS_PER_FILE 个滑窗（随机无放回下采样）。
    """
    if sig_cache is None:
        sig_cache = {}
    windows: list[tuple[Path, int, int]] = []
    for fp in file_list:
        key = str(fp.resolve())
        if key not in sig_cache:
            try:
                sig_cache[key] = (
                    _load_signal_from_mat(fp) if fp.suffix.lower() == ".mat"
                    else load_signal_from_npy(fp)
                )
            except Exception as e:
                print(f"  [跳过] {fp.name}: {e}", file=sys.stderr)
                continue
        sig = sig_cache[key]
        n   = len(sig)
        if n < WINDOW_SIZE:
            continue
        starts = list(range(0, n - WINDOW_SIZE + 1, STRIDE))
        if len(starts) > MAX_WINDOWS_PER_FILE:
            starts = rng.sample(starts, MAX_WINDOWS_PER_FILE)
        for s in starts:
            windows.append((fp, s, label))
    return windows


def _temporal_windows_from_file_list(
    file_list:  list[Path],
    label:      int,
    sig_cache:  dict[str, np.ndarray] | None = None,
) -> list[tuple[Path, int, int]]:
    """
    从文件列表提取滑窗，保持严格时序顺序（不随机打乱）。

    用于"单文件工况"的时序分段划分回退策略：
      - 对于超过 MAX_WINDOWS_PER_FILE 的文件，采用等间距下采样（保持时序覆盖均匀）
      - 多文件时按文件名排序后拼接，保证顺序确定
      - 调用方按返回列表的顺序切片即可得到 train / val / test 三段，
        每个 (path, start) 窗口仍仅属于一个集合（无重复）

    与 _windows_from_file_list 的区别：
      不随机 sample → 等间距 sample，保留时序单调性。
    """
    if sig_cache is None:
        sig_cache = {}
    windows: list[tuple[Path, int, int]] = []
    for fp in sorted(file_list, key=lambda p: p.name):
        key = str(fp.resolve())
        if key not in sig_cache:
            try:
                sig_cache[key] = (
                    _load_signal_from_mat(fp) if fp.suffix.lower() == ".mat"
                    else load_signal_from_npy(fp)
                )
            except Exception as e:
                print(f"  [跳过] {fp.name}: {e}", file=sys.stderr)
                continue
        sig = sig_cache[key]
        n   = len(sig)
        if n < WINDOW_SIZE:
            continue
        all_starts = list(range(0, n - WINDOW_SIZE + 1, STRIDE))
        if len(all_starts) > MAX_WINDOWS_PER_FILE:
            # 等间距下采样：保持时序覆盖均匀，不破坏单调性
            step = len(all_starts) / MAX_WINDOWS_PER_FILE
            all_starts = [all_starts[int(i * step)] for i in range(MAX_WINDOWS_PER_FILE)]
        for s in all_starts:
            windows.append((fp, s, label))
    return windows   # 时序单调递增（按文件名 → 按起点位置）


def _sample_pool(pool: list, n: int, rng: random.Random) -> list:
    """
    从 pool 取 n 个样本：
      pool 充足 → 无放回随机抽样；
      pool 不足 → 有放回补足至 n（允许重复）；
      pool 为空或 n<=0 → 返回 []。
    """
    if not pool or n <= 0:
        return []
    if len(pool) >= n:
        return rng.sample(pool, n)
    result = list(pool)
    while len(result) < n:
        result.extend(rng.choices(pool, k=min(len(pool), n - len(result))))
    return result[:n]


# =============================================================================
# 文件级索引构建（双模型核心函数）
# =============================================================================

def build_file_level_indices(
    root_real:  Path,
    root_gen:   Path | None,
    rng:        random.Random,
    sig_cache:  dict[str, np.ndarray] | None = None,
) -> tuple[list, list, list, list, list[str], list[tuple[int, int]]]:
    """
    按文件级别划分 Train / Val / Test，构建双模型所需的六个对象。

    返回值
    ------
    mixed_train_idx  : list[(Path, start, label)]  模型 1 训练集（真实+生成混合）
    real_train_idx   : list[(Path, start, label)]  模型 2 训练集（.mat 精确子集）
    shared_val_idx   : list[(Path, start, label)]  共用纯真实验证集
    shared_test_idx  : list[(Path, start, label)]  共用纯真实测试集
    class_names      : list[str]                   类别名（与标签索引对应）
    test_cond_keys   : list[(load, rpm)]            与 shared_test_idx 等长的工况键

    划分规则（每工况独立执行）
    --------------------------
    test_files  = 前 round(n_files × TEST_RATIO)  个文件的全部窗口
    val_files   = 次 round(n_files × VAL_RATIO)   个文件的全部窗口
    train_files = 剩余约 70% 文件
    ★ 同一 .mat 文件的任何窗口只属于三者之一

    混合策略（模型 1 训练）
    ----------------------
    n_real = round(128 × real_frac)    real_frac = max(0, 0.5 − dist × 0.125)
    n_gen  = 128 − n_real
    真实不足时有放回补采；余量由生成数据补齐。
    """
    if sig_cache is None:
        sig_cache = {}

    class_list  = collect_class_folders(root_real)
    class_names = [name for name, _ in class_list]

    gen_available = root_gen is not None and root_gen.is_dir()
    if not gen_available:
        print(f"[警告] 生成数据目录不存在: {root_gen}，模型 1 退化为全真实。",
              file=sys.stderr)

    mixed_train_idx: list = []
    shared_val_idx:  list = []
    shared_test_idx: list = []
    test_cond_keys:  list[tuple[int, int]] = []

    for folder_name, label in class_list:
        real_cls_dir = root_real / folder_name
        gen_cls_dir  = (root_gen / folder_name) if gen_available else None

        real_by_cond = _collect_mat_files_by_condition(real_cls_dir)
        gen_by_cond  = (
            _collect_gen_files_by_condition(gen_cls_dir)
            if (gen_cls_dir and gen_cls_dir.is_dir()) else {}
        )

        all_conds = set(real_by_cond) | set(gen_by_cond)

        # per-class logging counters
        n_real_tr = n_gen_tr = n_val = n_test = 0

        for cond_key in sorted(all_conds):
            load_v, rpm_v = cond_key
            real_files = real_by_cond.get(cond_key, [])
            gen_files  = gen_by_cond.get(cond_key, [])

            # ── 划分策略：文件数充足 → 文件级；单/少文件 → 时序分段 ─────────
            shuffled = list(real_files)
            rng.shuffle(shuffled)
            nf = len(shuffled)

            if nf >= _MIN_FILES_FOR_FILE_SPLIT:
                # ★ 路径 A：多文件工况 — 按文件边界严格划分（无文件内时序泄露）
                n_test_f = min(round(nf * TEST_RATIO), nf)
                n_val_f  = min(round(nf * VAL_RATIO),  max(0, nf - n_test_f))
                test_files  = shuffled[:n_test_f]
                val_files   = shuffled[n_test_f: n_test_f + n_val_f]
                train_files = shuffled[n_test_f + n_val_f:]
                val_wins        = _windows_from_file_list(val_files,  label, rng, sig_cache)
                test_wins       = _windows_from_file_list(test_files, label, rng, sig_cache)
                train_real_wins = _windows_from_file_list(train_files, label, rng, sig_cache)

            elif nf > 0:
                # ★ 路径 B：单/少文件工况（本数据集每工况仅 1 个 .mat 的情况）
                #   按时序分段：train=前70% | val=中10% | test=后20%
                #   等间距下采样保持时序均匀；每个(path,start)仍仅属于一个集合
                all_wins  = _temporal_windows_from_file_list(shuffled, label, sig_cache)
                n_w       = len(all_wins)
                n_test_w  = max(0, round(n_w * TEST_RATIO))
                n_val_w   = max(0, round(n_w * VAL_RATIO))
                n_test_w  = min(n_test_w, n_w)
                n_val_w   = min(n_val_w,  max(0, n_w - n_test_w))
                n_train_w = n_w - n_test_w - n_val_w
                # 时序顺序：训练用早期信号，测试用最新信号
                train_real_wins = all_wins[:n_train_w]
                val_wins        = all_wins[n_train_w: n_train_w + n_val_w]
                test_wins       = all_wins[n_train_w + n_val_w:]

            else:
                # 无真实文件：val/test/train 均为空（纯生成工况）
                train_real_wins = val_wins = test_wins = []

            # ── 提取 val / test 窗口（纯真实）————— 汇入共用集合 ──────────
            shared_val_idx.extend(val_wins)
            shared_test_idx.extend(test_wins)
            test_cond_keys.extend([cond_key] * len(test_wins))
            n_val  += len(val_wins)
            n_test += len(test_wins)

            # ── 计算训练目标数量（曼哈顿距离配比）───────────────────────
            if load_v == -1 or rpm_v == -1:
                # 无法解析工况：全量训练文件直接加入（不受 TOTAL_WINDOWS 限制）
                mixed_train_idx.extend(train_real_wins)
                n_real_tr += len(train_real_wins)
                continue

            real_frac  = _compute_manhattan_real_frac(rpm_v, load_v)
            n_real_tgt = round(TOTAL_WINDOWS_PER_CONDITION * real_frac)
            n_gen_tgt  = TOTAL_WINDOWS_PER_CONDITION - n_real_tgt

            gen_wins = _windows_from_file_list(gen_files, label, rng, sig_cache)
            # train_real_wins 已在上面的划分块中计算完毕

            sampled_real = _sample_pool(train_real_wins, n_real_tgt, rng)
            sampled_gen  = _sample_pool(gen_wins,        n_gen_tgt,  rng)

            mixed_train_idx.extend(sampled_real)
            mixed_train_idx.extend(sampled_gen)
            n_real_tr += len(sampled_real)
            n_gen_tr  += len(sampled_gen)

        print(
            f"[信息] 类别 {folder_name}: "
            f"训练(真实={n_real_tr} 生成={n_gen_tr} "
            f"总={n_real_tr + n_gen_tr}) | "
            f"验证={n_val} | 测试={n_test}"
        )

    if not mixed_train_idx:
        raise RuntimeError("混合训练集为空。请检查 DATA_ROOT 与 GEN_DATA_ROOT。")
    if not shared_test_idx:
        raise RuntimeError("共用测试集为空。请检查 DATA_ROOT 下是否有足够的 .mat 文件。")
    if not shared_val_idx:
        print("[警告] 共用验证集为空（每工况文件数太少）。尝试增大数据集或降低 VAL_RATIO。",
              file=sys.stderr)

    # 模型 2：直接从模型 1 训练索引中提取 .mat 精确子集
    # 即使边缘工况下数量极少（如 5 窗），也全部保留，体现无生成数据时的性能崩塌
    real_train_idx = [
        e for e in mixed_train_idx
        if Path(e[0]).suffix.lower() == ".mat"
    ]

    print(
        f"\n[统计] 模型1训练={len(mixed_train_idx)} | "
        f"模型2训练={len(real_train_idx)} | "
        f"共用验证={len(shared_val_idx)} | "
        f"共用测试={len(shared_test_idx)}"
    )

    # 打印混合策略摘要
    c0 = round(TOTAL_WINDOWS_PER_CONDITION * _compute_manhattan_real_frac(1800, 40))
    c4 = round(TOTAL_WINDOWS_PER_CONDITION * _compute_manhattan_real_frac(1000, 0))
    print(
        f"[混合策略] 每工况总={TOTAL_WINDOWS_PER_CONDITION} | "
        f"核心(dist=0): 真实{c0}+生成{TOTAL_WINDOWS_PER_CONDITION - c0} | "
        f"最边缘(dist=4): 真实{c4}+生成{TOTAL_WINDOWS_PER_CONDITION - c4}\n"
    )

    return (mixed_train_idx, real_train_idx,
            shared_val_idx, shared_test_idx,
            class_names, test_cond_keys)


# =============================================================================
# 评估工具
# =============================================================================

def evaluate_accuracy(model, data_loader, device) -> float:
    """返回该 DataLoader 上的分类准确率。"""
    import torch
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in data_loader:
            pred     = model(xb.to(device)).argmax(dim=-1)
            correct += int((pred == yb.to(device)).sum())
            total   += yb.numel()
    return correct / max(total, 1)


def _get_all_predictions(
    model, data_loader, device
) -> tuple[np.ndarray, np.ndarray]:
    """
    单次推理，返回 (predictions, true_labels) 两个 int64 numpy 数组。
    顺序与 data_loader 中的样本顺序一致（shuffle=False 时与索引对应）。
    """
    import torch
    model.eval()
    preds_list, labels_list = [], []
    with torch.no_grad():
        for xb, yb in data_loader:
            preds_list.append(model(xb.to(device)).argmax(dim=-1).cpu().numpy())
            labels_list.append(yb.numpy())
    if not preds_list:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(preds_list), np.concatenate(labels_list)


def _preds_to_cm(
    preds: np.ndarray, labels: np.ndarray, num_classes: int
) -> np.ndarray:
    """由预测数组和标签数组计算混淆矩阵，形状 (num_classes, num_classes)。"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels.flat, preds.flat):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1
    return cm


# 保留 evaluate_confusion_matrix 供 eval_only() 使用
def evaluate_confusion_matrix(
    model, data_loader, device, num_classes: int
) -> np.ndarray:
    preds, labels = _get_all_predictions(model, data_loader, device)
    return _preds_to_cm(preds, labels, num_classes)


def print_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    """在控制台打印混淆矩阵（行=真实，列=预测）+ 各类召回率。"""
    n = cm.shape[0]
    if cm.shape != (n, n) or len(class_names) != n:
        print(f"[警告] 混淆矩阵形状 {cm.shape} 与类别数 {len(class_names)} 不一致。")
        print(cm); return

    col_w   = max(5, len(str(int(cm.max()))) + 1)
    label_w = min(max(len(nm) for nm in class_names), 14)
    print("真实 \\ 预测".ljust(label_w + 5)
          + "".join(f"{j:>{col_w}}" for j in range(n)))
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
            rs = int(cm[i].sum())
            r  = cm[i, i] / rs if rs > 0 else 0.0
            print(f"  [{i}] {class_names[i]}: {r:.4f}  (n={rs})")


def print_per_distance_accuracy(
    preds:     np.ndarray,
    labels:    np.ndarray,
    cond_keys: list[tuple[int, int]],
) -> None:
    """
    按曼哈顿距离分组打印测试集准确率（核心 / 过渡 / 边缘）。
    cond_keys 与 preds / labels 的长度必须相同（与 shared_test_idx 等长）。
    """
    if len(preds) != len(cond_keys):
        print(f"[警告] preds({len(preds)}) 与 cond_keys({len(cond_keys)}) "
              "长度不一致，跳过距离分析。")
        return

    dist_groups: dict[int, list[int]] = {}
    for i, (load_v, rpm_v) in enumerate(cond_keys):
        d = _compute_manhattan_dist(rpm_v, load_v)
        dist_groups.setdefault(d, []).append(i)

    print("  [工况分组精度] 曼哈顿距离梯度（中心：1800rpm / 40load）:")
    print(f"  {'dist':>5}  {'类型':4}  {'正确/总数':>12}  {'准确率':>8}")
    edge_correct = edge_total = 0
    for d in sorted(dist_groups):
        idxs    = dist_groups[d]
        correct = int(sum(preds[i] == labels[i] for i in idxs))
        total   = len(idxs)
        acc     = correct / total if total > 0 else 0.0
        tag     = "核心" if d == 0 else ("边缘" if d >= EDGE_DIST_THRESHOLD else "过渡")
        print(f"  {d:>5}  {tag:4}  {correct:>5}/{total:<5}  {acc:>8.4f}")
        if d >= EDGE_DIST_THRESHOLD:
            edge_correct += correct
            edge_total   += total

    if edge_total > 0:
        print(
            f"  边缘汇总 (dist≥{EDGE_DIST_THRESHOLD}): "
            f"{edge_correct}/{edge_total} = {edge_correct / edge_total:.4f}"
        )
    print()


# =============================================================================
# 单模型 PyTorch 训练（双模型模式的核心子函数）
# =============================================================================

def train_one_model(
    train_idx:       list,
    class_names:     list[str],
    out_path:        Path,
    model_label:     str,
    device,
    cache:           dict,
    shared_val_idx:  list,
    shared_test_idx: list,
    test_cond_keys:  list[tuple[int, int]] | None = None,
) -> float:
    """
    训练单个 PyTorch 模型（架构由全局 MODEL_TYPE 决定）。

    - 超参数从 model.get_hparams(MODEL_TYPE) 读取（每架构独立配置）。
    - 支持线性 warmup + CosineAnnealing 调度器、梯度裁剪、label smoothing。
    - 早停监控 shared_val_idx（外部纯真实验证集）的 val_acc。
    - 最终在 shared_test_idx（外部纯真实测试集）上评估一次，并按距离分组打印。

    返回最终测试集准确率。
    """
    import math
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from model import arch_name, build_model, count_feature_layers, get_hparams

    num_classes = len(class_names)

    # ── 读取本架构专属超参 ───────────────────────────────────────────────────
    hp       = get_hparams(MODEL_TYPE)
    lr       = hp["lr"]
    wd       = hp["weight_decay"]
    batch_sz = hp["batch_size"]
    n_epochs = hp["epochs"]
    patience = hp["patience"]
    warmup   = hp["warmup_epochs"]
    clip     = hp["grad_clip"]
    lbl_sm   = hp["label_smoothing"]

    if not shared_val_idx:
        print(f"[错误] {model_label} 验证集为空，跳过训练。", file=sys.stderr)
        return 0.0
    if not shared_test_idx:
        print(f"[错误] {model_label} 测试集为空，跳过训练。", file=sys.stderr)
        return 0.0

    print(
        f"[{model_label}] 样本数 — "
        f"训练: {len(train_idx)} | "
        f"验证: {len(shared_val_idx)} (纯真实) | "
        f"测试: {len(shared_test_idx)} (共用纯真实) | "
        f"类别数: {num_classes}"
    )

    def collate(batch):
        import torch as _t
        return (
            _t.stack([b[0] for b in batch]),
            _t.tensor([b[1] for b in batch], dtype=_t.long),
        )

    dl_train = DataLoader(
        BearingNpyWindowDataset(list(train_idx), cache, NORMALIZE_PER_WINDOW),
        batch_size=batch_sz, shuffle=True, num_workers=0, collate_fn=collate,
    )
    dl_val = DataLoader(
        BearingNpyWindowDataset(list(shared_val_idx), cache, NORMALIZE_PER_WINDOW),
        batch_size=batch_sz, shuffle=False, num_workers=0, collate_fn=collate,
    )
    dl_test = DataLoader(
        BearingNpyWindowDataset(list(shared_test_idx), cache, NORMALIZE_PER_WINDOW),
        batch_size=batch_sz, shuffle=False, num_workers=0, collate_fn=collate,
    )

    model    = build_model(MODEL_TYPE, num_classes).to(device)
    n_layers = count_feature_layers(model)
    print(
        f"[{model_label}] {type(model).__name__} | "
        f"特征提取层数={n_layers} | 设备: {device} | "
        f"epochs={n_epochs} patience={patience}"
    )
    print(
        f"  [超参] lr={lr} wd={wd} batch={batch_sz} "
        f"warmup={warmup} clip={clip} ls={lbl_sm}"
    )

    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(label_smoothing=lbl_sm)

    # ── 调度器：线性 warmup → CosineAnnealing（单一 LambdaLR，无需 SequentialLR）
    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            # 线性预热：epoch 0 → lr×(1/warmup)，epoch warmup-1 → lr×1.0
            return float(epoch + 1) / float(max(warmup, 1))
        # 余弦退火：从 warmup 结束到 n_epochs
        cos_progress = (epoch - warmup) / float(max(n_epochs - warmup, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cos_progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    best_val_acc = -1.0
    patience_cnt = 0
    stopped_early = False

    for epoch in range(n_epochs):
        model.train()
        total_loss = n_seen = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n_seen     += xb.size(0)
        sched.step()

        train_loss = total_loss / max(n_seen, 1)
        val_acc    = evaluate_accuracy(model, dl_val, device)
        improved   = val_acc > best_val_acc + 1e-7

        if improved:
            best_val_acc = val_acc
            patience_cnt = 0
            ckpt = {
                "architecture":         arch_name(MODEL_TYPE),
                "state_dict":           model.state_dict(),
                "num_classes":          num_classes,
                "class_names":          class_names,
                "window_size":          WINDOW_SIZE,
                "normalize_per_window": NORMALIZE_PER_WINDOW,
                "best_val_acc":         best_val_acc,
                "epoch":                epoch + 1,
                "model_type":           model_label,
            }
            torch.save(ckpt, str(out_path))
            print(
                f"  [{model_label}] Epoch {epoch+1:03d}/{n_epochs} | "
                f"loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
                f"【保存】-> {out_path.name}"
            )
        else:
            patience_cnt += 1
            if (epoch + 1) % 20 == 0 or patience_cnt >= patience:
                print(
                    f"  [{model_label}] Epoch {epoch+1:03d}/{n_epochs} | "
                    f"loss={train_loss:.4f} | val_acc={val_acc:.4f} | "
                    f"早停计数 {patience_cnt}/{patience}"
                )

        if patience_cnt >= patience:
            print(f"  [{model_label}] 早停触发（连续 {patience} epoch 无提升）。")
            stopped_early = True
            break

    if not stopped_early:
        print(f"  [{model_label}] 已达最大 epoch={n_epochs}，未触发早停。")

    # ── 加载验证集最优权重，最终在共用纯真实测试集上评估一次 ────────────────
    if out_path.is_file():
        try:
            ck = torch.load(str(out_path), map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(str(out_path), map_location=device)
        model.load_state_dict(ck["state_dict"])

    preds, labels_arr = _get_all_predictions(model, dl_test, device)
    test_acc = float((preds == labels_arr).mean()) if len(preds) > 0 else 0.0
    cm       = _preds_to_cm(preds, labels_arr, num_classes)

    print(
        f"\n  [{model_label}] 测试集准确率（验证集最优权重）: "
        f"{test_acc:.4f} | val_best={best_val_acc:.4f}"
    )
    print(f"  [{model_label}] 模型已保存: {out_path.resolve()}")
    print(f"  [{model_label}] 混淆矩阵（行=真实，列=预测）:")
    print_confusion_matrix(cm, class_names)

    # 按曼哈顿距离分组打印精度（核心 / 过渡 / 边缘）
    if test_cond_keys is not None:
        print_per_distance_accuracy(preds, labels_arr, test_cond_keys)

    return test_acc


# =============================================================================
# 双模型训练主入口（默认运行模式）
# =============================================================================

def train_two_models() -> int:
    """
    构建文件级索引后，依次训练：
      模型 1 — 真实+生成混合（曼哈顿距离梯度配比）
      模型 2 — 模型 1 训练集中的 .mat 精确子集（纯真实）
    两个模型共用同一组纯真实 val / test，保证评估基准一致。
    """
    set_seed(SEED)
    try:
        import torch
    except ImportError:
        print("[错误] 需要 PyTorch: pip install torch", file=sys.stderr)
        return 1

    from model import N_FEATURE_LAYERS

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and DEVICE == "cuda":
        print("[警告] CUDA 不可用，使用 CPU 训练。")

    root_real = Path(DATA_ROOT)
    root_gen  = Path(GEN_DATA_ROOT)

    print("=" * 60)
    print(f"  双模型训练（文件级划分 · 曼哈顿距离混合策略）")
    print(f"  架构: {MODEL_TYPE.upper()}  （修改 model.py → SelectModel 切换）")
    print(f"  真实数据: {root_real}")
    print(f"  生成数据: {root_gen}")
    print(f"  划分比例: TEST={TEST_RATIO:.0%}  VAL={VAL_RATIO:.0%}"
          f"  TRAIN≈{1 - TEST_RATIO - VAL_RATIO:.0%}（文件级）")
    print(f"  混合配比: core(dist=0)→50%真实 … edge(dist=4)→0%真实")
    print("=" * 60 + "\n")

    sig_cache: dict[str, np.ndarray] = {}
    rng_build = random.Random(SEED)

    (mixed_idx, real_idx,
     shared_val_idx, shared_test_idx,
     class_names, test_cond_keys) = build_file_level_indices(
        root_real, root_gen, rng_build, sig_cache
    )

    out_mixed = SCRIPT_DIR / OUTPUT_MODEL_PATH_MIXED
    out_real  = SCRIPT_DIR / OUTPUT_MODEL_PATH_REAL_ONLY
    use_rf    = MODEL_TYPE.lower() == "rf"
    tag       = f"[{MODEL_TYPE.upper()}]"

    # ── 模型 1：混合 ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{tag}[模型1] 真实+生成混合 -> {out_mixed.name}")
    print(f"{'=' * 60}")
    set_seed(SEED)
    if use_rf:
        acc_mixed = train_rf_model(
            mixed_idx, class_names, out_mixed,
            "模型1-混合-RF", sig_cache, test_index=shared_test_idx,
        )
    else:
        acc_mixed = train_one_model(
            mixed_idx, class_names, out_mixed, "模型1-混合",
            device, sig_cache, shared_val_idx, shared_test_idx, test_cond_keys,
        )

    # ── 模型 2：纯真实 ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"{tag}[模型2] 纯真实 -> {out_real.name}")
    print(f"{'=' * 60}")
    set_seed(SEED)
    if use_rf:
        acc_real = train_rf_model(
            real_idx, class_names, out_real,
            "模型2-真实-RF", sig_cache, test_index=shared_test_idx,
        )
    else:
        acc_real = train_one_model(
            real_idx, class_names, out_real, "模型2-真实",
            device, sig_cache, shared_val_idx, shared_test_idx, test_cond_keys,
        )

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  双模型完成 | 共用纯真实测试集 {len(shared_test_idx)} 条")
    print(f"  架构: {MODEL_TYPE.upper()} | 特征层数: "
          f"{N_FEATURE_LAYERS if not use_rf else 'N/A（随机森林）'}")
    print(f"{'=' * 60}")
    print(f"  模型1（混合）测试准确率: {acc_mixed:.4f}  ({acc_mixed*100:.2f}%)")
    print(f"           {out_mixed.resolve()}")
    print(f"  模型2（真实）测试准确率: {acc_real:.4f}  ({acc_real*100:.2f}%)")
    print(f"           {out_real.resolve()}")
    print(f"\n  receive_udp_hil.py 中可设:")
    print(f'    MODEL_PATH = "{OUTPUT_MODEL_PATH_MIXED}"')
    print(f'    MODEL_PATH = "{OUTPUT_MODEL_PATH_REAL_ONLY}"')
    print("=" * 60)
    return 0


# =============================================================================
# RF 专用辅助函数（保持原有接口不变）
# =============================================================================

def _materialize_windows_for_rf(
    index: list,
    cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    from model import extract_rf_features
    X_list: list = []
    y_list: list = []
    for path, start, label in index:
        path = Path(path)
        key  = str(path.resolve())
        if key not in cache:
            try:
                cache[key] = (
                    _load_signal_from_mat(path) if path.suffix.lower() == ".mat"
                    else load_signal_from_npy(path)
                )
            except Exception as e:
                print(f"  [RF跳过] {path.name}: {e}", file=sys.stderr)
                continue
        sig = cache[key]
        if len(sig) < start + WINDOW_SIZE:
            continue
        w = sig[start: start + WINDOW_SIZE].astype(np.float32)
        if NORMALIZE_PER_WINDOW:
            w = (w - w.mean()) / (w.std() + 1e-6)
        X_list.append(extract_rf_features(w))
        y_list.append(label)
    if not X_list:
        return np.empty((0, 26), dtype=np.float64), np.empty(0, dtype=np.int64)
    return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.int64)


def train_rf_model(
    train_index: list,
    class_names: list[str],
    out_path:    Path,
    model_label: str,
    cache:       dict,
    test_index:  list | None = None,
) -> float:
    import torch
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("[错误] 随机森林需要 scikit-learn: pip install scikit-learn", file=sys.stderr)
        return 0.0
    from model import ARCH_RF, BearingRFWrapper

    num_classes = len(class_names)
    print(f"\n[{model_label}] 提取 RF 特征（训练集 {len(train_index)} 窗口）…")
    X_train, y_train = _materialize_windows_for_rf(train_index, cache)
    if len(y_train) == 0:
        print(f"[{model_label}] 训练集为空，跳过。", file=sys.stderr); return 0.0

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    train_acc = float((rf.predict(X_train) == y_train).mean())
    print(f"[{model_label}] 训练集准确率: {train_acc:.4f}")

    test_acc = train_acc
    if test_index:
        X_test, y_test = _materialize_windows_for_rf(test_index, cache)
        if len(y_test) > 0:
            test_acc = float((rf.predict(X_test) == y_test).mean())
            print(f"[{model_label}] 测试集准确率: {test_acc:.4f}")

    wrapper = BearingRFWrapper(rf, num_classes)
    ckpt = {
        "architecture":         ARCH_RF,
        "rf_wrapper":           wrapper,
        "num_classes":          num_classes,
        "class_names":          class_names,
        "window_size":          WINDOW_SIZE,
        "normalize_per_window": NORMALIZE_PER_WINDOW,
        "model_type":           model_label,
    }
    torch.save(ckpt, str(out_path))
    print(f"[{model_label}] 已保存: {out_path.resolve()}")
    return test_acc


# =============================================================================
# 以下为向后兼容的辅助函数（供 --single / --eval-only 模式使用）
# =============================================================================

def _split_counts_one_class(
    n: int, test_ratio: float, train_frac_of_rest: float
) -> tuple[int, int, int]:
    if n <= 0: return 0, 0, 0
    if n == 1: return 1, 0, 0
    if n == 2: return 1, 1, 0
    n_test = min(max(0, int(round(n * test_ratio))), n - 2)
    if n_test == 0 and test_ratio > 1e-9:
        n_test = min(1, n - 2)
    rest    = n - n_test
    if rest < 2:
        n_test, rest = max(0, n - 2), n - max(0, n - 2)
    n_val   = max(1, min(rest - 1, int(round(rest * (1.0 - train_frac_of_rest)))))
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
    rng = random.Random(seed)
    by_label: dict[int, list] = {}
    for item in index:
        by_label.setdefault(item[2], []).append(item)

    train_idx, val_idx, test_idx = [], [], []
    for _label in sorted(by_label):
        items = by_label[_label][:]
        rng.shuffle(items)
        n_train, n_val, n_test = _split_counts_one_class(
            len(items), test_ratio, train_frac_of_rest
        )
        test_idx.extend(items[:n_test])
        val_idx.extend(items[n_test: n_test + n_val])
        train_idx.extend(items[n_test + n_val:])

    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def build_window_index_wrapped(
    root: Path,
) -> tuple[list[tuple[Path, int, int]], list[str]]:
    class_list  = collect_class_folders(root)
    class_names = [name for name, _ in class_list]
    rng  = random.Random(SEED)
    index: list[tuple[Path, int, int]] = []
    for folder_name, label in class_list:
        folder = root / folder_name
        for npy_path in sorted(folder.rglob("*.npy")):
            try:
                sig = load_signal_from_npy(npy_path)
            except Exception as e:
                print(f"[跳过] {npy_path}: {e}", file=sys.stderr); continue
            n = int(sig.shape[0])
            if n < WINDOW_SIZE: continue
            starts = list(range(0, n - WINDOW_SIZE + 1, STRIDE))
            if len(starts) > MAX_WINDOWS_PER_FILE:
                starts = rng.sample(starts, MAX_WINDOWS_PER_FILE)
            for s in starts:
                index.append((npy_path, s, label))
    if not index:
        raise RuntimeError("没有可用窗口（请检查 DATA_ROOT 下是否有 .npy 文件）")
    print(
        f"[信息] 共 {len(class_names)} 类: {class_names} | "
        f"窗口总数: {len(index)}（每 .npy 最多 {MAX_WINDOWS_PER_FILE} 窗）"
    )
    return index, class_names


# =============================================================================
# --single 模式：传统单模型训练（窗口级划分，向后兼容）
# =============================================================================

def train() -> int:
    set_seed(SEED)
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("[错误] 需要 PyTorch: pip install torch", file=sys.stderr); return 1

    from model import arch_name, build_model, count_feature_layers

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
        f"[信息] 样本数 — 训练: {len(train_list)} | "
        f"验证: {len(val_list)} | 测试: {len(test_list)} | 类别数: {num_classes}"
    )

    if MODEL_TYPE.lower() == "rf":
        out_path = SCRIPT_DIR / OUTPUT_MODEL_PATH
        test_acc = train_rf_model(train_list, class_names, out_path, "单模型-RF", cache, test_list)
        print(f"\n[完成] RF 已保存: {out_path.resolve()} | 测试准确率: {test_acc:.4f}")
        return 0

    if not val_list:
        print("[错误] 验证集为空。", file=sys.stderr); return 1
    if not test_list:
        print("[错误] 测试集为空。", file=sys.stderr); return 1

    def collate(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
        )

    dl_train = DataLoader(
        BearingNpyWindowDataset(train_list, cache, NORMALIZE_PER_WINDOW),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate,
    )
    dl_val = DataLoader(
        BearingNpyWindowDataset(val_list, cache, NORMALIZE_PER_WINDOW),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate,
    )
    dl_test = DataLoader(
        BearingNpyWindowDataset(test_list, cache, NORMALIZE_PER_WINDOW),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate,
    )

    import math
    from model import get_hparams
    hp       = get_hparams(MODEL_TYPE)
    lr       = hp["lr"];    wd       = hp["weight_decay"]
    batch_sz = hp["batch_size"]
    n_epochs = hp["epochs"]; patience = hp["patience"]
    warmup   = hp["warmup_epochs"]
    clip     = hp["grad_clip"]; lbl_sm = hp["label_smoothing"]

    # 重建 DataLoader（使用架构对应 batch_size）
    dl_train = DataLoader(BearingNpyWindowDataset(train_list, cache, NORMALIZE_PER_WINDOW),
                          batch_size=batch_sz, shuffle=True,  num_workers=0, collate_fn=collate)
    dl_val   = DataLoader(BearingNpyWindowDataset(val_list,   cache, NORMALIZE_PER_WINDOW),
                          batch_size=batch_sz, shuffle=False, num_workers=0, collate_fn=collate)
    dl_test  = DataLoader(BearingNpyWindowDataset(test_list,  cache, NORMALIZE_PER_WINDOW),
                          batch_size=batch_sz, shuffle=False, num_workers=0, collate_fn=collate)

    model    = build_model(MODEL_TYPE, num_classes).to(device)
    n_layers = count_feature_layers(model)
    print(f"[模型] {type(model).__name__} | 特征提取层数={n_layers} | MODEL_TYPE={MODEL_TYPE}")
    print(f"  [超参] lr={lr} wd={wd} batch={batch_sz} epochs={n_epochs} patience={patience} "
          f"warmup={warmup} clip={clip} ls={lbl_sm}")

    opt  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    crit = nn.CrossEntropyLoss(label_smoothing=lbl_sm)

    def _lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return float(epoch + 1) / float(max(warmup, 1))
        cos_progress = (epoch - warmup) / float(max(n_epochs - warmup, 1))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * cos_progress)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    best_val_acc = -1.0
    patience_cnt = 0
    out_path = SCRIPT_DIR / OUTPUT_MODEL_PATH

    for epoch in range(n_epochs):
        model.train()
        total_loss = n_seen = 0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            if clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            total_loss += float(loss.item()) * xb.size(0)
            n_seen     += xb.size(0)
        sched.step()

        train_loss = total_loss / max(n_seen, 1)
        val_acc    = evaluate_accuracy(model, dl_val, device)
        improved   = val_acc > best_val_acc + 1e-7
        if improved:
            best_val_acc = val_acc; patience_cnt = 0
            ckpt = {
                "architecture": arch_name(MODEL_TYPE), "state_dict": model.state_dict(),
                "num_classes": num_classes, "class_names": class_names,
                "window_size": WINDOW_SIZE, "normalize_per_window": NORMALIZE_PER_WINDOW,
                "best_val_acc": best_val_acc, "epoch": epoch + 1,
            }
            torch.save(ckpt, str(out_path))
            print(f"Epoch {epoch+1:03d}/{n_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | 【保存】-> {out_path.name}")
        else:
            patience_cnt += 1
            print(f"Epoch {epoch+1:03d}/{n_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f} | 早停计数 {patience_cnt}/{patience}")

        if patience_cnt >= patience:
            print(f"[早停] 验证集准确率已连续 {patience} epoch 未提升。")
            break

    if out_path.is_file():
        try:
            ck = torch.load(str(out_path), map_location=device, weights_only=False)
        except TypeError:
            ck = torch.load(str(out_path), map_location=device)
        model.load_state_dict(ck["state_dict"])

    final_test_acc = evaluate_accuracy(model, dl_test, device)
    cm = evaluate_confusion_matrix(model, dl_test, device, num_classes)
    print(f"\n[最终] 测试集准确率: {final_test_acc:.4f} | val_best={best_val_acc:.4f}")
    print_confusion_matrix(cm, class_names)
    return 0


# =============================================================================
# --eval-only 模式（仅评估已保存的单模型 checkpoint）
# =============================================================================

def eval_only() -> int:
    set_seed(SEED)
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError:
        print("[错误] 需要 PyTorch: pip install torch", file=sys.stderr); return 1

    root = Path(DATA_ROOT)
    index, class_names = build_window_index_wrapped(root)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    num_classes = len(class_names)
    cache: dict[str, np.ndarray] = {}

    _, _, test_list = stratified_train_val_test(index, TEST_RATIO, TRAIN_IN_TRAINVAL, SEED)
    print(f"[信息] 测试集: {len(test_list)} 条 | 类别数: {num_classes}")
    if not test_list:
        print("[错误] 测试集为空。", file=sys.stderr); return 1

    def collate(batch):
        return (
            torch.stack([b[0] for b in batch]),
            torch.tensor([b[1] for b in batch], dtype=torch.long),
        )

    dl_test = DataLoader(
        BearingNpyWindowDataset(test_list, cache, NORMALIZE_PER_WINDOW),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate,
    )

    out_path = SCRIPT_DIR / OUTPUT_MODEL_PATH
    if not out_path.is_file():
        print(f"[错误] 未找到模型文件: {out_path.resolve()}", file=sys.stderr); return 1

    try:
        ck = torch.load(str(out_path), map_location=device, weights_only=False)
    except TypeError:
        ck = torch.load(str(out_path), map_location=device)

    ck_names = ck.get("class_names")
    if ck_names is not None and list(ck_names) != list(class_names):
        print("[警告] checkpoint class_names 与当前数据不一致，标签语义可能错位。")

    nc_ck = int(ck.get("num_classes", num_classes))
    if nc_ck != num_classes:
        print(f"[错误] checkpoint num_classes={nc_ck} 与当前 {num_classes} 不一致。",
              file=sys.stderr); return 1

    from model import build_model
    model = build_model(ck.get("architecture", "cnn1d_bearing_v1"), num_classes).to(device)
    model.load_state_dict(ck["state_dict"])

    test_acc = evaluate_accuracy(model, dl_test, device)
    cm = evaluate_confusion_matrix(model, dl_test, device, num_classes)
    print(f"\n[仅评估] 测试集准确率: {test_acc:.4f}  |  权重: {out_path.name}\n")
    print_confusion_matrix(cm, class_names)
    return 0


# =============================================================================
# 入口
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="轴承 1D 多架构诊断训练（默认：双模型文件级划分）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式说明：
  默认              train_two_models()  — 生成 model_mixed_<type>.pth + model_real_only_<type>.pth
  --single          train()             — 传统单模型训练（窗口级划分），生成 best_model_<type>.pth
  --eval-only       eval_only()         — 仅加载 best_model_<type>.pth 评估

架构切换：修改 model.py 顶部的 SelectModel 变量（不需要改动本文件）。
        """,
    )
    parser.add_argument("--eval-only", action="store_true",
                        help="不训练，仅评估 best_model_<type>.pth")
    parser.add_argument("--single", action="store_true",
                        help="传统单模型训练（生成 best_model_<type>.pth）")
    args = parser.parse_args()

    if args.eval_only:
        raise SystemExit(eval_only())
    elif args.single:
        raise SystemExit(train())
    else:
        raise SystemExit(train_two_models())
