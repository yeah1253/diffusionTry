"""
main.py — 数据质量评估全流程入口（对照实验：1D-CVAE 生成数据 vs 真实）

- 生成数据默认读取本目录（generative model）下 ``generated_samples_cvae``；
  若不存在则回退到仓库内 ``diffusionTry/generated_samples_cvae``（旧版 train_baselines 输出）。
- 评估结果（图、FID 等）写入 ``generative model/eval_results_cvae``。

数据模式：
  - USE_REAL_DATA=True：真实 + CVAE 生成
  - USE_REAL_DATA=False：Dummy 冒烟测试

评估模块：时频对比、TRTR/TSTR/TRTR-Augment/Gen-Only、t-SNE
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
_plt.rcParams["axes.unicode_minus"] = False
import warnings
warnings.filterwarnings("ignore", message=".*Glyph.*missing.*")

# 兼容两种启动方式：
# 1) python -m eval.main
# 2) python eval/main.py
if __package__ is None or __package__ == "":
    _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT_FOR_IMPORT = os.path.dirname(_THIS_DIR)
    if _PROJECT_ROOT_FOR_IMPORT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT_FOR_IMPORT)
    __package__ = "eval"

from .dataset import (
    make_dummy_datasets,
    load_real_and_gen_for_eval,
    denormalize_gen_by_condition,
)
from .eval_time_freq import run_time_freq_analysis
from .train_diagnosis import run_trtr_tstr
from .train_diagnosis_simple import run_trtr_tstr_simple
from .visualize_tsne import run_tsne_visualization


def main() -> None:
    # 配置
    SEQ_LENGTH = 1024
    FS = 25600.0
    TRAIN_RATIO = 0.7
    NUM_EPOCHS = 100
    BATCH_SIZE = 64
    LR = 5e-4
    LOW_FREQ_LIMIT = 1000.0 #频谱图横轴上限 (Hz)，只展示 0–1000 Hz，便于观察主要频率

    USE_SIMPLE_CLASSIFIER = False
    MAX_SAMPLES_PER_CLASS = 960 #每个故障类别最多保留的样本数
    MAX_SAMPLES_PER_GROUP = 40  #每个工况最多保留的样本数(40*24=960)
    BALANCE_GROUPS = True       # 各类别使用相同有效工况组数
    GROUPS_PER_CLASS = None     # None=自动取最小组数；可设为固定值如 6

    # generative model 根：.../controlled experiment/generative model
    _GM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _WORKSPACE_ROOT = os.path.dirname(os.path.dirname(_GM_ROOT))
    _DIFFTRY_ROOT = os.path.join(_WORKSPACE_ROOT, "diffusionTry")

    SAVE_DIR = os.path.join(_GM_ROOT, "eval_results_cvae")
    REAL_DATA_PATH = r"D:\data\轴承数据集"
    # 1D-CVAE 导出：class_0 … class_{K-1}，与 train_baselines 一致
    GEN_DATA_FOLDER = os.path.join(_GM_ROOT, "generated_samples_cvae")
    if not os.path.isdir(GEN_DATA_FOLDER):
        _fb = os.path.join(_DIFFTRY_ROOT, "generated_samples_cvae")
        if os.path.isdir(_fb):
            GEN_DATA_FOLDER = _fb
    USE_REAL_DATA = True
    NUM_CLASSES_DUMMY = 4
    SAMPLES_PER_CLASS_DUMMY = 80

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'═'*60}\n  轴承故障振动信号 — 数据质量评估（CVAE 生成基线）\n{'═'*60}")
    print(f"  设备: {device}  模式: {'真实+CVAE 生成' if USE_REAL_DATA else 'Dummy'}")
    print(f"  生成数据目录: {GEN_DATA_FOLDER}")
    print(f"  结果输出目录: {SAVE_DIR}\n")

    # 1. 加载数据集
    print("[1] 加载数据集...")
    t0 = time.time()
    if USE_REAL_DATA:
        _real_sub = sorted(
            d for d in os.listdir(REAL_DATA_PATH)
            if os.path.isdir(os.path.join(REAL_DATA_PATH, d))
        )
        _k = len(_real_sub) if _real_sub else 0
        gen_class_names = [f"class_{i}" for i in range(_k)] if _k else None
        real_train, gen_train, real_test, NUM_CLASSES, class_names = load_real_and_gen_for_eval(
            real_data_path=REAL_DATA_PATH,
            gen_data_folder=GEN_DATA_FOLDER,
            train_ratio=TRAIN_RATIO,
            seed=42,
            gen_class_names=gen_class_names,
            max_per_class=MAX_SAMPLES_PER_CLASS,
            max_per_group=MAX_SAMPLES_PER_GROUP,
            balance_groups=BALANCE_GROUPS,
            groups_per_class=GROUPS_PER_CLASS,
            # CVAE 按类扁平存放，无语境工况；不对齐真实 train 的 (类, 工况) 分布
            align_gen_to_real_train=False,
            seq_length=SEQ_LENGTH,
            overlap=0.5,
        )
    else:
        real_train, gen_train, real_test = make_dummy_datasets(
            num_classes=NUM_CLASSES_DUMMY,
            samples_per_class=SAMPLES_PER_CLASS_DUMMY,
            seq_length=SEQ_LENGTH,
            train_ratio=TRAIN_RATIO,
            seed=42,
        )
        NUM_CLASSES = NUM_CLASSES_DUMMY
        class_names = [f"故障类型 {i}" for i in range(NUM_CLASSES)]

    print(f"  真实 train/test: {len(real_train)}/{len(real_test)}  生成: {len(gen_train)}  类别: {NUM_CLASSES}")
    print(f"  耗时: {time.time() - t0:.2f}s\n")

    # =====================================================================
    # 2. 模块一：时频域物理特征对比分析
    # =====================================================================
    print("[2] 运行模块一：时频域物理特征对比分析...")
    t0 = time.time()

    # 从数据集中取出 numpy 数组
    real_signals_np = real_train.signals.numpy()  # (N, 1, L)，真实数据为物理幅值
    gen_signals_np = gen_train.signals.numpy()   # (M, 1, L)，生成数据为模型输出 [-1, 1]

    # 按工况物理量纲还原（Condition-wise Physical Dimension Restoration）
    # 训练时采用逐段 max-abs 归一化，模型输出为 [-1,1] 无量纲波形；
    # 按 (故障类 c, 工况 g=(load,rpm)) 对齐，用真实幅值 A_real 缩放：X_denorm = X_gen * A_real
    gen_signals_denorm = denormalize_gen_by_condition(
        real_signals=real_signals_np,
        real_labels=real_train.labels.numpy(),
        real_group_labels=real_train.group_labels,
        gen_signals=gen_signals_np,
        gen_labels=gen_train.labels.numpy(),
        gen_group_labels=gen_train.group_labels,
        amplitude_metric="mean_max_abs",  # 贴合训练 max-abs 逻辑
    )
    print(f"  生成信号按工况物理量纲还原完成（去均值 + 物理截断已包含）")

    run_time_freq_analysis(
        real_signals=real_signals_np,
        gen_signals=gen_signals_denorm,
        fs=FS,
        low_freq_limit=LOW_FREQ_LIMIT,
        num_display_waveforms=4,
        save_dir=SAVE_DIR,
        real_labels=real_train.labels.numpy(),
        real_group_labels=real_train.group_labels,
        gen_labels=gen_train.labels.numpy(),
        gen_group_labels=gen_train.group_labels,
        class_names=class_names,
    )
    print(f"  模块一耗时: {time.time() - t0:.2f}s\n")

    # =====================================================================
    # 3. 模块二：TRTR / TSTR / TRTR-Augment 下游分类对比
    # =====================================================================
    print("[3] 运行模块二：TRTR / TSTR / TRTR-Augment 下游分类对比...")
    t0 = time.time()

    if USE_SIMPLE_CLASSIFIER:
        results = run_trtr_tstr_simple(
            real_train=real_train,
            gen_train=gen_train,
            real_test=real_test,
            num_classes=NUM_CLASSES,
            class_names=class_names,
            classifier="rf",
        )
    else:
        results = run_trtr_tstr(
            real_train=real_train,
            gen_train=gen_train,
            real_test=real_test,
            num_classes=NUM_CLASSES,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            device=device,
        )
    print(f"  模块二耗时: {time.time() - t0:.2f}s\n")

    # =====================================================================
    # 4. 模块三：t-SNE 可视化（降维图）
    # =====================================================================
    print("[4] 运行模块三：t-SNE 特征空间可视化（降维图）...")
    t0 = time.time()

    tsne_save_path = os.path.join(SAVE_DIR, "tsne_visualization.png")

    # t-SNE 可视化 + FID（Fréchet 距离，替代易 NaN 的 KL）
    fid_score = run_tsne_visualization(
        model=results["model_trtr"],
        real_dataset=real_train,
        gen_dataset=gen_train,
        device=device,
        class_names=class_names,
        save_path=tsne_save_path,
        use_raw_features=False,
    )
    fid_path = os.path.join(SAVE_DIR, "fid_score.txt")
    with open(fid_path, "w", encoding="utf-8") as f:
        f.write("Fréchet Inception Distance (FID, 特征空间高斯假设)\n")
        f.write(f"FID(真实, 生成): {fid_score:.6f}\n")
    print(f"  FID 结果已保存到 {fid_path}")
    print(f"  模块三耗时: {time.time() - t0:.2f}s\n")

    # =====================================================================
    # 汇总
    # =====================================================================
    print(f"{'═'*60}")
    print(f"  全流程执行完毕！")
    print(f"{'═'*60}")
    print(f"  TRTR 准确率: {results['TRTR_acc']:.4f} ({results['TRTR_acc']*100:.2f}%)")
    print(f"  TSTR 准确率: {results['TSTR_acc']:.4f} ({results['TSTR_acc']*100:.2f}%)")
    if "MixAug_acc" in results:
        print(
            f"  TRTR-Augment 准确率: {results['MixAug_acc']:.4f} "
            f"({results['MixAug_acc']*100:.2f}%)  [验证集仅真实物理流形]"
        )
    print(f"  FID (特征空间): {fid_score:.4f}  (越小越接近真实分布)")
    print(f"  结果保存目录: {os.path.abspath(SAVE_DIR)}")
    print(f"\n  生成的文件:")
    for f in sorted(os.listdir(SAVE_DIR)):
        fpath = os.path.join(SAVE_DIR, f)
        if os.path.isfile(fpath):
            size_kb = os.path.getsize(fpath) / 1024
            print(f"    {f}  ({size_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    main()



