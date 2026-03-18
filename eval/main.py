"""
main.py — 数据质量评估全流程入口

支持三种数据模式：
  1. 真实+生成：REAL_DATA_PATH 与 GEN_DATA_FOLDER 均有 .npy 时，完整 TRTR/TSTR
  2. 仅生成：仅有 generated_samples_infer 时，划分为参考/生成做时频与 t-SNE
  3. Dummy：冒烟测试，使用合成数据

三大评估模块：
  1. 时频域物理特征对比分析（时域波形、FFT 频谱）
  2. 下游故障诊断分类 TRTR / TSTR 对比
  3. t-SNE 特征空间可视化（降维图）

运行方式：
  cd diffusionTry
  python -m eval.main

依赖：torch, numpy, matplotlib, scikit-learn, scipy
"""

from __future__ import annotations

import os
import time

# 处理 OpenMP 冲突（Windows 下 torch + numpy 可能会出现）
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib
# 若无 GUI，使用非交互后端
matplotlib.use("Agg")

from eval.dataset import make_dummy_datasets, load_real_and_gen_for_eval, load_gen_only_for_eval
from eval.eval_time_freq import run_time_freq_analysis
from eval.train_diagnosis import run_trtr_tstr
from eval.visualize_tsne import run_tsne_visualization


def main() -> None:
    """全流程冒烟测试入口。"""
    # =====================================================================
    # 基本配置
    # =====================================================================
    SEQ_LENGTH = 1024         # 信号长度（需与 infer/train 一致）
    FS = 25600.0              # 采样率 (Hz)
    TRAIN_RATIO = 0.7         # 真实数据 train/test 划分比例
    NUM_EPOCHS = 30           # 分类器训练轮数
    BATCH_SIZE = 32           # 批大小
    LR = 1e-3                 # 学习率
    LOW_FREQ_LIMIT = 1000.0   # 频谱低频聚焦上界 (Hz)
    SAVE_DIR = "./eval_results"  # 结果保存目录

    # ---------- 数据来源模式 ----------
    USE_REAL_DATA = True      # True: 使用真实+生成数据；False: Dummy 冒烟测试
    USE_GEN_ONLY = False      # 仅生成数据模式（无真实数据时自动启用）
    REAL_DATA_PATH = r"D:\data\轴承数据集\IF0.2"  # 真实数据路径
    GEN_DATA_FOLDER = "./generated_samples_infer"  # infer 生成数据目录
    GEN_LABEL = 0             # 生成数据对应类别标签（单类为 0）
    CLASS_NAMES = None        # 多类别时显式指定，如 ["NC", "IF0.2", "OF0.2"]
    NUM_CLASSES_DUMMY = 4
    SAMPLES_PER_CLASS_DUMMY = 80

    os.makedirs(SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'═'*60}")
    print(f"  轴承故障振动信号 — 数据质量评估全流程")
    print(f"{'═'*60}")
    print(f"  设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    mode_str = "真实+生成数据" if USE_REAL_DATA else "Dummy 冒烟测试"
    print(f"  模式: {mode_str}")
    print(f"  保存目录: {SAVE_DIR}")
    print()

    # =====================================================================
    # 1. 加载数据集
    # =====================================================================
    print("[1] 加载数据集...")
    t0 = time.time()

    if USE_REAL_DATA:
        if not os.path.isdir(GEN_DATA_FOLDER):
            print(f"  生成数据目录不存在: {GEN_DATA_FOLDER}，退化为 Dummy 模式")
            USE_REAL_DATA = False
        elif not os.path.isdir(REAL_DATA_PATH):
            print(f"  真实数据目录不存在: {REAL_DATA_PATH}")
            if os.path.isdir(GEN_DATA_FOLDER):
                print(f"  将使用生成数据划分进行时频对比与 t-SNE（仅生成模式）")
                USE_GEN_ONLY = True
            else:
                USE_REAL_DATA = False
        else:
            import glob
            npy_count = len(glob.glob(os.path.join(REAL_DATA_PATH, "*.npy")))
            subdirs = [d for d in os.listdir(REAL_DATA_PATH)
                       if os.path.isdir(os.path.join(REAL_DATA_PATH, d))]
            subdir_npy = sum(1 for d in subdirs
                             if glob.glob(os.path.join(REAL_DATA_PATH, d, "*.npy"))) if subdirs else 0
            if npy_count == 0 and subdir_npy == 0:
                print(f"  真实数据目录中无 .npy 文件（可能为 .mat 格式），使用仅生成模式")
                USE_GEN_ONLY = True
            else:
                USE_GEN_ONLY = False

    if USE_REAL_DATA and USE_GEN_ONLY:
        real_train, gen_train, real_test, NUM_CLASSES, class_names = load_gen_only_for_eval(
            gen_data_folder=GEN_DATA_FOLDER,
            train_ratio=TRAIN_RATIO,
            seed=42,
        )
    elif USE_REAL_DATA:
        real_train, gen_train, real_test, NUM_CLASSES, class_names = load_real_and_gen_for_eval(
            real_data_path=REAL_DATA_PATH,
            gen_data_folder=GEN_DATA_FOLDER,
            gen_label=GEN_LABEL,
            train_ratio=TRAIN_RATIO,
            seed=42,
            class_names=CLASS_NAMES,
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

    print(f"  真实训练集: {len(real_train)} 条")
    print(f"  生成训练集: {len(gen_train)} 条")
    print(f"  真实测试集: {len(real_test)} 条")
    print(f"  类别数: {NUM_CLASSES}")
    print(f"  耗时: {time.time() - t0:.2f}s\n")

    # =====================================================================
    # 2. 模块一：时频域物理特征对比分析
    # =====================================================================
    print("[2] 运行模块一：时频域物理特征对比分析...")
    t0 = time.time()

    # 从数据集中取出 numpy 数组
    real_signals_np = real_train.signals.numpy()  # (N, 1, L)
    gen_signals_np = gen_train.signals.numpy()    # (M, 1, L)

    run_time_freq_analysis(
        real_signals=real_signals_np,
        gen_signals=gen_signals_np,
        fs=FS,
        low_freq_limit=LOW_FREQ_LIMIT,
        num_display_waveforms=4,
        save_dir=SAVE_DIR,
    )
    print(f"  模块一耗时: {time.time() - t0:.2f}s\n")

    # =====================================================================
    # 3. 模块二：TRTR / TSTR 下游分类对比
    # =====================================================================
    print("[3] 运行模块二：TRTR / TSTR 下游分类对比...")
    t0 = time.time()

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

    # 使用原始信号统计特征进行 t-SNE（无需预训练模型）
    run_tsne_visualization(
        model=None,
        real_dataset=real_train,
        gen_dataset=gen_train,
        device=device,
        class_names=class_names,
        save_path=tsne_save_path,
        use_raw_features=True,
    )
    print(f"  模块三耗时: {time.time() - t0:.2f}s\n")

    # =====================================================================
    # 汇总
    # =====================================================================
    print(f"{'═'*60}")
    print(f"  全流程执行完毕！")
    print(f"{'═'*60}")
    print(f"  TRTR 准确率: {results['TRTR_acc']:.4f} ({results['TRTR_acc']*100:.2f}%)")
    print(f"  TSTR 准确率: {results['TSTR_acc']:.4f} ({results['TSTR_acc']*100:.2f}%)")
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



