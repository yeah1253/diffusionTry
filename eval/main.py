"""
main.py — 全流程冒烟测试入口

使用 dummy 合成数据（无需真实 .npy 文件）跑通三大评估模块的完整流程：
  1. 时频域物理特征对比分析
  2. 下游故障诊断分类 TRTR / TSTR 对比
  3. t-SNE 特征空间可视化

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

from eval.dataset import make_dummy_datasets
from eval.eval_time_freq import run_time_freq_analysis
from eval.train_diagnosis import run_trtr_tstr
from eval.visualize_tsne import run_tsne_visualization


def main() -> None:
    """全流程冒烟测试入口。"""
    # =====================================================================
    # 基本配置
    # =====================================================================
    NUM_CLASSES = 4           # 故障类别数
    SAMPLES_PER_CLASS = 80    # 每类样本数
    SEQ_LENGTH = 1024         # 信号长度
    FS = 25600.0              # 采样率 (Hz)
    TRAIN_RATIO = 0.7         # 训练集占比
    NUM_EPOCHS = 30           # 分类器训练轮数
    BATCH_SIZE = 32           # 批大小
    LR = 1e-3                 # 学习率
    LOW_FREQ_LIMIT = 1000.0   # 频谱低频聚焦上界 (Hz)
    SAVE_DIR = "./eval_results"  # 结果保存目录

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'═'*60}")
    print(f"  轴承故障振动信号 — 数据质量评估全流程")
    print(f"{'═'*60}")
    print(f"  设备: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  类别数: {NUM_CLASSES}")
    print(f"  每类样本数: {SAMPLES_PER_CLASS}")
    print(f"  信号长度: {SEQ_LENGTH}")
    print(f"  采样率: {FS} Hz")
    print(f"  保存目录: {SAVE_DIR}")
    print()

    # =====================================================================
    # 1. 生成 Dummy 数据
    # =====================================================================
    print("▶ 生成 Dummy 数据集...")
    t0 = time.time()

    real_train, gen_train, real_test = make_dummy_datasets(
        num_classes=NUM_CLASSES,
        samples_per_class=SAMPLES_PER_CLASS,
        seq_length=SEQ_LENGTH,
        train_ratio=TRAIN_RATIO,
        seed=42,
    )

    print(f"  真实训练集: {len(real_train)} 条")
    print(f"  生成训练集: {len(gen_train)} 条")
    print(f"  真实测试集: {len(real_test)} 条")
    print(f"  耗时: {time.time() - t0:.2f}s\n")

    # =====================================================================
    # 2. 模块一：时频域物理特征对比分析
    # =====================================================================
    print("▶ 运行模块一：时频域物理特征对比分析...")
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
    print("▶ 运行模块二：TRTR / TSTR 下游分类对比...")
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
    # 4. 模块三：t-SNE 可视化
    # =====================================================================
    print("▶ 运行模块三：t-SNE 特征空间可视化...")
    t0 = time.time()

    class_names = [f"故障类型 {i}" for i in range(NUM_CLASSES)]
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
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {f}  ({size_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    main()



