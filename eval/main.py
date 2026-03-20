"""
main.py — 数据质量评估全流程入口

数据模式：
  - USE_REAL_DATA=True：真实+生成数据，路径错误直接报错
  - USE_REAL_DATA=False：Dummy 冒烟测试

评估模块：时频对比、TRTR/TSTR、t-SNE
"""

from __future__ import annotations

import os
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as _plt
_plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "SimSun", "DejaVu Sans"]
_plt.rcParams["axes.unicode_minus"] = False
import warnings
warnings.filterwarnings("ignore", message=".*Glyph.*missing.*")

from eval.dataset import make_dummy_datasets, load_real_and_gen_for_eval
from eval.eval_time_freq import run_time_freq_analysis
from eval.train_diagnosis import run_trtr_tstr
from eval.train_diagnosis_simple import run_trtr_tstr_simple
from eval.visualize_tsne import run_tsne_visualization


def main() -> None:
    # 配置
    SEQ_LENGTH = 1024
    FS = 25600.0
    TRAIN_RATIO = 0.7
    NUM_EPOCHS = 30
    BATCH_SIZE = 32
    LR = 1e-3
    LOW_FREQ_LIMIT = 1000.0 #频谱图横轴上限 (Hz)，只展示 0–1000 Hz，便于观察主要频率

    USE_SIMPLE_CLASSIFIER = False
    MAX_SAMPLES_PER_CLASS = 960 #每个故障类别最多保留的样本数
    MAX_SAMPLES_PER_GROUP = 40  #每个工况最多保留的样本数(40*24=960)

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAVE_DIR = os.path.join(_PROJECT_ROOT, "eval_results")  #结果生成路径
    REAL_DATA_PATH = r"D:\data\轴承数据集"
    GEN_DATA_FOLDER = os.path.join(_PROJECT_ROOT, "generated_samples_infer")
    USE_REAL_DATA = True
    NUM_CLASSES_DUMMY = 4
    SAMPLES_PER_CLASS_DUMMY = 80

    os.makedirs(SAVE_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"{'═'*60}\n  轴承故障振动信号 — 数据质量评估\n{'═'*60}")
    print(f"  设备: {device}  模式: {'真实+生成' if USE_REAL_DATA else 'Dummy'}\n")

    # 1. 加载数据集
    print("[1] 加载数据集...")
    t0 = time.time()
    if USE_REAL_DATA:
        real_train, gen_train, real_test, NUM_CLASSES, class_names = load_real_and_gen_for_eval(
            real_data_path=REAL_DATA_PATH,
            gen_data_folder=GEN_DATA_FOLDER,
            train_ratio=TRAIN_RATIO,
            seed=42,
            max_per_class=MAX_SAMPLES_PER_CLASS,
            max_per_group=MAX_SAMPLES_PER_GROUP,
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

    # t-SNE 可视化 + KL 散度（量化真实/生成分布差异）
    kl_result = run_tsne_visualization(
        model=None,
        real_dataset=real_train,
        gen_dataset=gen_train,
        device=device,
        class_names=class_names,
        save_path=tsne_save_path,
        use_raw_features=True,
    )
    # 保存 KL 结果到文件
    kl_path = os.path.join(SAVE_DIR, "kl_divergence.txt")
    with open(kl_path, "w", encoding="utf-8") as f:
        f.write("KL 散度 (特征空间，高斯假设)\n")
        f.write(f"KL(真实||生成): {kl_result['kl_real_gen']:.6f}\n")
        f.write(f"KL(生成||真实): {kl_result['kl_gen_real']:.6f}\n")
        f.write(f"Jensen-Shannon: {kl_result['js']:.6f}\n")
    print(f"  KL 结果已保存到 {kl_path}")
    print(f"  模块三耗时: {time.time() - t0:.2f}s\n")

    # =====================================================================
    # 汇总
    # =====================================================================
    print(f"{'═'*60}")
    print(f"  全流程执行完毕！")
    print(f"{'═'*60}")
    print(f"  TRTR 准确率: {results['TRTR_acc']:.4f} ({results['TRTR_acc']*100:.2f}%)")
    print(f"  TSTR 准确率: {results['TSTR_acc']:.4f} ({results['TSTR_acc']*100:.2f}%)")
    print(f"  KL 散度 (JS): {kl_result['js']:.4f}  (0=分布相同)")
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



