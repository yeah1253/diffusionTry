"""
visualize_tsne.py — t-SNE 可视化模块

功能：
  提取 1D-ResNet 分类器 GAP 后（默认 256 维）的嵌入，
  使用 t-SNE 降维到 2D 进行可视化，直观展示真实数据与生成数据
  在特征空间中的分布差异。

依赖：torch, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from scipy.linalg import sqrtm

from .dataset import BearingSignalDataset, build_dataloader
from .models import BearingCNN1D


# ---------------------------------------------------------------------------
# Matplotlib 字体设置，避免中文乱码
# ---------------------------------------------------------------------------

def _ensure_chinese_font(font_families: Optional[List[str]] = None) -> None:
    """确保 matplotlib 使用支持中文的字体，避免图例/标题乱码。"""
    fallback = [
        "Microsoft YaHei",  # Windows 常见
        "SimHei",
        "SimSun",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    families = font_families or fallback
    existing = mpl.rcParams.get("font.sans-serif", [])
    mpl.rcParams["font.sans-serif"] = list(dict.fromkeys(families + list(existing)))
    mpl.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# FID 距离替换：Fréchet Inception Distance（特征空间高斯假设下的 2-Wasserstein 相关度量）
# ---------------------------------------------------------------------------

def compute_frechet_distance(
    real_feats: np.ndarray,
    gen_feats: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """
    Fréchet 距离::

        FID = ||mu_1 - mu_2||^2 + Tr(Sigma_1 + Sigma_2 - 2 sqrtm(Sigma_1 @ Sigma_2))

    sqrtm 由 scipy.linalg.sqrtm 给出；若结果为复矩阵则取实部。Sigma_k 加 eps*I 缓解奇异/秩亏导致的 NaN。
    """
    real_feats = np.asarray(real_feats, dtype=np.float64)
    gen_feats = np.asarray(gen_feats, dtype=np.float64)
    mu_1 = np.mean(real_feats, axis=0)
    mu_2 = np.mean(gen_feats, axis=0)
    diff = mu_1 - mu_2

    if real_feats.shape[0] < 2:
        sigma_1 = np.zeros((real_feats.shape[1], real_feats.shape[1]), dtype=np.float64)
    else:
        sigma_1 = np.cov(real_feats, rowvar=False)
    if gen_feats.shape[0] < 2:
        sigma_2 = np.zeros((gen_feats.shape[1], gen_feats.shape[1]), dtype=np.float64)
    else:
        sigma_2 = np.cov(gen_feats, rowvar=False)

    if sigma_1.ndim < 2:
        sigma_1 = np.atleast_2d(sigma_1)
        sigma_2 = np.atleast_2d(sigma_2)
    d = sigma_1.shape[0]
    sigma_1 = sigma_1 + eps * np.eye(d, dtype=np.float64)
    sigma_2 = sigma_2 + eps * np.eye(d, dtype=np.float64)

    covmean = sqrtm(sigma_1 @ sigma_2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = float(np.real(diff @ diff + np.trace(sigma_1 + sigma_2 - 2.0 * covmean)))
    return fid


# ---------------------------------------------------------------------------
# 特征提取
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    model: BearingCNN1D,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 1D-ResNet 中提取 GAP 后的 256 维嵌入（与 BearingResNet1D.forward_features 一致）。

    参数
    ----
    model : BearingCNN1D
        已训练的分类器。
    dataloader : DataLoader
        数据加载器。
    device : torch.device
        计算设备。

    返回
    ----
    features : np.ndarray, shape (N, 256)
        256 维特征向量。
    labels : np.ndarray, shape (N,)
        对应标签。
    """
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for signals, labels in dataloader:
        signals = signals.to(device)
        feat = model.features(signals)     # (B, 256)
        all_features.append(feat.cpu().numpy())
        all_labels.append(labels.numpy())

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return features, labels


@torch.no_grad()
def extract_raw_features(
    signals: np.ndarray,
) -> np.ndarray:
    """
    直接对原始信号做简单特征提取（无需模型），用于模型未训练时的 t-SNE。

    提取时域统计量：均值、标准差、RMS、峰值、峰峰值、偏度、峭度 + FFT 前 16 个分量。

    参数
    ----
    signals : np.ndarray, shape (N, 1, L) 或 (N, L)

    返回
    ----
    features : np.ndarray, shape (N, D)
    """
    if signals.ndim == 3:
        signals = signals.squeeze(1)

    feats_list = []
    for sig in signals:
        mean = np.mean(sig)
        std = np.std(sig)
        rms = np.sqrt(np.mean(sig ** 2))
        peak = np.max(np.abs(sig))
        pp = np.max(sig) - np.min(sig)
        skew = float(np.mean(((sig - mean) / (std + 1e-8)) ** 3))
        kurt = float(np.mean(((sig - mean) / (std + 1e-8)) ** 4))

        # FFT 前 16 个频率分量的幅值
        fft_amp = np.abs(np.fft.rfft(sig))[:16]

        feat = np.concatenate([
            [mean, std, rms, peak, pp, skew, kurt],
            fft_amp,
        ])
        feats_list.append(feat)

    return np.array(feats_list, dtype=np.float32)


# ---------------------------------------------------------------------------
# t-SNE 可视化
# ---------------------------------------------------------------------------

def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    source_tags: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "t-SNE 特征可视化",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    perplexity: float = 30.0,
    random_state: int = 42,
) -> Figure:
    """
    使用 t-SNE 对特征进行 2D 降维可视化。

    参数
    ----
    features : np.ndarray, shape (N, D)
        高维特征向量。
    labels : np.ndarray, shape (N,)
        类别标签。
    source_tags : np.ndarray, shape (N,)
        数据来源标签：0 = 真实数据, 1 = 生成数据。
    class_names : list[str] | None
        类别名称列表。
    title : str
        图标题。
    save_path : str | None
        保存路径。
    perplexity : float
        t-SNE 困惑度。
    random_state : int
        随机种子。
    """
    from sklearn.manifold import TSNE

    _ensure_chinese_font()

    # 自适应调整 perplexity
    n_samples = len(features)
    effective_perplexity = min(perplexity, max(5.0, n_samples / 4.0))

    # t-SNE 降维（兼容 scikit-learn 新旧版本）
    tsne_kwargs = dict(
        n_components=2,
        perplexity=effective_perplexity,
        random_state=random_state,
        learning_rate="auto",
        init="pca",
    )
    # scikit-learn >= 1.6 将 n_iter 重命名为 max_iter
    import inspect
    tsne_params = inspect.signature(TSNE.__init__).parameters
    if "max_iter" in tsne_params:
        tsne_kwargs["max_iter"] = 1000
    else:
        tsne_kwargs["n_iter"] = 1000
    tsne = TSNE(**tsne_kwargs)
    embedded = tsne.fit_transform(features)  # (N, 2)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # --- 子图 1: 按类别着色 ---
    ax1 = axes[0]
    num_classes = int(labels.max()) + 1
    if class_names is None:
        class_names = [f"类别 {i}" for i in range(num_classes)]

    # 兼容 matplotlib 3.7+：get_cmap(name, lut) 已弃用
    try:
        cmap = plt.colormaps["tab10"].resampled(num_classes)
    except AttributeError:
        cmap = plt.cm.get_cmap("tab10", num_classes)

    for c in range(num_classes):
        mask_real = (labels == c) & (source_tags == 0)
        mask_gen = (labels == c) & (source_tags == 1)

        # resampled colormap 的输入域为 [0,1]
        color = cmap(c / max(num_classes - 1, 1))
        ax1.scatter(
            embedded[mask_real, 0], embedded[mask_real, 1],
            c=[color], marker="o", s=20, alpha=0.7,
            label=f"{class_names[c]} (真实)",
        )
        ax1.scatter(
            embedded[mask_gen, 0], embedded[mask_gen, 1],
            c=[color], marker="x", s=20, alpha=0.5,
            label=f"{class_names[c]} (生成)",
        )

    ax1.set_title(f"{title} — 按类别", fontsize=12)
    ax1.legend(fontsize=7, loc="best", ncol=2)
    ax1.grid(True, alpha=0.2)

    # --- 子图 2: 按来源着色 ---
    ax2 = axes[1]
    mask_real_all = source_tags == 0
    mask_gen_all = source_tags == 1

    ax2.scatter(
        embedded[mask_real_all, 0], embedded[mask_real_all, 1],
        c="steelblue", marker="o", s=15, alpha=0.6, label="真实数据",
    )
    ax2.scatter(
        embedded[mask_gen_all, 0], embedded[mask_gen_all, 1],
        c="tomato", marker="x", s=15, alpha=0.5, label="生成数据",
    )

    ax2.set_title(f"{title} — 按来源", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[t-SNE] 已保存到 {save_path}")

    return fig


# ---------------------------------------------------------------------------
# 统合接口
# ---------------------------------------------------------------------------

def run_tsne_visualization(
    model: Optional[BearingCNN1D],
    real_dataset: BearingSignalDataset,
    gen_dataset: BearingSignalDataset,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    use_raw_features: bool = False,
) -> float:
    """
    一键运行 t-SNE 可视化，并计算真实/生成特征分布的 Fréchet 距离（FID）。

    返回
    ----
    float
        FID 分数（越小表示两分布越接近，同特征维度假设下可比较）。
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  模块三：t-SNE 特征空间可视化")
    print("=" * 60)

    batch_size = 64

    if use_raw_features or model is None:
        # 从原始信号提取统计特征
        print("\n  使用原始信号统计特征进行 t-SNE...")
        real_feats = extract_raw_features(real_dataset.signals.numpy())
        gen_feats = extract_raw_features(gen_dataset.signals.numpy())
    else:
        # 从 CNN 模型提取深度特征
        print("\n  使用 CNN 深度特征进行 t-SNE...")
        real_loader = build_dataloader(real_dataset, batch_size=batch_size, shuffle=False)
        gen_loader = build_dataloader(gen_dataset, batch_size=batch_size, shuffle=False)
        real_feats, _ = extract_features(model, real_loader, device)
        gen_feats, _ = extract_features(model, gen_loader, device)

    # 拼接特征与标签
    features = np.concatenate([real_feats, gen_feats], axis=0)
    labels = np.concatenate([
        real_dataset.labels.numpy(),
        gen_dataset.labels.numpy(),
    ], axis=0)
    source_tags = np.concatenate([
        np.zeros(len(real_dataset), dtype=np.int64),   # 0 = 真实
        np.ones(len(gen_dataset), dtype=np.int64),     # 1 = 生成
    ], axis=0)

    print(f"  真实数据样本数: {len(real_dataset)}")
    print(f"  生成数据样本数: {len(gen_dataset)}")
    print(f"  特征维度: {features.shape[1]}")

    # FID 距离替换：量化特征空间分布差异（避免 KL 行列式奇异导致 NaN）
    fid_score = compute_frechet_distance(real_feats, gen_feats)
    print(f"\n  [FID] 真实 vs 生成 特征分布 Fréchet 距离（越小越接近）:")
    print(f"    FID = {fid_score:.4f}")

    print(f"\n  正在运行 t-SNE...")
    plot_tsne(
        features, labels, source_tags,
        class_names=class_names,
        save_path=save_path,
    )

    print("  t-SNE 可视化完成！\n")
    return fid_score



