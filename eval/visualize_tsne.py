"""
visualize_tsne.py — t-SNE 可视化模块

功能：
  提取 1D-CNN 分类器倒数第二层（128 维特征）的嵌入，
  使用 t-SNE 降维到 2D 进行可视化，直观展示真实数据与生成数据
  在特征空间中的分布差异。

依赖：torch, numpy, matplotlib, scikit-learn
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .dataset import BearingSignalDataset, build_dataloader
from .models import BearingCNN1D


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
    从 1D-CNN 中提取倒数第二层的 128 维特征向量。

    利用 model.features 提取卷积特征，再 flatten 得到固定长度嵌入。

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
    features : np.ndarray, shape (N, 128)
        128 维特征向量。
    labels : np.ndarray, shape (N,)
        对应标签。
    """
    model.eval()
    all_features: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for signals, labels in dataloader:
        signals = signals.to(device)
        # 提取卷积特征（features 的输出为 (B, 128, 1)）
        feat = model.features(signals)     # (B, 128, 1)
        feat = feat.squeeze(-1)            # (B, 128)
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

    cmap = plt.cm.get_cmap("tab10", num_classes)

    for c in range(num_classes):
        mask_real = (labels == c) & (source_tags == 0)
        mask_gen = (labels == c) & (source_tags == 1)

        color = cmap(c)
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
) -> None:
    """
    一键运行 t-SNE 可视化。

    参数
    ----
    model : BearingCNN1D | None
        已训练的分类器。若为 None 或 use_raw_features=True，则使用原始特征。
    real_dataset : BearingSignalDataset
        真实数据。
    gen_dataset : BearingSignalDataset
        生成数据。
    device : torch.device | None
    class_names : list[str] | None
    save_path : str | None
    use_raw_features : bool
        是否直接从原始信号提取特征（不使用 CNN）。
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
    print(f"  正在运行 t-SNE...")

    plot_tsne(
        features, labels, source_tags,
        class_names=class_names,
        save_path=save_path,
    )

    print("  t-SNE 可视化完成！\n")



