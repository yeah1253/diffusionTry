"""
eval_time_freq.py — 时频域物理特征对比分析模块

功能：
  1. 时域波形对比：在同一张图中绘制真实信号与生成信号，x 轴映射为真实时间 (fs=25600 Hz)。
  2. FFT 频谱对比：计算频谱幅值并重叠对比（不同颜色 + alpha 透明度），
     特别放大并聚焦低频频段的重合性。

依赖：torch, numpy, matplotlib, scipy
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# ---------------------------------------------------------------------------
# 时域波形对比
# ---------------------------------------------------------------------------

def plot_time_domain(
    real_signal: np.ndarray,
    gen_signal: np.ndarray,
    fs: float = 25600.0,
    title: str = "时域波形对比",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 4),
) -> Figure:
    """
    在同一张图中绘制真实信号与生成信号的时域波形。

    参数
    ----
    real_signal : np.ndarray, shape (L,)
        真实振动信号（单条）。
    gen_signal : np.ndarray, shape (L,)
        生成振动信号（单条）。
    fs : float
        采样率 (Hz)，默认 25600。
    title : str
        图标题。
    save_path : str | None
        若不为 None，将图保存到该路径。
    figsize : tuple
        图像尺寸。

    返回
    ----
    fig : matplotlib.figure.Figure
    """
    L = len(real_signal)
    t = np.arange(L) / fs  # 时间轴（秒）

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(t, real_signal, color="steelblue", alpha=0.8, linewidth=0.7, label="真实信号")
    ax.plot(t, gen_signal, color="tomato", alpha=0.6, linewidth=0.7, label="生成信号")
    ax.set_xlabel("时间 (s)", fontsize=11)
    ax.set_ylabel("幅值", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[时域波形] 已保存到 {save_path}")

    return fig


def plot_time_domain_batch(
    real_signals: np.ndarray,
    gen_signals: np.ndarray,
    fs: float = 25600.0,
    num_display: int = 4,
    title: str = "时域波形对比（多条）",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> Figure:
    """
    绘制多条真实/生成信号的时域波形对比（子图排列）。

    参数
    ----
    real_signals : np.ndarray, shape (N, L) 或 (N, 1, L)
    gen_signals  : np.ndarray, shape (M, L) 或 (M, 1, L)
    num_display  : int
        展示前几条。
    """
    # 统一维度
    if real_signals.ndim == 3:
        real_signals = real_signals.squeeze(1)
    if gen_signals.ndim == 3:
        gen_signals = gen_signals.squeeze(1)

    n = min(num_display, len(real_signals), len(gen_signals))
    L = real_signals.shape[-1]
    t = np.arange(L) / fs

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(t, real_signals[i], color="steelblue", alpha=0.8, lw=0.7, label="真实信号")
        ax.plot(t, gen_signals[i], color="tomato", alpha=0.6, lw=0.7, label="生成信号")
        ax.set_ylabel(f"样本 {i}", fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
            ax.set_title(title, fontsize=13)

    axes[-1].set_xlabel("时间 (s)", fontsize=11)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[时域波形-批量] 已保存到 {save_path}")

    return fig


# ---------------------------------------------------------------------------
# FFT 频谱对比
# ---------------------------------------------------------------------------

def compute_fft_spectrum(
    signal: np.ndarray,
    fs: float = 25600.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算单条信号的单边 FFT 幅值谱。

    返回
    ----
    freqs : np.ndarray, shape (N_fft,)
        频率轴 (Hz)。
    amplitude : np.ndarray, shape (N_fft,)
        归一化幅值谱。
    """
    L = len(signal)
    fft_vals = np.fft.rfft(signal)
    amplitude = np.abs(fft_vals) * 2.0 / L  # 归一化幅值
    amplitude[0] /= 2.0                     # 直流分量不需要 ×2
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    return freqs, amplitude


def compute_avg_fft_spectrum(
    signals: np.ndarray,
    fs: float = 25600.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算多条信号的平均 FFT 幅值谱。

    参数
    ----
    signals : np.ndarray, shape (N, L) 或 (N, 1, L)

    返回
    ----
    freqs : np.ndarray
    avg_amplitude : np.ndarray
    """
    if signals.ndim == 3:
        signals = signals.squeeze(1)

    all_amps = []
    for sig in signals:
        _, amp = compute_fft_spectrum(sig, fs)
        all_amps.append(amp)

    avg_amp = np.mean(all_amps, axis=0)
    freqs = np.fft.rfftfreq(signals.shape[-1], d=1.0 / fs)
    return freqs, avg_amp


def plot_fft_comparison(
    real_signals: np.ndarray,
    gen_signals: np.ndarray,
    fs: float = 25600.0,
    low_freq_limit: float = 1000.0,
    title: str = "FFT 频谱对比",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> Figure:
    """
    绘制真实数据与生成数据的平均 FFT 频谱对比图。

    包含两个子图：
      1. 全频段对比
      2. 低频频段放大对比（0 ~ low_freq_limit Hz）

    参数
    ----
    real_signals : np.ndarray, shape (N, L) 或 (N, 1, L)
    gen_signals  : np.ndarray, shape (M, L) 或 (M, 1, L)
    fs : float
        采样率 (Hz)。
    low_freq_limit : float
        低频放大区上界 (Hz)。
    title : str
        图标题。
    save_path : str | None
        保存路径。
    """
    freqs_r, amp_r = compute_avg_fft_spectrum(real_signals, fs)
    freqs_g, amp_g = compute_avg_fft_spectrum(gen_signals, fs)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

    # --- 子图 1: 全频段 ---
    ax1.plot(freqs_r, amp_r, color="steelblue", alpha=0.85, lw=0.9, label="真实信号 (平均)")
    ax1.plot(freqs_g, amp_g, color="tomato", alpha=0.65, lw=0.9, label="生成信号 (平均)")
    ax1.set_xlabel("频率 (Hz)", fontsize=11)
    ax1.set_ylabel("幅值", fontsize=11)
    ax1.set_title(f"{title} — 全频段", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # --- 子图 2: 低频聚焦 ---
    low_mask_r = freqs_r <= low_freq_limit
    low_mask_g = freqs_g <= low_freq_limit

    ax2.plot(freqs_r[low_mask_r], amp_r[low_mask_r],
             color="steelblue", alpha=0.85, lw=1.0, label="真实信号 (平均)")
    ax2.plot(freqs_g[low_mask_g], amp_g[low_mask_g],
             color="tomato", alpha=0.65, lw=1.0, label="生成信号 (平均)")
    ax2.set_xlabel("频率 (Hz)", fontsize=11)
    ax2.set_ylabel("幅值", fontsize=11)
    ax2.set_title(f"{title} — 低频聚焦 (0 ~ {low_freq_limit:.0f} Hz)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # 在低频区标注主要峰值频率
    _annotate_peaks(ax2, freqs_r[low_mask_r], amp_r[low_mask_r], color="steelblue", n_peaks=3)
    _annotate_peaks(ax2, freqs_g[low_mask_g], amp_g[low_mask_g], color="tomato", n_peaks=3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[FFT 频谱] 已保存到 {save_path}")

    return fig


def _annotate_peaks(
    ax: plt.Axes,
    freqs: np.ndarray,
    amps: np.ndarray,
    color: str = "black",
    n_peaks: int = 3,
) -> None:
    """在频谱图上标注前 n 个幅值最大的峰值。"""
    if len(amps) < n_peaks:
        return

    # 简易峰值检测：选取幅值最大的 n 个位置
    peak_indices = np.argsort(amps)[-n_peaks:]
    for idx in peak_indices:
        freq_val = freqs[idx]
        amp_val = amps[idx]
        ax.annotate(
            f"{freq_val:.0f} Hz",
            xy=(freq_val, amp_val),
            xytext=(freq_val + 20, amp_val * 1.05),
            fontsize=8,
            color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
        )


# ---------------------------------------------------------------------------
# 统合接口：一键运行时频域分析
# ---------------------------------------------------------------------------

def run_time_freq_analysis(
    real_signals: np.ndarray,
    gen_signals: np.ndarray,
    fs: float = 25600.0,
    low_freq_limit: float = 1000.0,
    num_display_waveforms: int = 4,
    save_dir: Optional[str] = None,
) -> None:
    """
    一键运行时频域对比分析，生成并保存所有图表。

    参数
    ----
    real_signals : np.ndarray, shape (N, 1, L) 或 (N, L)
    gen_signals  : np.ndarray, shape (M, 1, L) 或 (M, L)
    fs : float
        采样率 (Hz)。
    low_freq_limit : float
        低频聚焦上界 (Hz)。
    num_display_waveforms : int
        时域对比展示的样本数量。
    save_dir : str | None
        保存目录，None 则仅显示不保存。
    """
    import os
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # 统一维度
    if real_signals.ndim == 3:
        real_flat = real_signals.squeeze(1)
    else:
        real_flat = real_signals
    if gen_signals.ndim == 3:
        gen_flat = gen_signals.squeeze(1)
    else:
        gen_flat = gen_signals

    print("=" * 60)
    print("  模块一：时频域物理特征对比分析")
    print("=" * 60)

    # 1) 时域波形对比
    print("\n[1/2] 绘制时域波形对比...")
    waveform_path = os.path.join(save_dir, "time_domain_comparison.png") if save_dir else None
    plot_time_domain_batch(
        real_flat, gen_flat,
        fs=fs,
        num_display=num_display_waveforms,
        save_path=waveform_path,
    )

    # 2) FFT 频谱对比
    print("[2/2] 绘制 FFT 频谱对比...")
    fft_path = os.path.join(save_dir, "fft_spectrum_comparison.png") if save_dir else None
    plot_fft_comparison(
        real_flat, gen_flat,
        fs=fs,
        low_freq_limit=low_freq_limit,
        save_path=fft_path,
    )

    # 打印统计信息
    print(f"\n--- 数据统计 ---")
    print(f"  真实信号: {real_flat.shape[0]} 条, 长度 {real_flat.shape[1]}, "
          f"幅值范围 [{real_flat.min():.4f}, {real_flat.max():.4f}]")
    print(f"  生成信号: {gen_flat.shape[0]} 条, 长度 {gen_flat.shape[1]}, "
          f"幅值范围 [{gen_flat.min():.4f}, {gen_flat.max():.4f}]")

    # 简单频域一致性指标：平均频谱的余弦相似度
    _, amp_r = compute_avg_fft_spectrum(real_flat, fs)
    _, amp_g = compute_avg_fft_spectrum(gen_flat, fs)
    cosine_sim = np.dot(amp_r, amp_g) / (np.linalg.norm(amp_r) * np.linalg.norm(amp_g) + 1e-8)
    print(f"  平均频谱余弦相似度: {cosine_sim:.6f}")
    print()


