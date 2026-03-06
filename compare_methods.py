"""
compare_methods.py
==================
对比三种工况插值方法在真实 generated_grid 数据上的效果：
  A. 朴素双线性插值（时域直接加权）
  B. 频域幅值-相位解耦插值（FFT + 幅值插值 + IFFT）
  C. 阶次跟踪 + 双维度 IDW 插值

目标工况：load=30, rpm=2350（排除该点后用邻域插值，再与真实数据对比）

用法：
  python compare_methods.py --target-load 30 --target-rpm 2350
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from scipy.signal import correlate, hilbert, resample


# ─────────────────────────── data loading ───────────────────────────

def load_filtered_mean(cond_dir: Path) -> np.ndarray:
    files = sorted(cond_dir.glob("filtered_*.npy"))
    if files:
        return np.stack([np.load(f).astype(np.float64).reshape(-1) for f in files]).mean(0)
    arr = np.load(cond_dir / "all_generated.npy").astype(np.float64)
    if arr.ndim == 2:
        return arr.mean(0)
    return arr


def load_all_samples(cond_dir: Path) -> np.ndarray:
    files = sorted(cond_dir.glob("filtered_*.npy"))
    if files:
        return np.stack([np.load(f).astype(np.float64).reshape(-1) for f in files])
    arr = np.load(cond_dir / "all_generated.npy").astype(np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def discover_grid(grid_root: Path) -> List[Tuple[float, float]]:
    pts = []
    for ld in grid_root.iterdir():
        if not ld.is_dir() or not ld.name.startswith("load_"):
            continue
        try:
            lv = float(ld.name[5:])
        except ValueError:
            continue
        for rp in ld.iterdir():
            if not rp.is_dir() or not rp.name.startswith("rpm_"):
                continue
            try:
                rv = float(rp.name[4:])
            except ValueError:
                continue
            pts.append((lv, rv))
    return sorted(pts)


# ─────────────────────────── neighbor helpers ───────────────────────────

def _scale(arr: np.ndarray) -> float:
    u = np.unique(arr)
    if u.size < 2:
        return 1.0
    d = np.diff(np.sort(u))
    return float(np.median(d[d > 0])) or 1.0


def idw_weights(pts: np.ndarray, target: np.ndarray, k: int, power: float):
    ls = _scale(pts[:, 0])
    rs = _scale(pts[:, 1])
    d = np.sqrt(((pts[:, 0] - target[0]) / ls) ** 2 + ((pts[:, 1] - target[1]) / rs) ** 2)
    order = np.argsort(d)
    sel = order[:max(1, min(k, len(order)))]
    ds = d[sel]
    if ds[0] < 1e-12:
        w = np.zeros_like(ds)
        w[0] = 1.0
        return sel, w
    w = 1.0 / np.power(ds + 1e-12, power)
    return sel, w / w.sum()


def bilinear_corners(pts: np.ndarray, target: np.ndarray):
    loads, rpms = pts[:, 0], pts[:, 1]
    tl, tr = target
    lo_l_arr = loads[loads < tl]
    hi_l_arr = loads[loads > tl]
    lo_r_arr = rpms[rpms < tr]
    hi_r_arr = rpms[rpms > tr]
    if not (lo_l_arr.size and hi_l_arr.size and lo_r_arr.size and hi_r_arr.size):
        return None
    lo_l, hi_l = lo_l_arr.max(), hi_l_arr.min()
    lo_r, hi_r = lo_r_arr.max(), hi_r_arr.min()
    tx = (tl - lo_l) / (hi_l - lo_l)
    ty = (tr - lo_r) / (hi_r - lo_r)
    corners = [(lo_l, lo_r), (hi_l, lo_r), (lo_l, hi_r), (hi_l, hi_r)]
    weights = [(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty]
    return corners, weights


# ─────────────────────────── Method A: bilinear time-domain ───────────────────────────

def method_a_bilinear(train: Dict[Tuple[float, float], np.ndarray],
                      target_load: float, target_rpm: float) -> np.ndarray:
    pts = np.array(list(train.keys()), dtype=np.float64)
    result = bilinear_corners(pts, np.array([target_load, target_rpm]))
    if result is None:
        sel, w = idw_weights(pts, np.array([target_load, target_rpm]), k=4, power=2.0)
        sigs = np.stack([train[tuple(pts[i].tolist())] for i in sel])
        return np.tensordot(w, sigs, axes=(0, 0))
    corners, weights = result
    sigs = np.stack([train[c] for c in corners])
    return np.tensordot(np.array(weights), sigs, axes=(0, 0))


# ─────────────────────────── Method B: spectral amplitude-phase ───────────────────────────

def method_b_spectral(train: Dict[Tuple[float, float], np.ndarray],
                      target_load: float, target_rpm: float) -> np.ndarray:
    """
    FFT → IDW 幅值谱插值 + 邻域相位循环加权平均 → IFFT
    相位使用邻域信号 FFT 的循环均值（circular mean），
    避免 0°/360° 跳变，不依赖任何目标信号信息。
    """
    pts = np.array(list(train.keys()), dtype=np.float64)
    n_sig = len(next(iter(train.values())))
    sel, w = idw_weights(pts, np.array([target_load, target_rpm]), k=6, power=2.0)

    neighbor_ffts = np.stack([rfft(train[tuple(pts[i].tolist())]) for i in sel])  # (k, F)

    # 幅值谱 IDW 加权插值
    mags = np.abs(neighbor_ffts)                                    # (k, F)
    mag_interp = np.tensordot(w, mags, axes=(0, 0))                # (F,)

    # 相位循环加权均值：对复数单位向量加权求和后取角度
    # exp(i*phi) 的加权和的 angle 即为循环均值，天然处理相位环绕
    unit_phasors = neighbor_ffts / (np.abs(neighbor_ffts) + 1e-12)  # (k, F)
    phase_vec = np.tensordot(w, unit_phasors, axes=(0, 0))           # (F,)
    phase_interp = np.angle(phase_vec)                               # (F,)

    return irfft(mag_interp * np.exp(1j * phase_interp), n=n_sig).real


# ─────────────────────────── Method C: order tracking + IDW ───────────────────────────

def align_and_average(samples: np.ndarray) -> np.ndarray:
    ref = samples[0]
    aligned = [np.roll(s, int(np.argmax(correlate(ref, s, mode="full"))) - (len(s) - 1))
               for s in samples]
    return np.mean(aligned, axis=0)




def method_c_order_tracking(
        train: Dict[Tuple[float, float], np.ndarray],
        target_load: float, target_rpm: float,
        fs: float, n_angle: int, k: int = 6, power: float = 2.0) -> np.ndarray:

    pts = np.array(list(train.keys()), dtype=np.float64)
    sel, w = idw_weights(pts, np.array([target_load, target_rpm]), k=k, power=power)

    N = n_angle  # 统一输出长度

    aligned_sources: List[np.ndarray] = []
    rmss: List[float] = []

    # ── 步骤 1：频率规整（Frequency Warping via Resampling）──
    for idx in sel:
        rpm, load = pts[idx]
        avg = train[(rpm, load)]                           # 预处理好的系综平均信号

        # 将源信号重采样到"如果在目标转速下采集"的等效长度
        new_len = int(len(avg) * (rpm / target_rpm))
        warped = resample(avg, new_len)

        # 长度对齐到 N
        if new_len >= N:
            warped = warped[:N]                            # 截断
        else:
            warped = np.pad(warped, (0, N - new_len), mode="constant")  # 零填充

        # RMS 归一化，保存能量
        rms = float(np.sqrt(np.mean(warped ** 2)) + 1e-12)
        rmss.append(rms)
        aligned_sources.append(warped / rms)

    # ── 步骤 2：跨工况包络微对齐（Envelope-based Cross-Correlation）──
    # 用包络互相关而非原始波形，避开高频噪声的干扰
    ref_idx = int(np.argmax(w))
    ref_env = np.abs(hilbert(aligned_sources[ref_idx]))

    final_aligned: List[np.ndarray] = []
    for curr_sig in aligned_sources:
        curr_env = np.abs(hilbert(curr_sig))
        corr = correlate(ref_env, curr_env, mode="full")
        lag = int(np.argmax(corr)) - (len(curr_env) - 1)
        final_aligned.append(np.roll(curr_sig, lag))

    # ── 步骤 3：形态与能量的双维度 IDW 插值 ──
    norm_interp = np.tensordot(w, np.stack(final_aligned), axes=(0, 0))

    # 能量保真补偿：重新强制归一化为单位能量，防止相消衰减导致幅值偏低
    norm_interp_rms = float(np.sqrt(np.mean(norm_interp ** 2)) + 1e-12)
    norm_interp = norm_interp / norm_interp_rms

    # 能量标量 IDW 加权
    rms_interp = float(np.dot(w, np.array(rmss)))

    # ── 步骤 4：直接输出 ──
    return norm_interp * rms_interp



# ─────────────────────────── metrics ───────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    nrmse = rmse / (float(np.ptp(y_true)) + 1e-12)
    tc = y_true - y_true.mean()
    pc = y_pred - y_pred.mean()
    pearson = float(np.dot(tc, pc) / (np.linalg.norm(tc) * np.linalg.norm(pc) + 1e-12))
    ft = np.abs(rfft(y_true))
    fp = np.abs(rfft(y_pred))
    spec_nrmse = float(np.sqrt(np.mean((fp - ft) ** 2)) / (np.linalg.norm(ft) + 1e-12))

    # 平移不变最佳对齐相关系数（Max Aligned Pearson r）
    # 将两信号标准化为零均值单位方差后，取全序列互相关的峰值除以信号长度
    yt_norm = tc / (float(np.std(y_true)) + 1e-12)
    yp_norm = pc / (float(np.std(y_pred)) + 1e-12)
    cross_corr = correlate(yt_norm, yp_norm, mode="full")
    max_aligned_r = float(np.max(cross_corr) / len(y_true))

    return dict(mae=mae, rmse=rmse, nrmse=nrmse, pearson_r=pearson,
                spec_nrmse=spec_nrmse, max_aligned_r=max_aligned_r)


# ─────────────────────────── visualization ───────────────────────────

def plot_all(y_true: np.ndarray, preds: Dict[str, np.ndarray], fs: float, out: Path):
    colors = {"Bilinear (A)": "tab:blue",
              "Spectral (B)": "tab:orange",
              "OrderTracking (C)": "tab:green"}
    n = len(y_true)
    t = np.arange(n) / fs
    freqs = rfftfreq(n, 1 / fs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    axes[0].plot(t, y_true, "k", lw=1.8, label="True", zorder=5)
    axes[1].plot(freqs, np.abs(rfft(y_true)), "k", lw=1.8, label="True", zorder=5)

    for name, sig in preds.items():
        c = colors.get(name)
        axes[0].plot(t, sig, lw=1.0, alpha=0.85, label=name, color=c)
        axes[1].plot(freqs, np.abs(rfft(sig)), lw=1.0, alpha=0.85, label=name, color=c)

    axes[0].set_title("时域对比  (Time-domain comparison)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("频域幅值谱对比  (Frequency-domain magnitude)")
    axes[1].set_xlabel("Hz")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, min(float(freqs[-1]), 5000))

    fig.suptitle("插值方法对比  load=30, rpm=2350", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {out}")


def plot_single(y_true: np.ndarray, y_pred: np.ndarray, method_name: str, fs: float, out: Path):
    """Plot one method against ground truth in time and frequency domains."""
    n = len(y_true)
    t = np.arange(n) / fs
    freqs = rfftfreq(n, 1 / fs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    axes[0].plot(t, y_true, "k", lw=1.8, label="True", zorder=5)
    axes[0].plot(t, y_pred, "tab:red", lw=1.1, alpha=0.9, label=method_name)
    axes[0].set_title("Time-domain comparison")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(freqs, np.abs(rfft(y_true)), "k", lw=1.8, label="True", zorder=5)
    axes[1].plot(freqs, np.abs(rfft(y_pred)), "tab:red", lw=1.1, alpha=0.9, label=method_name)
    axes[1].set_title("Frequency-domain magnitude")
    axes[1].set_xlabel("Hz")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, min(float(freqs[-1]), 5000))

    fig.suptitle(f"{method_name} vs True  (load=30, rpm=2350)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved → {out}")


# ─────────────────────────── main ───────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compare three interpolation methods on real generated_grid data")
    parser.add_argument("--grid-root", default="./generated_grid")
    parser.add_argument("--target-load", type=float, default=30)
    parser.add_argument("--target-rpm", type=float, default=2350)
    parser.add_argument("--fs", type=float, default=25600,
                        help="Assumed sampling rate for order-tracking (Hz)")
    parser.add_argument("--idw-k", type=int, default=6)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--out-dir", default="./interpolation_comparison")
    args = parser.parse_args()

    grid_root = Path(args.grid_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Discovering grid points …")
    all_pts = discover_grid(grid_root)
    target_key = (args.target_load, args.target_rpm)
    if target_key not in all_pts:
        raise SystemExit(f"Target {target_key} not found in grid. "
                         f"Available loads: {sorted(set(p[0] for p in all_pts))[:10]} …")

    print(f"Grid contains {len(all_pts)} conditions.")

    # ── load data ──
    print("Loading representative signals (this may take a moment) …")
    sig_map: Dict[Tuple[float, float], np.ndarray] = {}
    multi_map: Dict[Tuple[float, float], np.ndarray] = {}
    for lv, rv in all_pts:
        cdir = grid_root / f"load_{int(lv)}" / f"rpm_{int(rv)}"
        if not cdir.exists():
            continue
        sig_map[(lv, rv)] = load_filtered_mean(cdir)
        multi_map[(lv, rv)] = load_all_samples(cdir)

    y_true = sig_map[target_key]
    n = len(y_true)

    train_sig = {k: v for k, v in sig_map.items() if k != target_key}
    train_keys_excl = [k for k in all_pts if k != target_key and k in sig_map]
    multi_excl = {k: v for k, v in multi_map.items() if k != target_key}

    # ── Method A ──
    print("Running Method A (bilinear, time-domain) …")
    pred_a = method_a_bilinear(train_sig, args.target_load, args.target_rpm)

    # ── Method B ──
    print("Running Method B (spectral, neighbor phase circular mean) …")
    pred_b = method_b_spectral(train_sig, args.target_load, args.target_rpm)

    # ── Method C ──
    print("Running Method C (order tracking + IDW) …")
    pred_c = method_c_order_tracking(
        train_sig,
        target_load=args.target_load, target_rpm=args.target_rpm,
        fs=args.fs, n_angle=n,
        k=args.idw_k, power=args.idw_power,
    )

    # ── print metrics ──
    results = {
        "Bilinear (A)":       (pred_a, compute_metrics(y_true, pred_a)),
        "Spectral (B)":       (pred_b, compute_metrics(y_true, pred_b)),
        "OrderTracking (C)":  (pred_c, compute_metrics(y_true, pred_c)),
    }

    print("\n" + "─" * 95)
    print(f"{'Method':<22}  {'MAE':>9}  {'RMSE':>9}  {'NRMSE':>8}  {'Pearson_r':>10}  {'Spec_NRMSE':>11}  {'Max_Align_R':>12}")
    print("─" * 95)
    for name, (_, m) in results.items():
        print(f"{name:<22}  {m['mae']:9.4f}  {m['rmse']:9.4f}  {m['nrmse']:8.4f}"
              f"  {m['pearson_r']:10.4f}  {m['spec_nrmse']:11.4f}  {m['max_aligned_r']:12.4f}")
    print("─" * 95)

    # ── save ──
    for name, (pred, _) in results.items():
        tag = name.split()[0].lower()
        np.save(out_dir / f"pred_{tag}.npy", pred)
    np.save(out_dir / "true.npy", y_true)

    report = {
        "target": {"load": args.target_load, "rpm": args.target_rpm},
        "metrics": {n: m for n, (_, m) in results.items()},
    }
    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    plot_all(y_true,
             {n: p for n, (p, _) in results.items()},
             fs=args.fs,
             out=out_dir / "comparison.png")

    single_plot_names = {
        "Bilinear (A)": "comparison_bilinear.png",
        "Spectral (B)": "comparison_spectral.png",
        "OrderTracking (C)": "comparison_ordertracking.png",
    }
    for method_name, (pred, _) in results.items():
        plot_single(
            y_true=y_true,
            y_pred=pred,
            method_name=method_name,
            fs=args.fs,
            out=out_dir / single_plot_names[method_name],
        )

    print(f"\nAll outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()

