import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate

# ---------------------- synthetic data generation ----------------------

def synth_signal(rpm: float, load: float, fs: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic bearing-like vibration signal with random phase."""
    t = np.arange(n) / fs
    base_freq = rpm / 60.0  # Hz fundamental related to shaft speed
    # two harmonics and a load-dependent amplitude
    amp = 0.5 + 0.02 * load
    sig = (
        amp * np.sin(2 * np.pi * base_freq * t + rng.uniform(0, 2 * np.pi))
        + 0.3 * np.sin(2 * np.pi * 2.5 * base_freq * t + rng.uniform(0, 2 * np.pi))
        + 0.2 * np.sin(2 * np.pi * 5.2 * base_freq * t + rng.uniform(0, 2 * np.pi))
    )
    noise = 0.05 * rng.standard_normal(size=n)
    return sig + noise


def generate_dataset(rpms: List[float], loads: List[float], fs: float, n: int, samples_per_cond: int,
                     seed: int = 0) -> Dict[Tuple[float, float], np.ndarray]:
    rng = np.random.default_rng(seed)
    data = {}
    for rpm in rpms:
        for load in loads:
            samples = [synth_signal(rpm, load, fs, n, rng) for _ in range(samples_per_cond)]
            data[(rpm, load)] = np.stack(samples, axis=0)
    return data

# ---------------------- processing steps ----------------------

def align_and_average(samples: np.ndarray) -> np.ndarray:
    """Step A: phase-align via cross-correlation then average."""
    ref = samples[0]
    aligned = []
    for s in samples:
        corr = correlate(ref, s, mode="full")
        lag = np.argmax(corr) - (len(s) - 1)
        aligned.append(np.roll(s, lag))
    return np.mean(aligned, axis=0)


def resample_to_angle(sig: np.ndarray, rpm: float, fs: float, n_angle: int) -> Tuple[np.ndarray, np.ndarray]:
    """Step B: resample signal to angular domain; returns (theta_grid, s_theta)."""
    n = len(sig)
    t = np.arange(n) / fs
    theta = 2 * np.pi * (rpm / 60.0) * t  # radians
    theta_grid = np.linspace(theta[0], theta[-1], n_angle, endpoint=False)
    s_theta = np.interp(theta_grid, theta, sig)
    return theta_grid, s_theta


def rms_and_normalize(sig: np.ndarray) -> Tuple[float, np.ndarray]:
    rms = float(np.sqrt(np.mean(sig ** 2)) + 1e-12)
    return rms, sig / rms


def idw_weights(points: np.ndarray, target: np.ndarray, power: float, k: int) -> np.ndarray:
    d = np.linalg.norm(points - target[None, :], axis=1)
    order = np.argsort(d)
    sel = order[: max(1, min(k, len(order)))]
    dsel = d[sel]
    if dsel[0] < 1e-12:
        w = np.zeros_like(dsel)
        w[0] = 1.0
        return sel, w
    w = 1.0 / np.power(dsel + 1e-12, power)
    w = w / w.sum()
    return sel, w


def order_tracking_idw(data: Dict[Tuple[float, float], np.ndarray], fs: float, n_angle: int,
                       target_rpm: float, target_load: float, power: float = 2.0, k: int = 6) -> np.ndarray:
    # preprocess each condition
    cond_keys = list(data.keys())
    pts = np.array(cond_keys, dtype=np.float64)  # (rpm, load)

    processed = []
    for (rpm, load), samples in data.items():
        avg = align_and_average(samples)
        _, s_theta = resample_to_angle(avg, rpm=rpm, fs=fs, n_angle=n_angle)
        rms, norm = rms_and_normalize(s_theta)
        processed.append((rpm, load, norm, rms))

    processed_pts = np.array([[p[0], p[1]] for p in processed], dtype=np.float64)
    target = np.array([target_rpm, target_load], dtype=np.float64)
    sel_idx, weights = idw_weights(processed_pts, target, power=power, k=k)

    # shape interpolation
    norms = np.stack([processed[i][2] for i in sel_idx], axis=0)  # (k, n_angle)
    norm_interp = np.tensordot(weights, norms, axes=(0, 0))

    # energy interpolation
    rmss = np.array([processed[i][3] for i in sel_idx], dtype=np.float64)
    rms_interp = float(np.dot(weights, rmss))

    # resample back to time domain at target rpm
    t = np.arange(n_angle) / fs
    theta_target = 2 * np.pi * (target_rpm / 60.0) * t
    # map norm_interp(theta_grid) back: theta_grid is uniform 0..2pi*rev; reuse same grid length
    theta_grid = np.linspace(theta_target[0], theta_target[-1], n_angle, endpoint=False)
    time_sig = np.interp(theta_target, theta_grid, norm_interp) * rms_interp
    return time_sig

# ---------------------- visualization ----------------------

def plot_compare(sig_true: np.ndarray, sig_pred: np.ndarray, fs: float, title: str, out: Path) -> None:
    t = np.arange(len(sig_true)) / fs
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    axes[0].plot(t, sig_true, label="True", lw=1.2)
    axes[0].plot(t, sig_pred, label="Pred", lw=1.0, alpha=0.9)
    axes[0].set_title(title + " (time domain)")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    freqs = rfftfreq(len(sig_true), 1 / fs)
    ft_true = np.abs(rfft(sig_true))
    ft_pred = np.abs(rfft(sig_pred))
    axes[1].plot(freqs, ft_true, label="True mag", lw=1.2)
    axes[1].plot(freqs, ft_pred, label="Pred mag", lw=1.0, alpha=0.9)
    axes[1].set_title("Frequency domain (magnitude)")
    axes[1].set_xlabel("Hz")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

# ---------------------- main ----------------------

def main():
    parser = argparse.ArgumentParser(description="Order-tracking + IDW interpolation demo with synthetic data")
    parser.add_argument("--fs", type=float, default=25600, help="Sampling rate")
    parser.add_argument("--n", type=int, default=2048, help="Samples per signal")
    parser.add_argument("--samples-per-cond", type=int, default=10, help="Samples per condition")
    parser.add_argument("--target-rpm", type=float, default=2350)
    parser.add_argument("--target-load", type=float, default=30)
    parser.add_argument("--power", type=float, default=2.0, help="IDW power")
    parser.add_argument("--k", type=int, default=6, help="IDW neighbors")
    parser.add_argument("--out-dir", type=str, default="./order_tracking_demo")
    args = parser.parse_args()

    rpms = [2200, 2300, 2400, 2500]
    loads = [20, 30, 40]
    fs = args.fs
    n = args.n
    n_angle = n  # keep same length in angle grid

    data = generate_dataset(rpms, loads, fs, n, samples_per_cond=args.samples_per_cond, seed=42)

    # build a synthetic "ground truth" for target condition to evaluate
    gt = synth_signal(args.target_rpm, args.target_load, fs, n, np.random.default_rng(123))

    pred = order_tracking_idw(
        data,
        fs=fs,
        n_angle=n_angle,
        target_rpm=args.target_rpm,
        target_load=args.target_load,
        power=args.power,
        k=args.k,
    )

    # metrics
    diff = pred - gt
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rms_gt = float(np.sqrt(np.mean(gt ** 2)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pred.npy", pred)
    np.save(out_dir / "gt.npy", gt)
    json.dump({"mae": mae, "rmse": rmse, "rms_gt": rms_gt}, open(out_dir / "report.json", "w"), indent=2)

    plot_compare(gt, pred, fs=fs, title="Order-tracking IDW interpolation", out=out_dir / "comparison.png")
    print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, rms_gt={rms_gt:.4f}")
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()

