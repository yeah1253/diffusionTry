"""
interpolate_b_augment.py
========================
Method B (spectral amplitude-phase decoupled interpolation) + Data Augmentation.

Differences from interpolate_b.py:
  - No on-grid validation / metrics / report
  - After interpolation, the single predicted signal is augmented into N samples
    (default 10) using a two-stage pipeline:

  Stage 1 – Frequency-domain augmentation (operates on complex spectrum)
    1. Spectral Envelope Modulation  : smooth random gain curve via CubicSpline
    2. Phase Jitter (power-preserving): random phase noise ±jitter_rad, DC kept fixed

  Stage 2 – Time-domain augmentation (operates on reconstructed waveform)
    3. Random Cyclic Shift + Amplitude Scale : np.roll + scalar multiply
    4. SNR-Controlled Noise Injection        : Gaussian noise at random SNR (dB)

Usage (interactive):
  python interpolate_b_augment.py

Usage (CLI):
  python interpolate_b_augment.py --target-load 25 --target-rpm 2350
  python interpolate_b_augment.py --target-load 25 --target-rpm 2350 --n-aug 10 --k 6
"""

import argparse
import io
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows console encoding
try:
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass


class _Tee(io.TextIOBase):
    """Write to both the original stream and a log file simultaneously."""
    def __init__(self, stream, log_path: Path):
        self._stream = stream
        self._log = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, s):
        try:
            self._stream.write(s)
            self._stream.flush()
        except Exception:
            pass
        self._log.write(s)
        return len(s)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        self._log.flush()


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import rfft, irfft, rfftfreq
from scipy.interpolate import CubicSpline


# ═══════════════════════════════════════════════════════════════════
#  Data loading  (identical to interpolate_b.py)
# ═══════════════════════════════════════════════════════════════════

def load_filtered_mean(cond_dir: Path) -> np.ndarray:
    """Load mean of all filtered_*.npy in a condition dir; fall back to all_generated.npy."""
    files = sorted(cond_dir.glob("filtered_*.npy"))
    if files:
        return np.stack([np.load(f).astype(np.float64).reshape(-1) for f in files]).mean(0)
    arr = np.load(cond_dir / "all_generated.npy").astype(np.float64)
    return arr.mean(0) if arr.ndim == 2 else arr


def discover_grid(grid_root: Path) -> List[Tuple[float, float]]:
    """Scan generated_grid and return sorted list of (load, rpm) condition points."""
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


def load_grid(grid_root: Path,
              all_pts: List[Tuple[float, float]]) -> Dict[Tuple[float, float], np.ndarray]:
    """Load ensemble-averaged signals for all conditions."""
    sig_map: Dict[Tuple[float, float], np.ndarray] = {}
    for lv, rv in all_pts:
        cdir = grid_root / f"load_{int(lv)}" / f"rpm_{int(rv)}"
        if cdir.exists():
            sig_map[(lv, rv)] = load_filtered_mean(cdir)
    return sig_map


# ═══════════════════════════════════════════════════════════════════
#  IDW weights  (identical to interpolate_b.py)
# ═══════════════════════════════════════════════════════════════════

def _scale(arr: np.ndarray) -> float:
    u = np.unique(arr)
    if u.size < 2:
        return 1.0
    d = np.diff(np.sort(u))
    return float(np.median(d[d > 0])) or 1.0


def idw_weights(pts: np.ndarray, target: np.ndarray, k: int, power: float):
    """Compute IDW weights; returns (selected_indices, weights)."""
    ls = _scale(pts[:, 0])
    rs = _scale(pts[:, 1])
    d = np.sqrt(((pts[:, 0] - target[0]) / ls) ** 2 +
                ((pts[:, 1] - target[1]) / rs) ** 2)
    order = np.argsort(d)
    sel = order[:max(1, min(k, len(order)))]
    ds = d[sel]
    if ds[0] < 1e-12:
        w = np.zeros_like(ds)
        w[0] = 1.0
        return sel, w
    w = 1.0 / np.power(ds + 1e-12, power)
    return sel, w / w.sum()


# ═══════════════════════════════════════════════════════════════════
#  Method B interpolation core  (identical to interpolate_b.py)
# ═══════════════════════════════════════════════════════════════════

def interpolate_spectral_b(
        sig_map: Dict[Tuple[float, float], np.ndarray],
        target_load: float,
        target_rpm: float,
        k: int = 6,
        power: float = 2.0,
) -> np.ndarray:
    """
    Method B: spectral amplitude-phase decoupled interpolation.
    Returns a single reconstructed time-domain signal.
    """
    pts = np.array(list(sig_map.keys()), dtype=np.float64)
    n_sig = len(next(iter(sig_map.values())))
    sel, w = idw_weights(pts, np.array([target_load, target_rpm]), k=k, power=power)

    neighbor_ffts = np.stack(
        [rfft(sig_map[(float(pts[i, 0]), float(pts[i, 1]))]) for i in sel]
    )

    mags = np.abs(neighbor_ffts)
    mag_interp = np.tensordot(w, mags, axes=(0, 0))

    unit_phasors = neighbor_ffts / (np.abs(neighbor_ffts) + 1e-12)
    phase_vec    = np.tensordot(w, unit_phasors, axes=(0, 0))
    phase_interp = np.angle(phase_vec)

    return irfft(mag_interp * np.exp(1j * phase_interp), n=n_sig).real


# ═══════════════════════════════════════════════════════════════════
#  Two-stage data augmentation
# ═══════════════════════════════════════════════════════════════════

def _stage1_freq_augment(
        signal: np.ndarray,
        gain_low: float = 0.6,
        gain_high: float = 1.4,
        n_ctrl_pts: int = 5,
        jitter_rad: float = 0.25,
        rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Stage 1: Frequency-domain augmentation on the complex spectrum.

    Step A – Spectral Envelope Modulation:
        Build a smooth gain curve from n_ctrl_pts random control points via
        CubicSpline and multiply the complex spectrum bin-by-bin.

    Step B – Power-preserving Phase Jitter:
        Keep magnitude unchanged; add uniform random noise in [-jitter_rad, +jitter_rad]
        to each phase bin except DC (index 0).
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signal)
    spec = rfft(signal)          # complex, shape (F,)
    F = len(spec)

    # ── Step A: Spectral Envelope Modulation ──────────────────────
    # Control point x-positions spread uniformly across [0, F-1]
    x_ctrl = np.linspace(0, F - 1, n_ctrl_pts)
    y_ctrl = rng.uniform(gain_low, gain_high, size=n_ctrl_pts)
    cs = CubicSpline(x_ctrl, y_ctrl, bc_type="natural")
    x_all = np.arange(F, dtype=np.float64)
    gain_curve = np.clip(cs(x_all), 0.0, None)   # ensure non-negative gain
    spec = spec * gain_curve                       # modulate complex spectrum

    # ── Step B: Power-preserving Phase Jitter ─────────────────────
    mag   = np.abs(spec)
    phase = np.angle(spec)
    # inject noise to all bins except DC (index 0)
    noise = rng.uniform(-jitter_rad, jitter_rad, size=F)
    noise[0] = 0.0               # DC phase stays fixed
    phase_new = phase + noise
    spec_new  = mag * np.exp(1j * phase_new)

    return irfft(spec_new, n=n).real


def _stage2_time_augment(
        signal: np.ndarray,
        scale_low: float = 0.85,
        scale_high: float = 1.15,
        snr_db_low: float = 15.0,
        snr_db_high: float = 25.0,
        rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Stage 2: Time-domain macroscopic augmentation.

    Step C – Random Cyclic Shift + Amplitude Scale:
        Circular shift by a random integer steps; multiply by a random scalar.

    Step D – SNR-Controlled Gaussian Noise Injection:
        Compute AC power (after removing DC); inject white Gaussian noise
        calibrated to a randomly chosen SNR in [snr_db_low, snr_db_high].
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signal)

    # ── Step C: Cyclic Shift + Amplitude Scale ────────────────────
    shift  = rng.integers(0, n)
    scale  = rng.uniform(scale_low, scale_high)
    sig_c  = np.roll(signal, shift) * scale

    # ── Step D: SNR-Controlled Noise Injection ────────────────────
    # AC power: remove DC first
    sig_ac = sig_c - sig_c.mean()
    ac_power = float(np.mean(sig_ac ** 2))

    snr_db    = rng.uniform(snr_db_low, snr_db_high)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = ac_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(max(noise_power, 0.0)), size=n)
    sig_out = sig_c + noise

    return sig_out


def augment_signal(
        signal: np.ndarray,
        n_aug: int = 10,
        gain_low: float = 0.6,
        gain_high: float = 1.4,
        n_ctrl_pts: int = 5,
        jitter_rad: float = 0.25,
        scale_low: float = 0.85,
        scale_high: float = 1.15,
        snr_db_low: float = 15.0,
        snr_db_high: float = 25.0,
        seed: Optional[int] = None,
) -> np.ndarray:
    """
    Run the full two-stage augmentation pipeline n_aug times.

    Returns
    -------
    augmented : np.ndarray, shape (n_aug, len(signal))
        Each row is an independently augmented variant of the input signal.
    """
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(n_aug):
        # Stage 1: frequency-domain
        s1 = _stage1_freq_augment(
            signal,
            gain_low=gain_low,
            gain_high=gain_high,
            n_ctrl_pts=n_ctrl_pts,
            jitter_rad=jitter_rad,
            rng=rng,
        )
        # Stage 2: time-domain
        s2 = _stage2_time_augment(
            s1,
            scale_low=scale_low,
            scale_high=scale_high,
            snr_db_low=snr_db_low,
            snr_db_high=snr_db_high,
            rng=rng,
        )
        results.append(s2)
    return np.stack(results, axis=0)   # (n_aug, N)


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_augmented(
        pred: np.ndarray,
        augmented: np.ndarray,
        fs: float,
        target_load: float,
        target_rpm: float,
        out: Path,
        max_show: int = 5,
) -> None:
    """
    Plot the interpolated base signal and up to max_show augmented variants
    in both time and frequency domains.
    """
    n = len(pred)
    t = np.arange(n) / fs
    freqs = rfftfreq(n, 1.0 / fs)

    n_show = min(max_show, len(augmented))
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    # ── Time domain ──────────────────────────────────────────────
    axes[0].plot(t, pred, color="black", lw=2.0, label="Interpolated (base)", zorder=10)
    cmap = plt.cm.get_cmap("tab10", n_show)
    for i in range(n_show):
        axes[0].plot(t, augmented[i], color=cmap(i), lw=0.8, alpha=0.7,
                     label=f"Aug #{i+1}")
    axes[0].set_title(f"Time-domain  |  load={target_load}, rpm={target_rpm}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].grid(alpha=0.3)

    # ── Frequency domain ─────────────────────────────────────────
    axes[1].plot(freqs, np.abs(rfft(pred)), color="black", lw=2.0,
                 label="Interpolated (base)", zorder=10)
    for i in range(n_show):
        axes[1].plot(freqs, np.abs(rfft(augmented[i])), color=cmap(i),
                     lw=0.8, alpha=0.7, label=f"Aug #{i+1}")
    axes[1].set_title("Frequency-domain magnitude")
    axes[1].set_xlabel("Hz")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend(fontsize=7, ncol=3)
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, min(float(freqs[-1]), 5000))

    fig.suptitle(
        f"Method B Interpolation + Data Augmentation  |  "
        f"load={target_load}, rpm={target_rpm}  |  {len(augmented)} augmented samples",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved  -> {out}", flush=True)


# ═══════════════════════════════════════════════════════════════════
#  Interactive prompt
# ═══════════════════════════════════════════════════════════════════

def prompt_condition(all_pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    loads = sorted(set(p[0] for p in all_pts))
    rpms  = sorted(set(p[1] for p in all_pts))
    print(f"\nGrid load range : {loads[0]:.0f} ~ {loads[-1]:.0f}  (step ~{loads[1]-loads[0]:.0f})", flush=True)
    print(f"Grid RPM  range : {rpms[0]:.0f} ~ {rpms[-1]:.0f}  (step ~{rpms[1]-rpms[0]:.0f})", flush=True)
    print("(Target may be off-grid; neighbor conditions will be used for interpolation)\n", flush=True)
    while True:
        try:
            tl = float(input("Enter target Load : ").strip())
            tr = float(input("Enter target RPM  : ").strip())
        except ValueError:
            print("  Invalid input, please enter numbers.", flush=True)
            continue
        if tl < loads[0] or tl > loads[-1]:
            print(f"  Warning: Load={tl} is outside grid range [{loads[0]}, {loads[-1]}]. Extrapolating.", flush=True)
        if tr < rpms[0] or tr > rpms[-1]:
            print(f"  Warning: RPM={tr} is outside grid range [{rpms[0]}, {rpms[-1]}]. Extrapolating.", flush=True)
        return float(tl), float(tr)


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Method B spectral interpolation + two-stage data augmentation.\n"
            "Produces n_aug augmented bearing signals for any target condition."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── grid / interpolation ──────────────────────────────────────
    parser.add_argument("--grid-root",    default="./generated_grid",
                        help="Path to generated_grid root directory")
    parser.add_argument("--target-load",  type=float, default=None)
    parser.add_argument("--target-rpm",   type=float, default=None)
    parser.add_argument("--fs",           type=float, default=25600,
                        help="Sampling frequency in Hz")
    parser.add_argument("--k",            type=int,   default=6,
                        help="Number of IDW neighbors")
    parser.add_argument("--power",        type=float, default=2.0,
                        help="IDW distance exponent")
    # ── augmentation ─────────────────────────────────────────────
    parser.add_argument("--n-aug",        type=int,   default=10,
                        help="Number of augmented samples to generate (default: 10)")
    parser.add_argument("--gain-low",     type=float, default=0.6,
                        help="Spectral envelope gain lower bound (default: 0.6)")
    parser.add_argument("--gain-high",    type=float, default=1.4,
                        help="Spectral envelope gain upper bound (default: 1.4)")
    parser.add_argument("--n-ctrl-pts",   type=int,   default=5,
                        help="Number of CubicSpline control points for envelope (default: 5)")
    parser.add_argument("--jitter-rad",   type=float, default=0.25,
                        help="Phase jitter amplitude in radians (default: 0.25)")
    parser.add_argument("--scale-low",    type=float, default=0.85,
                        help="Amplitude scale lower bound (default: 0.85)")
    parser.add_argument("--scale-high",   type=float, default=1.15,
                        help="Amplitude scale upper bound (default: 1.15)")
    parser.add_argument("--snr-low",      type=float, default=15.0,
                        help="SNR lower bound in dB (default: 15)")
    parser.add_argument("--snr-high",     type=float, default=25.0,
                        help="SNR upper bound in dB (default: 25)")
    parser.add_argument("--seed",         type=int,   default=None,
                        help="Random seed for reproducibility")
    # ── output ───────────────────────────────────────────────────
    parser.add_argument("--out-dir",      default="./interpolation_b_augment_results",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"
    sys.stdout = _Tee(sys.stdout, log_path)

    try:
        _run(args, out_dir)
    except Exception:
        traceback.print_exc()
        sys.exit(1)


def _run(args, out_dir: Path) -> None:
    grid_root = Path(args.grid_root)
    if not grid_root.exists():
        print(f"ERROR: grid_root directory not found: '{grid_root}'", flush=True)
        sys.exit(1)

    # ── discover & load grid ──────────────────────────────────────
    print("Discovering grid points ...", flush=True)
    all_pts = discover_grid(grid_root)
    if not all_pts:
        print("ERROR: No condition directories found in grid_root.", flush=True)
        sys.exit(1)
    print(f"Found {len(all_pts)} conditions.", flush=True)

    # ── target condition ─────────────────────────────────────────
    if args.target_load is not None and args.target_rpm is not None:
        target_load = args.target_load
        target_rpm  = args.target_rpm
    else:
        target_load, target_rpm = prompt_condition(all_pts)
    print(f"Target condition: load={target_load}, rpm={target_rpm}", flush=True)

    # ── load signals (all grid points, no exclusion needed) ──────
    print("Loading signals ...", flush=True)
    sig_map = load_grid(grid_root, all_pts)
    print(f"Loaded {len(sig_map)} signals.", flush=True)

    # ── Method B interpolation ────────────────────────────────────
    print("Running Method B interpolation ...", flush=True)
    pred = interpolate_spectral_b(
        sig_map,
        target_load=target_load,
        target_rpm=target_rpm,
        k=args.k,
        power=args.power,
    )
    print("Interpolation complete.", flush=True)

    tag = f"load{target_load:.0f}_rpm{target_rpm:.0f}"

    # save base interpolated signal
    pred_path = out_dir / f"pred_{tag}.npy"
    np.save(pred_path, pred)
    print(f"Base prediction saved  -> {pred_path}", flush=True)

    # ── Data Augmentation ─────────────────────────────────────────
    print(f"\nRunning two-stage augmentation  (n_aug={args.n_aug}) ...", flush=True)
    print(f"  Stage-1 params : gain=[{args.gain_low}, {args.gain_high}], "
          f"ctrl_pts={args.n_ctrl_pts}, jitter={args.jitter_rad} rad", flush=True)
    print(f"  Stage-2 params : scale=[{args.scale_low}, {args.scale_high}], "
          f"SNR=[{args.snr_low}, {args.snr_high}] dB", flush=True)

    augmented = augment_signal(
        pred,
        n_aug=args.n_aug,
        gain_low=args.gain_low,
        gain_high=args.gain_high,
        n_ctrl_pts=args.n_ctrl_pts,
        jitter_rad=args.jitter_rad,
        scale_low=args.scale_low,
        scale_high=args.scale_high,
        snr_db_low=args.snr_low,
        snr_db_high=args.snr_high,
        seed=args.seed,
    )
    print(f"Augmentation complete.  Output shape: {augmented.shape}", flush=True)

    # save all augmented samples as a single .npy array (n_aug, N)
    aug_stack_path = out_dir / f"augmented_{tag}.npy"
    np.save(aug_stack_path, augmented)
    print(f"Augmented stack saved  -> {aug_stack_path}  shape={augmented.shape}", flush=True)

    # also save each sample individually for convenience
    for i, s in enumerate(augmented):
        sp = out_dir / f"augmented_{tag}_{i:02d}.npy"
        np.save(sp, s)
    print(f"Individual files saved -> {out_dir}/augmented_{tag}_00.npy  ...  "
          f"augmented_{tag}_{args.n_aug-1:02d}.npy", flush=True)

    # ── Visualization ─────────────────────────────────────────────
    fig_path = out_dir / f"augmentation_{tag}.png"
    plot_augmented(
        pred=pred,
        augmented=augmented,
        fs=args.fs,
        target_load=target_load,
        target_rpm=target_rpm,
        out=fig_path,
    )

    print(f"\nDone. All outputs saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()


