"""
interpolate_b.py
================
Method B (spectral amplitude-phase decoupled interpolation) for arbitrary target conditions.

Algorithm:
  1. Load ensemble-averaged signals from generated_grid for all neighbor conditions
  2. IDW-weighted interpolation of amplitude spectra
  3. Circular weighted mean of phase spectra (no target signal needed)
  4. IFFT to reconstruct the time-domain signal

Usage (interactive):
  python interpolate_b.py

Usage (CLI):
  python interpolate_b.py --target-load 30 --target-rpm 2350
  python interpolate_b.py --target-load 25 --target-rpm 2100 --k 8 --power 3
"""

import argparse
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Fix Windows console encoding so all print() calls work regardless of locale
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
from scipy.signal import correlate


# ─────────────────────────── data loading ───────────────────────────

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


# ─────────────────────────── IDW weights ───────────────────────────

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
    if ds[0] < 1e-12:          # exact grid hit
        w = np.zeros_like(ds)
        w[0] = 1.0
        return sel, w
    w = 1.0 / np.power(ds + 1e-12, power)
    return sel, w / w.sum()


# ─────────────────────────── Method B core ───────────────────────────

def interpolate_spectral_b(
        sig_map: Dict[Tuple[float, float], np.ndarray],
        target_load: float,
        target_rpm: float,
        k: int = 6,
        power: float = 2.0,
) -> np.ndarray:
    """
    Method B: spectral amplitude-phase decoupled interpolation.

    Steps:
      1. IDW-select k nearest neighbor conditions
      2. rfft each neighbor signal -> extract magnitude spectra
      3. IDW-weighted interpolation of magnitude spectra
      4. Circular weighted mean of phase spectra (exp(i*phi) sum -> angle)
      5. irfft to reconstruct time-domain signal
    """
    pts = np.array(list(sig_map.keys()), dtype=np.float64)   # columns: (load, rpm)
    n_sig = len(next(iter(sig_map.values())))
    sel, w = idw_weights(pts, np.array([target_load, target_rpm]), k=k, power=power)

    neighbor_ffts = np.stack(
        [rfft(sig_map[(float(pts[i, 0]), float(pts[i, 1]))]) for i in sel]  # (k, F)
    )

    # -- magnitude IDW interpolation --
    mags = np.abs(neighbor_ffts)                                     # (k, F)
    mag_interp = np.tensordot(w, mags, axes=(0, 0))                 # (F,)

    # -- phase circular weighted mean --
    # normalise each FFT bin to unit circle, weighted sum, then take angle
    unit_phasors = neighbor_ffts / (np.abs(neighbor_ffts) + 1e-12)  # (k, F)
    phase_vec    = np.tensordot(w, unit_phasors, axes=(0, 0))       # (F,)
    phase_interp = np.angle(phase_vec)                              # (F,)

    return irfft(mag_interp * np.exp(1j * phase_interp), n=n_sig).real


# ─────────────────────────── visualization ───────────────────────────

def plot_result(
        pred: np.ndarray,
        y_true: Optional[np.ndarray],
        fs: float,
        target_load: float,
        target_rpm: float,
        out: Path,
) -> None:
    """Plot interpolation result (time + freq domain), overlay true signal if available."""
    n = len(pred)
    t = np.arange(n) / fs
    freqs = rfftfreq(n, 1 / fs)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    if y_true is not None:
        axes[0].plot(t, y_true, "k", lw=1.8, label="True (generated_grid)", zorder=5)
        axes[1].plot(freqs, np.abs(rfft(y_true)), "k", lw=1.8,
                     label="True (generated_grid)", zorder=5)

    axes[0].plot(t, pred, "tab:orange", lw=1.2, alpha=0.9, label="Method B (interpolated)")
    axes[0].set_title(f"Time-domain  |  load={target_load}, rpm={target_rpm}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(freqs, np.abs(rfft(pred)), "tab:orange", lw=1.2, alpha=0.9,
                 label="Method B (interpolated)")
    axes[1].set_title("Frequency-domain magnitude")
    axes[1].set_xlabel("Hz")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(0, min(float(freqs[-1]), 5000))

    fig.suptitle(f"Method B Interpolation  -  load={target_load}, rpm={target_rpm}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Figure saved  -> {out}", flush=True)


# ─────────────────────────── metrics ───────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mae   = float(np.mean(np.abs(diff)))
    rmse  = float(np.sqrt(np.mean(diff ** 2)))
    nrmse = rmse / (float(np.ptp(y_true)) + 1e-12)
    tc = y_true - y_true.mean()
    pc = y_pred - y_pred.mean()
    pearson = float(np.dot(tc, pc) /
                    (np.linalg.norm(tc) * np.linalg.norm(pc) + 1e-12))
    ft = np.abs(rfft(y_true))
    fp = np.abs(rfft(y_pred))
    spec_nrmse = float(np.sqrt(np.mean((fp - ft) ** 2)) /
                       (np.linalg.norm(ft) + 1e-12))
    yt_norm = tc / (float(np.std(y_true)) + 1e-12)
    yp_norm = pc / (float(np.std(y_pred)) + 1e-12)
    cross_corr = correlate(yt_norm, yp_norm, mode="full")
    max_aligned_r = float(np.max(cross_corr) / len(y_true))
    return dict(mae=mae, rmse=rmse, nrmse=nrmse, pearson_r=pearson,
                spec_nrmse=spec_nrmse, max_aligned_r=max_aligned_r)


def print_metrics(m: Dict[str, float]) -> None:
    print(f"  MAE          = {m['mae']:.6f}", flush=True)
    print(f"  RMSE         = {m['rmse']:.6f}", flush=True)
    print(f"  NRMSE        = {m['nrmse']:.6f}", flush=True)
    print(f"  Pearson_r    = {m['pearson_r']:.6f}  (point-to-point; affected by random phase)", flush=True)
    print(f"  Spec_NRMSE   = {m['spec_nrmse']:.6f}  (spectral fit error; lower is better)", flush=True)
    print(f"  Max_Align_R  = {m['max_aligned_r']:.6f}  (shift-invariant corr; closer to 1 is better)", flush=True)


# ─────────────────────────── interactive input ───────────────────────────

def prompt_condition(all_pts: List[Tuple[float, float]]) -> Tuple[float, float]:
    """Prompt user for target condition with range validation."""
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
        return tl, tr


# ─────────────────────────── main ───────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Method B spectral interpolation: generate bearing signals for any target condition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--grid-root",    default="./generated_grid",
                        help="Path to generated_grid root directory")
    parser.add_argument("--target-load",  type=float, default=None,
                        help="Target load value; if omitted, prompted interactively")
    parser.add_argument("--target-rpm",   type=float, default=None,
                        help="Target RPM value; if omitted, prompted interactively")
    parser.add_argument("--fs",           type=float, default=25600,
                        help="Sampling frequency in Hz (used for time-axis in plots)")
    parser.add_argument("--k",            type=int,   default=6,
                        help="Number of IDW neighbors")
    parser.add_argument("--power",        type=float, default=2.0,
                        help="IDW distance exponent")
    parser.add_argument("--out-dir",      default="./interpolation_b_results",
                        help="Output directory")
    parser.add_argument("--no-compare",   action="store_true",
                        help="Skip metric comparison even if target is on the grid")
    args = parser.parse_args()

    # -- set up output dir and tee logger FIRST before any print --
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
    """Main logic after logging is set up."""
    grid_root = Path(args.grid_root)
    if not grid_root.exists():
        print(f"ERROR: grid_root directory not found: '{grid_root}'", flush=True)
        sys.exit(1)

    # -- discover grid --
    print("Discovering grid points ...", flush=True)
    all_pts = discover_grid(grid_root)
    if not all_pts:
        print("ERROR: No condition directories found in grid_root.", flush=True)
        sys.exit(1)
    print(f"Found {len(all_pts)} conditions.", flush=True)

    # -- get target condition --
    if args.target_load is not None and args.target_rpm is not None:
        target_load = args.target_load
        target_rpm  = args.target_rpm
    else:
        target_load, target_rpm = prompt_condition(all_pts)

    print(f"Target condition: load={target_load}, rpm={target_rpm}", flush=True)

    # -- load data --
    print("Loading signals ...", flush=True)
    sig_map = load_grid(grid_root, all_pts)
    print(f"Loaded {len(sig_map)} signals.", flush=True)

    # check if target is exactly on the grid
    target_key = (target_load, target_rpm)
    is_on_grid = target_key in sig_map

    # exclude target itself from training data for fair evaluation
    train_map = {k: v for k, v in sig_map.items() if k != target_key}

    # -- interpolate --
    print("Running Method B interpolation ...", flush=True)
    pred = interpolate_spectral_b(
        train_map,
        target_load=target_load,
        target_rpm=target_rpm,
        k=args.k,
        power=args.power,
    )
    print("Interpolation complete.", flush=True)


    tag = f"load{target_load:.0f}_rpm{target_rpm:.0f}"

    # -- save predicted signal --
    pred_path = out_dir / f"pred_{tag}.npy"
    np.save(pred_path, pred)
    print(f"Prediction saved -> {pred_path}", flush=True)

    # -- metrics (only if target is on the grid) --
    y_true: Optional[np.ndarray] = None
    report: Dict = {
        "target": {"load": target_load, "rpm": target_rpm},
        "on_grid": is_on_grid,
        "params": {"k": args.k, "power": args.power},
        "metrics": None,
    }

    if is_on_grid and not args.no_compare:
        y_true = sig_map[target_key]
        print("\n[On-grid target] Computing error metrics:", flush=True)
        m = compute_metrics(y_true, pred)
        print_metrics(m)
        report["metrics"] = m
        true_path = out_dir / f"true_{tag}.npy"
        np.save(true_path, y_true)
        print(f"Ground truth saved -> {true_path}", flush=True)
    else:
        msg = "[Off-grid target] Pure interpolation mode - no ground truth to compare." \
              if not is_on_grid else "[--no-compare] Skipping metric comparison."
        print(f"\n{msg}", flush=True)

    # -- save report --
    report_path = out_dir / f"report_{tag}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report   saved -> {report_path}", flush=True)

    # -- plot --
    fig_path = out_dir / f"comparison_{tag}.png"
    plot_result(
        pred=pred,
        y_true=y_true,
        fs=args.fs,
        target_load=target_load,
        target_rpm=target_rpm,
        out=fig_path,
    )

    print(f"\nDone. All outputs saved to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()




