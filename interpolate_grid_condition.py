import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np


Point = Tuple[float, float]


def _parse_value_from_dirname(name: str, prefix: str) -> Optional[float]:
    if not name.startswith(prefix):
        return None
    raw = name[len(prefix):]
    try:
        return float(raw)
    except ValueError:
        return None


def discover_grid_points(grid_root: Path) -> List[Point]:
    points: List[Point] = []
    if not grid_root.exists():
        raise FileNotFoundError(f"Grid directory does not exist: {grid_root}")

    for load_dir in grid_root.iterdir():
        if not load_dir.is_dir():
            continue
        load_val = _parse_value_from_dirname(load_dir.name, "load_")
        if load_val is None:
            continue

        for rpm_dir in load_dir.iterdir():
            if not rpm_dir.is_dir():
                continue
            rpm_val = _parse_value_from_dirname(rpm_dir.name, "rpm_")
            if rpm_val is None:
                continue
            points.append((load_val, rpm_val))

    if not points:
        raise RuntimeError(f"No condition points found under: {grid_root}")

    return sorted(points)


def _load_all_generated_mean(cond_dir: Path) -> np.ndarray:
    p = cond_dir / "all_generated.npy"
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    arr = np.load(p)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected shape in {p}: {arr.shape}")
    return arr.mean(axis=0).astype(np.float64)


def _load_filtered_mean(cond_dir: Path) -> np.ndarray:
    files = sorted(cond_dir.glob("filtered_*.npy"))
    if not files:
        return _load_all_generated_mean(cond_dir)

    signals = []
    for p in files:
        sig = np.load(p).astype(np.float64).reshape(-1)
        signals.append(sig)
    return np.stack(signals, axis=0).mean(axis=0)


def load_representative_signal(grid_root: Path, load_val: float, rpm_val: float, aggregation: str) -> np.ndarray:
    cond_dir = grid_root / f"load_{int(load_val)}" / f"rpm_{int(rpm_val)}"
    if not cond_dir.exists():
        raise FileNotFoundError(f"Condition folder not found: {cond_dir}")

    if aggregation == "all_mean":
        return _load_all_generated_mean(cond_dir)
    if aggregation == "filtered_mean":
        return _load_filtered_mean(cond_dir)
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def load_point_signals(grid_root: Path, points: List[Point], aggregation: str) -> Dict[Point, np.ndarray]:
    mapping: Dict[Point, np.ndarray] = {}
    for load_val, rpm_val in points:
        try:
            mapping[(load_val, rpm_val)] = load_representative_signal(grid_root, load_val, rpm_val, aggregation)
        except Exception:
            continue

    if not mapping:
        raise RuntimeError("No valid signal data found in grid folders.")

    lengths = {v.shape[0] for v in mapping.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"Signals have inconsistent lengths: {sorted(lengths)}")

    return mapping


def _neighbor_pair(axis_values: np.ndarray, target: float, exclude_exact: bool) -> Tuple[Optional[float], Optional[float]]:
    values = np.unique(axis_values)
    lower_candidates = values[values < target] if exclude_exact else values[values <= target]
    upper_candidates = values[values > target] if exclude_exact else values[values >= target]

    lower = float(lower_candidates.max()) if lower_candidates.size > 0 else None
    upper = float(upper_candidates.min()) if upper_candidates.size > 0 else None
    return lower, upper


def _bilinear_or_linear(
    p00: np.ndarray,
    p10: np.ndarray,
    p01: np.ndarray,
    p11: np.ndarray,
    tx: float,
    ty: float,
) -> np.ndarray:
    a = (1 - tx) * p00 + tx * p10
    b = (1 - tx) * p01 + tx * p11
    return (1 - ty) * a + ty * b


def bilinear_interpolate(
    point_signals: Dict[Point, np.ndarray],
    target_load: float,
    target_rpm: float,
    exclude_exact: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    loads = np.array([k[0] for k in point_signals.keys()], dtype=np.float64)
    rpms = np.array([k[1] for k in point_signals.keys()], dtype=np.float64)

    l0, l1 = _neighbor_pair(loads, target_load, exclude_exact)
    r0, r1 = _neighbor_pair(rpms, target_rpm, exclude_exact)

    if l0 is None or l1 is None or r0 is None or r1 is None:
        raise RuntimeError("Cannot form a 2x2 neighborhood for bilinear interpolation.")

    c00 = (l0, r0)
    c10 = (l1, r0)
    c01 = (l0, r1)
    c11 = (l1, r1)

    required = [c00, c10, c01, c11]
    missing = [c for c in required if c not in point_signals]
    if missing:
        raise RuntimeError(f"Missing bilinear corner points: {missing}")

    tx = (target_load - l0) / (l1 - l0)
    ty = (target_rpm - r0) / (r1 - r0)

    pred = _bilinear_or_linear(
        point_signals[c00],
        point_signals[c10],
        point_signals[c01],
        point_signals[c11],
        tx,
        ty,
    )

    meta = {
        "method": "bilinear",
        "neighbors": [c00, c10, c01, c11],
        "tx": float(tx),
        "ty": float(ty),
    }
    return pred, meta


def _estimate_axis_scale(values: np.ndarray) -> float:
    uniq = np.unique(values)
    if uniq.size < 2:
        return 1.0
    diffs = np.diff(np.sort(uniq))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 1.0
    return float(np.median(diffs))


def idw_interpolate(
    point_signals: Dict[Point, np.ndarray],
    target_load: float,
    target_rpm: float,
    k: int,
    power: float,
    exclude_exact: bool,
) -> Tuple[np.ndarray, Dict[str, object]]:
    points = np.array(list(point_signals.keys()), dtype=np.float64)
    load_scale = _estimate_axis_scale(points[:, 0])
    rpm_scale = _estimate_axis_scale(points[:, 1])

    d_load = (points[:, 0] - target_load) / max(load_scale, 1e-12)
    d_rpm = (points[:, 1] - target_rpm) / max(rpm_scale, 1e-12)
    d = np.sqrt(d_load ** 2 + d_rpm ** 2)

    if exclude_exact:
        d = np.where(d < 1e-12, np.inf, d)

    order = np.argsort(d)
    finite_idx = [idx for idx in order if np.isfinite(d[idx])]
    if not finite_idx:
        raise RuntimeError("No valid neighbors for IDW interpolation.")

    chosen = finite_idx[: max(1, min(k, len(finite_idx)))]
    distances = d[chosen]

    if np.any(distances < 1e-12):
        exact_idx = chosen[int(np.argmin(distances))]
        exact_point = tuple(points[exact_idx].tolist())
        return point_signals[(exact_point[0], exact_point[1])].copy(), {
            "method": "idw_exact_hit",
            "neighbors": [exact_point],
            "weights": [1.0],
            "distance_norm": [0.0],
        }

    weights = 1.0 / np.power(distances + 1e-12, power)
    weights = weights / np.sum(weights)

    stacked = []
    neighbor_points = []
    for idx in chosen:
        p = tuple(points[idx].tolist())
        neighbor_points.append((float(p[0]), float(p[1])))
        stacked.append(point_signals[(p[0], p[1])])

    pred = np.tensordot(weights, np.stack(stacked, axis=0), axes=(0, 0))
    meta = {
        "method": "idw",
        "neighbors": neighbor_points,
        "weights": [float(x) for x in weights],
        "distance_norm": [float(x) for x in distances],
        "load_scale": load_scale,
        "rpm_scale": rpm_scale,
    }
    return pred, meta


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))

    y_range = float(np.ptp(y_true) + 1e-12)
    nrmse = rmse / y_range

    true_centered = y_true - y_true.mean()
    pred_centered = y_pred - y_pred.mean()
    denom = np.linalg.norm(true_centered) * np.linalg.norm(pred_centered) + 1e-12
    pearson = float(np.dot(true_centered, pred_centered) / denom)

    fft_true = np.abs(np.fft.rfft(y_true))
    fft_pred = np.abs(np.fft.rfft(y_pred))
    spec_rmse = float(np.sqrt(np.mean((fft_pred - fft_true) ** 2)))
    spec_true_norm = float(np.linalg.norm(fft_true) + 1e-12)
    spec_nrmse = spec_rmse / spec_true_norm

    return {
        "mae": mae,
        "rmse": rmse,
        "nrmse_range": float(nrmse),
        "max_abs_error": max_abs,
        "pearson_r": pearson,
        "spectrum_rmse": spec_rmse,
        "spectrum_nrmse": float(spec_nrmse),
    }


def interpolate_and_evaluate(
    grid_root: Path,
    target_load: float,
    target_rpm: float,
    aggregation: str,
    interpolation: str,
    exclude_target: bool,
    idw_k: int,
    idw_power: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    points = discover_grid_points(grid_root)
    point_signals = load_point_signals(grid_root, points, aggregation)

    target_key = (float(target_load), float(target_rpm))
    if target_key not in point_signals:
        raise RuntimeError(
            f"Target condition ({target_load}, {target_rpm}) not found. "
            "Need true data for accuracy comparison."
        )
    y_true = point_signals[target_key]

    train_points = dict(point_signals)
    if exclude_target and target_key in train_points:
        del train_points[target_key]

    method_meta: Dict[str, object]
    if interpolation in {"bilinear", "auto"}:
        try:
            y_pred, method_meta = bilinear_interpolate(
                train_points, target_load, target_rpm, exclude_exact=exclude_target
            )
        except Exception:
            if interpolation == "bilinear":
                raise
            y_pred, method_meta = idw_interpolate(
                train_points,
                target_load,
                target_rpm,
                k=idw_k,
                power=idw_power,
                exclude_exact=exclude_target,
            )
    elif interpolation == "idw":
        y_pred, method_meta = idw_interpolate(
            train_points,
            target_load,
            target_rpm,
            k=idw_k,
            power=idw_power,
            exclude_exact=exclude_target,
        )
    else:
        raise ValueError(f"Unsupported interpolation mode: {interpolation}")

    metrics = compute_metrics(y_true, y_pred)
    report: Dict[str, object] = {
        "target": {"load": float(target_load), "rpm": float(target_rpm)},
        "aggregation": aggregation,
        "interpolation": interpolation,
        "exclude_target": exclude_target,
        "method_meta": method_meta,
        "metrics": metrics,
        "signal_length": int(y_true.shape[0]),
        "grid_point_count": int(len(point_signals)),
    }
    return y_pred, y_true, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interpolate target condition from generated_grid and compare with real generated data."
    )
    parser.add_argument("--grid-root", type=str, default="./generated_grid", help="Path to generated grid root.")
    parser.add_argument("--target-load", type=float, required=True, help="Target load value, e.g. 30")
    parser.add_argument("--target-rpm", type=float, required=True, help="Target rpm value, e.g. 2350")
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["all_mean", "filtered_mean"],
        default="filtered_mean",
        help="How to aggregate each condition folder into one representative signal.",
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=["auto", "bilinear", "idw"],
        default="auto",
        help="Interpolation method: auto prefers bilinear then falls back to IDW.",
    )
    parser.add_argument("--idw-k", type=int, default=8, help="Neighbor count for IDW.")
    parser.add_argument("--idw-power", type=float, default=2.0, help="Distance power for IDW.")
    parser.add_argument(
        "--include-target",
        action="store_true",
        help="Include target point as interpolation source if it exists (not recommended for fair evaluation).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./interpolation_results",
        help="Directory to save report and signals.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grid_root = Path(args.grid_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_pred, y_true, report = interpolate_and_evaluate(
        grid_root=grid_root,
        target_load=args.target_load,
        target_rpm=args.target_rpm,
        aggregation=args.aggregation,
        interpolation=args.interpolation,
        exclude_target=not args.include_target,
        idw_k=args.idw_k,
        idw_power=args.idw_power,
    )

    np.save(output_dir / "interpolated_signal.npy", y_pred)
    np.save(output_dir / "target_signal.npy", y_true)
    np.save(output_dir / "difference.npy", y_pred - y_true)

    report_path = output_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    method_meta = cast(Dict[str, object], report["method_meta"])
    metrics = cast(Dict[str, float], report["metrics"])

    print("Interpolation complete.")
    print(f"Target: load={args.target_load}, rpm={args.target_rpm}")
    print(f"Aggregation: {args.aggregation}")
    print(f"Method: {method_meta.get('method', 'unknown')}")
    for key, value in metrics.items():
        print(f"{key}: {value:.6e}")
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()

