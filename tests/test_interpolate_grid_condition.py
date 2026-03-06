import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "interpolate_grid_condition.py"


def _write_condition(base: Path, load: int, rpm: int, signal: np.ndarray) -> None:
    cond = base / f"load_{load}" / f"rpm_{rpm}"
    cond.mkdir(parents=True, exist_ok=True)
    # Save two samples with identical values so mean is deterministic.
    samples = np.stack([signal, signal], axis=0)
    np.save(cond / "all_generated.npy", samples)
    np.save(cond / "filtered_0.npy", signal)


def _build_linear_signal(load: float, rpm: float, length: int = 64) -> np.ndarray:
    x = np.linspace(0.0, 1.0, length)
    # Linear in (load, rpm), so bilinear interpolation should be exact.
    return (2.0 * load + 0.1 * rpm) + 0.5 * x


def test_bilinear_leave_one_out_exact() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="grid_interp_test_"))
    out = tmp / "out"
    try:
        grid = tmp / "generated_grid"
        loads = [28, 30, 32]
        rpms = [2300, 2350, 2400]

        for l in loads:
            for r in rpms:
                _write_condition(grid, l, r, _build_linear_signal(l, r))

        cmd = [
            sys.executable,
            str(SCRIPT),
            "--grid-root",
            str(grid),
            "--target-load",
            "30",
            "--target-rpm",
            "2350",
            "--aggregation",
            "all_mean",
            "--interpolation",
            "bilinear",
            "--output-dir",
            str(out),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        with (out / "report.json").open("r", encoding="utf-8") as f:
            report = json.load(f)

        rmse = float(report["metrics"]["rmse"])
        mae = float(report["metrics"]["mae"])
        assert rmse < 1e-10, f"rmse too large: {rmse}"
        assert mae < 1e-10, f"mae too large: {mae}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    test_bilinear_leave_one_out_exact()
    print("test_interpolate_grid_condition.py passed")

