#!/usr/bin/env python3
"""Validate prepared HDF5 data using VQF orientation estimation.

Runs qmt.oriEstOfflineVQF on accelerometer/gyroscope data (and magnetometer
where available) and compares against the optical ground truth stored in each
file.  Low inclination RMSE confirms that the IMU data and ground truth
orientations are physically consistent.

Usage:
    python scripts/validate_vqf.py                        # all datasets
    python scripts/validate_vqf.py --dataset TUM-VI       # specific dataset folder
    python scripts/validate_vqf.py --no-mag               # ignore magnetometer
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import qmt


def load_hdf5(path: Path) -> dict:
    """Load IMU data and ground truth from standardized HDF5."""
    with h5py.File(path, "r") as f:
        d = {
            "acc": np.column_stack([f["acc_x"][:], f["acc_y"][:], f["acc_z"][:]]),
            "gyr": np.column_stack([f["gyr_x"][:], f["gyr_y"][:], f["gyr_z"][:]]),
            "gt": np.column_stack(
                [f["opt_a"][:], f["opt_b"][:], f["opt_c"][:], f["opt_d"][:]]
            ),
            "dt": float(f["dt"][0]),
        }
        if "mag_x" in f:
            d["mag"] = np.column_stack(
                [f["mag_x"][:], f["mag_y"][:], f["mag_z"][:]]
            )
    return d


def validate_file(path: Path, use_mag: bool = True) -> dict | None:
    """Run VQF on one file and return error metrics vs ground truth."""
    d = load_hdf5(path)
    acc, gyr, gt, dt = d["acc"], d["gyr"], d["gt"], d["dt"]
    mag = d.get("mag") if use_mag else None

    # Run offline VQF
    vqf = qmt.oriEstOfflineVQF(gyr, acc, mag=mag, params={"Ts": dt})

    # Valid ground truth mask (non-NaN)
    valid = ~np.isnan(gt[:, 0])
    if valid.sum() < 10:
        return None

    gt_v = gt[valid]
    vqf_v = vqf[valid]

    # Normalize quaternions to unit norm
    gt_v = gt_v / np.linalg.norm(gt_v, axis=1, keepdims=True)
    vqf_v = vqf_v / np.linalg.norm(vqf_v, axis=1, keepdims=True)

    # Align VQF to GT reference frame at first valid sample
    q_offset = qmt.qmult(gt_v[0], qmt.qinv(vqf_v[0]))
    vqf_aligned = qmt.qmult(q_offset, vqf_v)

    # Error quaternion: identity when perfect
    q_err = qmt.qrel(gt_v, vqf_aligned)

    # Total angular error
    angles = np.abs(qmt.quatAngle(q_err))

    # Heading / inclination decomposition
    heading, inclination = qmt.headingInclinationAngle(q_err)

    return {
        "samples": len(acc),
        "valid": int(valid.sum()),
        "has_mag": "mag" in d,
        "used_mag": mag is not None,
        "total_rmse": np.rad2deg(np.sqrt(np.mean(angles**2))),
        "incl_rmse": np.rad2deg(np.sqrt(np.mean(inclination**2))),
        "head_rmse": np.rad2deg(np.sqrt(np.mean(heading**2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate prepared HDF5 data with offline VQF."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Root data directory (default: ./data)",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        metavar="NAME",
        help="Dataset folder names to validate (default: all)",
    )
    parser.add_argument(
        "--no-mag",
        action="store_true",
        help="Ignore magnetometer even when available",
    )
    args = parser.parse_args()

    # Discover HDF5 files
    if args.dataset:
        dirs = [args.data_dir / name for name in args.dataset]
    else:
        dirs = sorted(d for d in args.data_dir.iterdir() if d.is_dir())

    files = []
    for d in dirs:
        if not d.exists():
            print(f"Warning: {d} not found, skipping")
            continue
        files.extend(sorted(d.glob("*.hdf5")))

    if not files:
        print("No HDF5 files found.")
        sys.exit(1)

    hdr = (
        f"{'File':<45} {'N':>7} {'Valid':>7} {'Mag':>5} "
        f"{'Total':>7} {'Incl':>7} {'Head':>7}"
    )
    print(hdr)
    print("-" * len(hdr))

    for path in files:
        label = f"{path.parent.name}/{path.stem}"
        if len(label) > 44:
            label = "…" + label[-43:]
        try:
            result = validate_file(path, use_mag=not args.no_mag)
        except Exception as e:
            print(f"{label:<45} ERROR: {e}")
            continue
        if result is None:
            print(f"{label:<45} (insufficient ground truth)")
            continue
        mag_str = (
            "yes" if result["used_mag"] else ("avail" if result["has_mag"] else "-")
        )
        print(
            f"{label:<45} {result['samples']:>7} {result['valid']:>7} {mag_str:>5} "
            f"{result['total_rmse']:>6.1f}° {result['incl_rmse']:>6.1f}° "
            f"{result['head_rmse']:>6.1f}°"
        )

    print("-" * len(hdr))
    print("Total = angular RMSE, Incl = inclination RMSE, Head = heading RMSE (degrees)")


if __name__ == "__main__":
    main()
