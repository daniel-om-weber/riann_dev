"""OxIOD dataset — Oxford Inertial Odometry Dataset.

Source: http://deepio.cs.ox.ac.uk/
Output: data/OxIOD/OxIOD::{category}_data{N}_{M}:fixed.hdf5  (71 files)

Requires manual download: user must obtain the ZIP and place it in
data/raw/OxIOD/.  The script prints instructions if the data is missing.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ._common import fix_quaternion_flips, interpolate_nans, write_hdf5

G = 9.80665  # m/s^2

# Expected output files derived from existing data directory listing.
EXPECTED_FILES = [
    "OxIOD::handbag_data1_1:fixed",
    "OxIOD::handbag_data1_2:fixed",
    "OxIOD::handbag_data1_3:fixed",
    "OxIOD::handbag_data1_4:fixed",
    "OxIOD::handbag_data2_1:fixed",
    "OxIOD::handbag_data2_2:fixed",
    "OxIOD::handbag_data2_3:fixed",
    "OxIOD::handbag_data2_4:fixed",
    "OxIOD::handheld_data1_1:fixed",
    "OxIOD::handheld_data1_2:fixed",
    "OxIOD::handheld_data1_3:fixed",
    "OxIOD::handheld_data1_4:fixed",
    "OxIOD::handheld_data1_5:fixed",
    "OxIOD::handheld_data1_6:fixed",
    "OxIOD::handheld_data1_7:fixed",
    "OxIOD::handheld_data2_1:fixed",
    "OxIOD::handheld_data2_2:fixed",
    "OxIOD::handheld_data2_3:fixed",
    "OxIOD::handheld_data3_1:fixed",
    "OxIOD::handheld_data3_2:fixed",
    "OxIOD::handheld_data3_3:fixed",
    "OxIOD::handheld_data3_4:fixed",
    "OxIOD::handheld_data3_5:fixed",
    "OxIOD::handheld_data4_1:fixed",
    "OxIOD::handheld_data4_2:fixed",
    "OxIOD::handheld_data4_3:fixed",
    "OxIOD::handheld_data4_4:fixed",
    "OxIOD::handheld_data4_5:fixed",
    "OxIOD::handheld_data5_1:fixed",
    "OxIOD::handheld_data5_2:fixed",
    "OxIOD::handheld_data5_3:fixed",
    "OxIOD::handheld_data5_4:fixed",
    "OxIOD::pocket_data1_1:fixed",
    "OxIOD::pocket_data1_2:fixed",
    "OxIOD::pocket_data1_3:fixed",
    "OxIOD::pocket_data1_4:fixed",
    "OxIOD::pocket_data1_5:fixed",
    "OxIOD::pocket_data2_1:fixed",
    "OxIOD::pocket_data2_2:fixed",
    "OxIOD::pocket_data2_3:fixed",
    "OxIOD::pocket_data2_4:fixed",
    "OxIOD::pocket_data2_5:fixed",
    "OxIOD::pocket_data2_6:fixed",
    "OxIOD::running_data1_1:fixed",
    "OxIOD::running_data1_2:fixed",
    "OxIOD::running_data1_3:fixed",
    "OxIOD::running_data1_4:fixed",
    "OxIOD::running_data1_5:fixed",
    "OxIOD::running_data1_6:fixed",
    "OxIOD::running_data1_7:fixed",
    "OxIOD::slow_walking_data1_1:fixed",
    "OxIOD::slow_walking_data1_2:fixed",
    "OxIOD::slow_walking_data1_3:fixed",
    "OxIOD::slow_walking_data1_4:fixed",
    "OxIOD::slow_walking_data1_5:fixed",
    "OxIOD::slow_walking_data1_6:fixed",
    "OxIOD::slow_walking_data1_7:fixed",
    "OxIOD::slow_walking_data1_8:fixed",
    "OxIOD::trolley_data1_1:fixed",
    "OxIOD::trolley_data1_2:fixed",
    "OxIOD::trolley_data1_3:fixed",
    "OxIOD::trolley_data1_4:fixed",
    "OxIOD::trolley_data1_5:fixed",
    "OxIOD::trolley_data1_6:fixed",
    "OxIOD::trolley_data1_7:fixed",
    "OxIOD::trolley_data2_1:fixed",
    "OxIOD::trolley_data2_2:fixed",
    "OxIOD::trolley_data2_3:fixed",
    "OxIOD::trolley_data2_4:fixed",
    "OxIOD::trolley_data2_5:fixed",
    "OxIOD::trolley_data2_6:fixed",
]


def _parse_expected(name: str) -> tuple[str, int, int]:
    """Parse 'OxIOD::category_dataN_M:fixed' → (category, N, M)."""
    m = re.match(r"OxIOD::(.+)_data(\d+)_(\d+):fixed", name)
    assert m, f"Cannot parse {name}"
    return m.group(1), int(m.group(2)), int(m.group(3))


def download(raw_dir: Path) -> None:
    """OxIOD requires manual download — print instructions."""
    oxiod_dir = raw_dir / "OxIOD"
    if not oxiod_dir.exists() or not any(oxiod_dir.iterdir()):
        print(
            "\n"
            "  ┌──────────────────────────────────────────────────────────┐\n"
            "  │  OxIOD requires manual download.                        │\n"
            "  │                                                         │\n"
            "  │  1. Visit http://deepio.cs.ox.ac.uk/                    │\n"
            "  │  2. Fill out the Google Form to get the download link    │\n"
            "  │  3. Download the ZIP and extract it to:                  │\n"
            f"  │     {oxiod_dir}/                      │\n"
            "  │                                                         │\n"
            "  │  The extracted folder should contain subdirs like:       │\n"
            "  │     handheld/data1/syn/imu1.csv                         │\n"
            "  │     handheld/data1/syn/vi1.csv                          │\n"
            "  └──────────────────────────────────────────────────────────┘\n"
        )
    else:
        print("  OxIOD raw data found.")


def _find_imu_vi_pairs(oxiod_dir: Path) -> list[tuple[str, int, int, Path, Path]]:
    """Discover imu/vi CSV pairs in the OxIOD directory tree.

    Returns list of (category, data_num, seq_num, imu_path, vi_path).
    """
    pairs = []
    for expected in EXPECTED_FILES:
        cat, dnum, snum = _parse_expected(expected)
        # OxIOD directory structure: {category}/data{N}/syn/imu{M}.csv
        imu_path = oxiod_dir / cat / f"data{dnum}" / "syn" / f"imu{snum}.csv"
        vi_path = oxiod_dir / cat / f"data{dnum}" / "syn" / f"vi{snum}.csv"
        if imu_path.exists() and vi_path.exists():
            pairs.append((cat, dnum, snum, imu_path, vi_path))
    return pairs


def convert(raw_dir: Path, out_dir: Path) -> None:
    raw_dir = raw_dir / "OxIOD"
    out_dir = out_dir / "OxIOD"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print("  OxIOD raw data not found. Run with download step first.")
        return

    pairs = _find_imu_vi_pairs(raw_dir)
    if not pairs:
        print("  No imu/vi CSV pairs found in OxIOD raw data.")
        return

    for cat, dnum, snum, imu_path, vi_path in pairs:
        out_name = f"OxIOD::{cat}_data{dnum}_{snum}:fixed.hdf5"
        out_path = out_dir / out_name
        if out_path.exists():
            print(f"  Skipping (exists): {out_path.name}")
            continue

        # ── Load IMU CSV ──
        # Headers include unit suffixes: gravity_x(G), rotation_rate_x(radians/s), etc.
        # Strip parenthetical suffixes for easier access.
        imu_df = pd.read_csv(imu_path)
        imu_df.columns = imu_df.columns.str.strip().str.replace(r"\(.*\)", "", regex=True)

        # Reconstruct raw accelerometer: gravity + user_acceleration
        # Both are in G units → convert to m/s^2
        acc_x = (imu_df["gravity_x"] + imu_df["user_acc_x"]) * G
        acc_y = (imu_df["gravity_y"] + imu_df["user_acc_y"]) * G
        acc_z = (imu_df["gravity_z"] + imu_df["user_acc_z"]) * G

        gyr_x = imu_df["rotation_rate_x"]
        gyr_y = imu_df["rotation_rate_y"]
        gyr_z = imu_df["rotation_rate_z"]

        mag_x = imu_df["magnetic_field_x"]
        mag_y = imu_df["magnetic_field_y"]
        mag_z = imu_df["magnetic_field_z"]

        imu_time = imu_df["Time"].values

        # ── Load VI (visual-inertial ground truth) CSV ──
        vi_df = pd.read_csv(vi_path)
        vi_df.columns = vi_df.columns.str.strip()

        # Quaternion: scalar-last in CSV (rotation.x/y/z/w) → reorder to wxyz
        vi_time = vi_df["Time"].values
        quat_vi = vi_df[["rotation.w", "rotation.x", "rotation.y", "rotation.z"]].values

        # ── Align timestamps ──
        # Build IMU dataframe with uniform columns
        imu_aligned = pd.DataFrame({
            "time": imu_time,
            "acc_x": acc_x.values if hasattr(acc_x, "values") else acc_x,
            "acc_y": acc_y.values if hasattr(acc_y, "values") else acc_y,
            "acc_z": acc_z.values if hasattr(acc_z, "values") else acc_z,
            "gyr_x": gyr_x.values if hasattr(gyr_x, "values") else gyr_x,
            "gyr_y": gyr_y.values if hasattr(gyr_y, "values") else gyr_y,
            "gyr_z": gyr_z.values if hasattr(gyr_z, "values") else gyr_z,
            "mag_x": mag_x.values if hasattr(mag_x, "values") else mag_x,
            "mag_y": mag_y.values if hasattr(mag_y, "values") else mag_y,
            "mag_z": mag_z.values if hasattr(mag_z, "values") else mag_z,
        })

        vi_aligned = pd.DataFrame({
            "time": vi_time,
            "qw": quat_vi[:, 0],
            "qx": quat_vi[:, 1],
            "qy": quat_vi[:, 2],
            "qz": quat_vi[:, 3],
        })

        imu_aligned.sort_values("time", inplace=True)
        vi_aligned.sort_values("time", inplace=True)

        merged = pd.merge_asof(imu_aligned, vi_aligned, on="time", direction="nearest")

        # ── Resample to uniform 100 Hz ──
        t0 = merged["time"].iloc[0]
        merged["time_rel"] = merged["time"] - t0
        dt_target = 0.01  # 100 Hz
        t_max = merged["time_rel"].iloc[-1]
        n_samples = int(t_max / dt_target) + 1
        t_uniform = np.arange(n_samples) * dt_target

        # Interpolate each signal to uniform grid
        result = {}
        for col in ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z",
                     "mag_x", "mag_y", "mag_z", "qw", "qx", "qy", "qz"]:
            result[col] = np.interp(t_uniform, merged["time_rel"].values, merged[col].values)

        acc = np.column_stack([result["acc_x"], result["acc_y"], result["acc_z"]])
        gyr = np.column_stack([result["gyr_x"], result["gyr_y"], result["gyr_z"]])
        mag = np.column_stack([result["mag_x"], result["mag_y"], result["mag_z"]])
        quat = np.column_stack([result["qw"], result["qx"], result["qy"], result["qz"]])

        quat = fix_quaternion_flips(quat)

        print(f"  Writing {out_path.name}  ({len(acc)} samples)")
        write_hdf5(out_path, acc, gyr, quat, dt=dt_target, mag=mag)
