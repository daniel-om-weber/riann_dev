#!/usr/bin/env python3
"""Compare newly generated HDF5 files against reference data/ files.

Usage:
    python scripts/compare_hdf5.py --new-dir /tmp/riann_verify/data --dataset Myon
    python scripts/compare_hdf5.py --new-dir /tmp/riann_verify/data              # all datasets
    python scripts/compare_hdf5.py --new-dir ... --allow-extra-keys -v           # verbose
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def compare_one(ref_path: Path, new_path: Path, allow_extra: bool) -> list[str]:
    """Compare a single HDF5 file pair. Returns list of error strings (empty = match)."""
    errors = []
    warnings = []

    with h5py.File(ref_path, "r") as ref, h5py.File(new_path, "r") as new:
        ref_keys = set(ref.keys())
        new_keys = set(new.keys())

        # 1. Key check
        missing = ref_keys - new_keys
        extra = new_keys - ref_keys
        if missing:
            errors.append(f"Missing keys in new: {sorted(missing)}")
        if extra:
            if allow_extra:
                warnings.append(f"Extra keys in new (allowed): {sorted(extra)}")
            else:
                errors.append(f"Extra keys in new: {sorted(extra)}")

        # Compare shared keys
        shared = ref_keys & new_keys
        for key in sorted(shared):
            ref_data = ref[key][:]
            new_data = new[key][:]

            # 2. Shape check
            if ref_data.shape != new_data.shape:
                errors.append(
                    f"  {key}: shape mismatch ref={ref_data.shape} new={new_data.shape}"
                )
                continue

            # 3. Dtype check
            if ref_data.dtype != np.float32:
                errors.append(f"  {key}: ref dtype {ref_data.dtype} != float32")
            if new_data.dtype != np.float32:
                errors.append(f"  {key}: new dtype {new_data.dtype} != float32")

            # 4. NaN pattern check
            ref_nan = np.isnan(ref_data)
            new_nan = np.isnan(new_data)
            if not np.array_equal(ref_nan, new_nan):
                ref_count = ref_nan.sum()
                new_count = new_nan.sum()
                mismatch_count = (ref_nan != new_nan).sum()
                errors.append(
                    f"  {key}: NaN pattern mismatch — "
                    f"ref has {ref_count} NaNs, new has {new_count} NaNs, "
                    f"{mismatch_count} positions differ"
                )
                # Find first divergent index
                diff_idx = np.argmax(ref_nan != new_nan)
                errors.append(f"    first divergent index: {diff_idx}")
                continue

            # 5. Value check
            if not np.allclose(ref_data, new_data, atol=1e-6, rtol=1e-5, equal_nan=True):
                abs_diff = np.abs(ref_data - new_data)
                # Ignore NaN positions
                valid = ~np.isnan(abs_diff)
                if valid.any():
                    max_diff = abs_diff[valid].max()
                    max_idx = np.unravel_index(
                        np.nanargmax(abs_diff), abs_diff.shape
                    )
                    errors.append(
                        f"  {key}: values differ — max abs diff = {max_diff:.2e} at index {max_idx}"
                    )
                else:
                    errors.append(f"  {key}: values differ (all NaN comparison)")

    return errors, warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare new HDF5 files against reference.")
    parser.add_argument("--new-dir", type=Path, required=True, help="Directory with new HDF5 files")
    parser.add_argument(
        "--ref-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data",
        help="Reference data directory (default: ./data)",
    )
    parser.add_argument("--dataset", type=str, help="Compare only this dataset subdirectory")
    parser.add_argument("--allow-extra-keys", action="store_true", help="Treat extra keys as warnings")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show per-file details")
    args = parser.parse_args()

    ref_base = args.ref_dir
    new_base = args.new_dir

    # Discover dataset subdirectories
    if args.dataset:
        datasets = [args.dataset]
    else:
        datasets = sorted(
            d.name for d in ref_base.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    summary_rows = []
    all_ok = True

    for ds in datasets:
        ref_dir = ref_base / ds
        new_dir = new_base / ds

        if not ref_dir.exists():
            print(f"WARNING: reference directory not found: {ref_dir}")
            continue

        ref_files = sorted(ref_dir.glob("*.hdf5"))
        if not ref_files:
            print(f"WARNING: no HDF5 files in {ref_dir}")
            continue

        match_count = 0
        fail_count = 0
        missing_count = 0
        extra_count = 0
        file_errors = {}

        for ref_path in ref_files:
            new_path = new_dir / ref_path.name
            if not new_path.exists():
                missing_count += 1
                file_errors[ref_path.name] = ["FILE MISSING"]
                continue

            errors, warnings = compare_one(ref_path, new_path, args.allow_extra_keys)
            if errors:
                fail_count += 1
                file_errors[ref_path.name] = errors
            else:
                match_count += 1

            if warnings and args.verbose:
                for w in warnings:
                    print(f"  WARN [{ds}/{ref_path.name}]: {w}")

        # Check for extra files in new dir
        if new_dir.exists():
            new_files = {f.name for f in new_dir.glob("*.hdf5")}
            ref_names = {f.name for f in ref_files}
            extra_names = new_files - ref_names
            extra_count = len(extra_names)

        total = len(ref_files)
        status = "PASS" if (fail_count == 0 and missing_count == 0) else "FAIL"
        if status == "FAIL":
            all_ok = False
        summary_rows.append((ds, total, match_count, fail_count, missing_count, extra_count, status))

        # Print per-file errors
        if file_errors and (args.verbose or status == "FAIL"):
            for fname, errs in sorted(file_errors.items()):
                print(f"\n  FAIL {ds}/{fname}:")
                for e in errs:
                    print(f"    {e}")

    # Summary table
    print("\n" + "=" * 75)
    print(f"{'Dataset':<25} {'Files':>5} {'Match':>5} {'Fail':>5} {'Miss':>5} {'Extra':>5} {'Status':>7}")
    print("-" * 75)
    for row in summary_rows:
        ds, total, match, fail, miss, extra, status = row
        print(f"{ds:<25} {total:>5} {match:>5} {fail:>5} {miss:>5} {extra:>5} {status:>7}")
    print("=" * 75)

    overall = "ALL PASS" if all_ok else "SOME FAILURES"
    print(f"\nOverall: {overall}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
