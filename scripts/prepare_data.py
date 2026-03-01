#!/usr/bin/env python3
"""Download raw IMU datasets and convert to standardized HDF5 format.

Usage:
    python scripts/prepare_data.py                        # all datasets
    python scripts/prepare_data.py --dataset broad euroc   # specific datasets
    python scripts/prepare_data.py --keep-raw              # keep raw downloads
    python scripts/prepare_data.py --data-dir ./data       # custom output dir
    python scripts/prepare_data.py --list                  # list available datasets
"""

import argparse
import shutil
import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from riann.prep import PREPARERS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download raw IMU datasets and convert to HDF5."
    )
    parser.add_argument(
        "--dataset", nargs="+", metavar="NAME",
        help=f"Datasets to process (default: all). Choices: {', '.join(PREPARERS)}",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / "data",
        help="Output directory for HDF5 files (default: ./data)",
    )
    parser.add_argument(
        "--keep-raw", action="store_true",
        help="Keep raw downloaded files after conversion",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_datasets",
        help="List available datasets and exit",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download step (use existing raw files)",
    )
    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets:")
        for name in PREPARERS:
            print(f"  {name}")
        return

    names = args.dataset or list(PREPARERS.keys())
    for name in names:
        if name not in PREPARERS:
            print(f"Error: unknown dataset '{name}'. Use --list to see choices.")
            sys.exit(1)

    raw_dir = args.data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name in names:
        mod = PREPARERS[name]
        print(f"\n{'='*60}")
        print(f"  Dataset: {name}")
        print(f"{'='*60}")

        if not args.skip_download:
            print(f"\n[{name}] Downloading ...")
            mod.download(raw_dir)

        print(f"\n[{name}] Converting ...")
        mod.convert(raw_dir, args.data_dir)

    if not args.keep_raw and raw_dir.exists():
        print(f"\nCleaning up raw downloads: {raw_dir}")
        shutil.rmtree(raw_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
