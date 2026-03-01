"""Dataset configuration, file discovery, splits, and DataLoader creation."""

from pathlib import Path

import h5py
import numpy as np

from tsfast.tsdata import (
    FileEntry,
    WindowedDataset,
    get_hdf_files,
)
from tsfast.tsdata.pipeline import DataLoaders, create_dls_from_readers
from tsfast.tsdata.readers import HDF5Signals, Resampled

# ── Signal definitions ────────────────────────────────────────────────────────

u_raw = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"]
u_dt = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "dt"]
y = ["opt_a", "opt_b", "opt_c", "opt_d"]

# ── Data paths ────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ── Split definitions ─────────────────────────────────────────────────────────

MYON_VALID_IDS = [14, 39, 21]
MYON_TEST_IDS = [29, 22, 35]

TUMVI_TRAIN_ROOMS = ["room1", "room2", "room3"]


def _myon_split(files: list[Path]) -> dict[str, list[Path]]:
    """Split Myon files by subject ID into train/valid/test."""
    valid_prefixes = [f"{x:02d}_" for x in MYON_VALID_IDS]
    test_prefixes = [f"{x:02d}_" for x in MYON_TEST_IDS]

    valid, test, train = [], [], []
    for f in files:
        name = f.name
        if any(name.startswith(p) for p in valid_prefixes):
            valid.append(f)
        elif any(name.startswith(p) for p in test_prefixes):
            test.append(f)
        else:
            train.append(f)
    return {"train": train, "valid": valid, "test": test}


def _tumvi_split(files: list[Path]) -> dict[str, list[Path]]:
    """Split TUM-VI files by room number into train/test."""
    train, test = [], []
    for f in files:
        if any(room in f.name for room in TUMVI_TRAIN_ROOMS):
            train.append(f)
        else:
            test.append(f)
    return {"train": train, "valid": [], "test": test}


def get_files(data_dir: Path | None = None) -> dict[str, list[Path]]:
    """Discover all dataset files and return train/valid/test splits.

    Combines Myon and TUM-VI splits. Other datasets (OxIOD, RepoIMU,
    EuRoC-MAV, Caruso-Sassari) are placed in the test set.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR

    f_myon = get_hdf_files(data_dir / "Myon", recurse=False)
    f_tumvi = get_hdf_files(data_dir / "TUM-VI", recurse=False)

    myon = _myon_split(f_myon)
    tumvi = _tumvi_split(f_tumvi)

    train = myon["train"] + tumvi["train"]
    valid = myon["valid"]
    test = myon["test"] + tumvi["test"]

    # Add other datasets as test-only
    for name in ["OxIOD", "RepoIMU", "EuRoC-MAV", "Caruso-Sassari", "Caruso-Sassari_orig"]:
        extra = get_hdf_files(data_dir / name, recurse=False)
        test.extend(extra)

    return {"train": train, "valid": valid, "test": test}


def _read_src_fs(path: Path) -> float:
    """Read source sampling frequency from an HDF5 file's dt dataset."""
    with h5py.File(path, "r") as f:
        return float(1.0 / f["dt"][0])


# ── DataLoader creation ──────────────────────────────────────────────────────


def get_dls(
    data_dir: Path | None = None,
    win_sz: int = 9000,
    stp_sz: int = 60,
    bs: int = 64,
    n_batches_train: int = 300,
    targ_fs: list[float] | None = None,
) -> DataLoaders:
    """Create DataLoaders for GAE training with on-the-fly resampling.

    Args:
        data_dir: root data directory (defaults to repo data/)
        win_sz: window size in resampled samples
        stp_sz: step size between training windows
        bs: batch size
        n_batches_train: number of training batches per epoch
        targ_fs: target sampling frequencies for resampling.
            Defaults to 100 equidistant rates between 50 and 500 Hz.
    """
    if targ_fs is None:
        targ_fs = np.linspace(50, 500, 100).tolist()

    splits = get_files(data_dir)

    # Build per-file resampling factors
    def _make_entries(files: list[Path]) -> list[FileEntry]:
        entries = []
        for f in files:
            src_fs = _read_src_fs(f)
            for tf in targ_fs:
                factor = tf / src_fs
                entries.append(FileEntry(path=str(f), resampling_factor=factor))
        return entries

    train_entries = _make_entries(splits["train"])
    valid_entries = _make_entries(splits["valid"])
    test_entries = _make_entries(splits["test"]) if splits["test"] else []

    # Readers with on-the-fly resampling
    inputs = Resampled(HDF5Signals(u_dt, dt_idx=6))
    targets = Resampled(HDF5Signals(y))

    # Build datasets
    train_ds = WindowedDataset(train_entries, inputs, targets, win_sz=win_sz, stp_sz=stp_sz)
    valid_ds = WindowedDataset(valid_entries, inputs, targets, win_sz=win_sz, stp_sz=win_sz)

    # Build DataLoaders
    from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    import torch

    n_samples_train = n_batches_train * bs
    train_sampler = RandomSampler(train_ds, replacement=True, num_samples=n_samples_train)
    valid_sampler = SequentialSampler(valid_ds)

    train_dl = DataLoader(
        train_ds, batch_size=bs, sampler=train_sampler,
        drop_last=True, pin_memory=torch.cuda.is_available(),
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=bs, sampler=valid_sampler,
        drop_last=False, pin_memory=torch.cuda.is_available(),
    )

    test_dl = None
    if test_entries:
        test_ds = WindowedDataset(test_entries, inputs, targets, win_sz=None)
        test_dl = DataLoader(test_ds, batch_size=1, sampler=SequentialSampler(test_ds))

    return DataLoaders(train=train_dl, valid=valid_dl, test=test_dl)
