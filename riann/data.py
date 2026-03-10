"""Dataset splits and DataLoader creation for RIANN training."""

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, WeightedRandomSampler

from tsfast.tsdata import get_hdf_files
from tsfast.tsdata.dataset import FileEntry, WindowedDataset
from tsfast.tsdata.pipeline import DataLoaders
from tsfast.tsdata.readers import HDF5Signals, Resampled

u_dt = ["acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z", "dt"]
y = ["opt_a", "opt_b", "opt_c", "opt_d"]

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

MYON_VALID_IDS = {14, 39, 21}
MYON_TEST_IDS = {29, 22, 35}
TUMVI_TRAIN_ROOMS = {"room1", "room2", "room3"}
TEST_DATASETS = ["OxIOD", "EuRoC-MAV", "Caruso-Sassari", "RepoIMU", "Caruso-Sassari_orig"]


def _myon_id(f: Path) -> int:
    return int(f.name.split("_")[0])


def get_files(data_dir: Path | None = None) -> dict[str, list[Path]]:
    """Discover HDF5 files and split into train/valid/test.

    Train: Myon train subjects + TUM-VI rooms 1-3.
    Valid: Myon valid subjects + TUM-VI remaining rooms.
    Test:  All external datasets + Myon test subjects.
    """
    d = Path(data_dir) if data_dir else DATA_DIR

    f_myon = get_hdf_files(d / "Myon", recurse=False)
    f_tumvi = get_hdf_files(d / "TUM-VI", recurse=False)

    all_myon_split = MYON_VALID_IDS | MYON_TEST_IDS
    myon_train = [f for f in f_myon if _myon_id(f) not in all_myon_split]
    myon_valid = [f for f in f_myon if _myon_id(f) in MYON_VALID_IDS]
    myon_test = [f for f in f_myon if _myon_id(f) in MYON_TEST_IDS]

    tumvi_train = [f for f in f_tumvi if any(r in f.name for r in TUMVI_TRAIN_ROOMS)]
    tumvi_valid = [f for f in f_tumvi if not any(r in f.name for r in TUMVI_TRAIN_ROOMS)]

    test_external = [f for name in TEST_DATASETS for f in get_hdf_files(d / name, recurse=False)]

    return {
        "train": myon_train + tumvi_train,
        "valid": myon_valid + tumvi_valid,
        "test": myon_test + test_external,
    }


def _read_src_fs(path: Path) -> float:
    with h5py.File(path, "r") as f:
        return float(1.0 / f["dt"][0])


def get_dls(
    data_dir: Path | None = None,
    win_sz: int = 9000,
    stp_sz: int = 60,
    bs: int = 64,
    n_batches_train: int = 300,
    targ_fs: list[float] | None = None,
    targ_fs_count: int | None = None,
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
        targ_fs_count: number of equidistant frequencies between 50-500 Hz.
            Shorthand alternative to targ_fs.
    """
    if targ_fs is None:
        count = targ_fs_count if targ_fs_count is not None else 100
        targ_fs = np.linspace(50, 500, count).tolist()

    splits = get_files(data_dir)
    raw_inp, raw_tgt = HDF5Signals(u_dt), HDF5Signals(y)
    rs_inp, rs_tgt = Resampled(raw_inp, dt_idx=6), Resampled(raw_tgt)

    train_entries = [
        FileEntry(str(f), tf / _read_src_fs(f))
        for f in splits["train"] for tf in targ_fs
    ]
    valid_entries = [FileEntry(str(f)) for f in splits["valid"]]
    test_entries = [FileEntry(str(f)) for f in splits["test"]]

    train_ds = WindowedDataset(train_entries, rs_inp, rs_tgt, win_sz=win_sz, stp_sz=stp_sz)
    valid_ds = WindowedDataset(valid_entries, raw_inp, raw_tgt, win_sz=win_sz, stp_sz=win_sz)

    # Weighted sampling: equal probability per (file, target_frequency) entry
    counts = train_ds._counts.astype(np.float64)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    sample_weights = np.repeat(weights, train_ds._counts)
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights), num_samples=n_batches_train * bs, replacement=True,
    )

    pin = torch.cuda.is_available()
    train_dl = DataLoader(train_ds, batch_size=bs, drop_last=True, pin_memory=pin, sampler=sampler)
    valid_dl = DataLoader(valid_ds, batch_size=bs, drop_last=False, pin_memory=pin,
                          sampler=SequentialSampler(valid_ds))
    test_dl = None
    if test_entries:
        test_ds = WindowedDataset(test_entries, raw_inp, raw_tgt, win_sz=None)
        test_dl = DataLoader(test_ds, batch_size=1, sampler=SequentialSampler(test_ds))

    return DataLoaders(train=train_dl, valid=valid_dl, test=test_dl)
