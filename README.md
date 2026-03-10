# RIANN Training Code

This repository contains the reimplemented training pipeline for [RIANN](https://www.mdpi.com/2673-2688/2/3/28) (Robust IMU-based Attitude Neural Network), a GRU-based model for real-time orientation estimation from inertial measurement unit (IMU) data.

The pretrained RIANN model has been publicly available for years at [daniel-om-weber/riann](https://github.com/daniel-om-weber/riann) (`pip install riann`), but the original training code was never published — it was lost in a hacking attack shortly after the paper's release. This repository provides a full reimplementation of the training pipeline, including data preparation, training, and evaluation.

## Installation

Requires Python >= 3.10 and [uv](https://docs.astral.sh/uv/).

```bash
# Clone (includes prepared datasets via Git LFS)
git lfs install
git clone https://github.com/daniel-om-weber/riann_dev.git
cd riann_dev

# Install dependencies
uv sync

# Also install data preparation dependencies (optional)
uv sync --extra prep
```

## Data

Prepared HDF5 datasets are included in the repository via Git LFS. They are ready to use after cloning.

| Dataset | Role | Files | Description |
|---------|------|-------|-------------|
| Myon (BROAD) | Train / Valid / Test | 39 | IMU benchmark with optical motion capture ground truth |
| TUM-VI | Train / Valid | 6 | Visual-inertial dataset (indoor rooms) |
| OxIOD | Test | 71 | Oxford Inertial Odometry Dataset |
| EuRoC-MAV | Test | 6 | Micro aerial vehicle dataset |
| RepoIMU | Test | 21 | TStick IMU recordings |
| Caruso-Sassari | Test | 18 | Multi-IMU dataset |

To regenerate from source (downloads from public URLs):

```bash
uv run python scripts/prepare_data.py              # all datasets
uv run python scripts/prepare_data.py --list        # list available datasets
uv run python scripts/prepare_data.py --dataset broad euroc  # specific datasets
uv run python scripts/prepare_data.py --keep-raw    # keep raw downloads
```

## Training

```bash
uv run python train.py
```

Trains a 2-layer GRU with on-the-fly resampling across 50–500 Hz. See `train.py` for the full configuration.

## Utility Scripts

- `scripts/prepare_data.py` — download raw IMU datasets and convert to standardized HDF5
- `scripts/validate_vqf.py` — validate prepared data by comparing VQF orientation estimates against optical ground truth
- `scripts/compare_hdf5.py` — compare newly generated HDF5 files against reference data

## HDF5 Format

Each dataset file contains 1D float32 arrays:

| Channel | Unit | Description |
|---------|------|-------------|
| `acc_x`, `acc_y`, `acc_z` | m/s² | Accelerometer |
| `gyr_x`, `gyr_y`, `gyr_z` | rad/s | Gyroscope |
| `opt_a`, `opt_b`, `opt_c`, `opt_d` | — | Ground truth quaternion (w, x, y, z) |
| `dt` | s | Sampling interval (scalar) |
| `mag_x`, `mag_y`, `mag_z` | — | Magnetometer (optional) |
| `movement_mask` | — | Binary activity indicator (optional) |

## Citation

```bibtex
@article{Weber2021RIANN,
  title     = {{RIANN---A Robust Neural Network Outperforms Attitude Estimation Filters}},
  author    = {Weber, Daniel and G{\"u}hmann, Clemens and Seel, Thomas},
  journal   = {AI},
  volume    = {2},
  number    = {3},
  pages     = {444--463},
  year      = {2021},
  publisher = {MDPI},
  doi       = {10.3390/ai2030028}
}
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
