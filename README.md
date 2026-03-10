# RIANN

Recurrent Inertial Attitude Neural Network — a GRU-based deep learning model for real-time orientation estimation from inertial measurement unit (IMU) data.

RIANN takes raw accelerometer and gyroscope readings and outputs orientation quaternions. It is trained on 6 standardized datasets using the [tsfast](https://pypi.org/project/tsfast/) library and supports on-the-fly resampling across a wide range of sampling frequencies (50–500 Hz).

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
| Myon (BROAD) | Train / Valid / Test | 39 | Human motion capture from Myon exoskeleton |
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

Trains a 2-layer GRU (hidden size 200) with the Ranger optimizer for 512 epochs. Uses inclination angle as the loss function and reports mean/RMS inclination error in degrees. CUDA graphs are enabled when a GPU is available.

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

## License

Apache-2.0 — see [LICENSE](LICENSE).
