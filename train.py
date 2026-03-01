"""Train a General Attitude Estimator using tsfast."""

from tsfast.models.rnn import RNNLearner
from tsfast.models.scaling import StandardScaler
from tsfast.training import ignore_nan
from tsfast.training.transforms import bias, noise_grouped
from tsfast.quaternions import (
    QuaternionAugmentation,
    abs_inclination,
    augmentation_groups,
    mean_inclination_deg,
    rms_inclination_deg,
)

from riann.data import get_dls

# ── Data ──────────────────────────────────────────────────────────────────────

dls = get_dls()

# ── Model + Training ─────────────────────────────────────────────────────────

lrn = RNNLearner(
    dls,
    loss_func=ignore_nan(abs_inclination),
    metrics=[ignore_nan(rms_inclination_deg), ignore_nan(mean_inclination_deg)],
    num_layers=2,
    hidden_size=200,
    rnn_type="gru",
    normalization="layernorm",
    sub_seq_len=1800,
    input_norm=StandardScaler,
    augmentations=[
        QuaternionAugmentation(augmentation_groups([3, 3])),
        bias(std=[0, 0, 0, 0.005, 0.005, 0.005, 0], p=0.5),
        noise_grouped(std_std=[0.025, 0.01, 0.0], std_idx=[0, 0, 0, 1, 1, 1, 2], p=0.4),
    ],
    grad_clip=0.01,
    return_state=True,
)

lrn.fit_flat_cos(n_epoch=200, lr=3e-3, pct_start=0.75)
