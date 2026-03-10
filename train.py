"""Train RIANN with best hyperparameters from paper (experiment 38, trial 79f72714)."""

from functools import partial

import numpy as np
import pytorch_optimizer

from tsfast.training.learners import RNNLearner
from tsfast.models.scaling import ScaledModel, StandardScaler
from tsfast.training.transforms import bias, noise_grouped
from tsfast.training import ignore_nan, nan_mean
from tsfast.quaternions import (
    QuaternionAugmentation,
    abs_inclination,
    augmentation_groups,
    inclination_angle,
    mean_inclination_deg,
    rms_inclination_deg,
)

from riann.data import get_dls

# ── Fixed input normalization (precomputed from training data) ───────────────
QUAT_IDENTITY = [1.0, 0.0, 0.0, 0.0]
DLS_MEAN = np.array([-0.6845, -0.4654, 0.2514, -0.0715, 0.0343, 0.1019, 0.0035])
DLS_STD = np.array([7.1055, 7.0015, 7.1748, 2.0147, 1.9856, 2.0225, 1.8847e-03])

# ── Data ──────────────────────────────────────────────────────────────────────

dls = get_dls(n_batches_train=60, targ_fs_count=427)

# ── Model + Training ─────────────────────────────────────────────────────────

lrn = RNNLearner(
    dls,
    loss_func= nan_mean(inclination_angle,QUAT_IDENTITY),
    metrics=[ignore_nan(mean_inclination_deg),
        ignore_nan(rms_inclination_deg)
    ],
    num_layers=2,
    hidden_size=200,
    n_skip=0,
    sub_seq_len=1800,
    rnn_type="gru",
    normalization="",
    input_norm=None,
    opt_func=partial(pytorch_optimizer.Ranger, betas=(0.95, 0.99), eps=1e-6, weight_decay=0.01, use_gc=False),
    # Augmentation parameters from experiment 38 were tuned for post-normalization
    # application (seqdata applied noise/bias after batch normalization). Here,
    # augmentations run pre-normalization (before ScaledModel), so we scale each
    # std by DLS_STD to preserve effective strength: σ_pre = σ_post × DLS_STD.
    augmentations=[
        QuaternionAugmentation(augmentation_groups([3, 3])),
        bias(std=[0, 0, 0, 0.006556, 0.006461, 0.006581, 0], p=0.1067),
        noise_grouped(std_std=[0.6410, 0.00712, 0.0], std_idx=[0, 0, 0, 1, 1, 1, 2], p=0.6123),
    ],
    grad_clip=1.0,
    cuda_graph=True,
)

# Wrap with fixed normalization (bypasses data-dependent StandardScaler)
lrn.model = ScaledModel(lrn.model, input_norm=StandardScaler(DLS_MEAN, DLS_STD))

lrn.fit_flat_cos(n_epoch=512, lr=3e-3, pct_start=0.15)
