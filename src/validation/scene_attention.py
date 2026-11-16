"""Model-free helpers of the scene_attention (GNN_ATTN) exchange: scene groups and score inputs."""

import math

import numpy as np

from validation.model import nn


def scene_groups(offsets):
    """{"scene_group_<n>": [videos with n scenes, n] batch-local scene rows}; videos without scenes are absent."""
    offsets = np.asarray(offsets, dtype=np.int64)
    counts = np.diff(offsets)
    return {
        f"scene_group_{size}": offsets[:-1][counts == size, None] + np.arange(size)
        for size in np.unique(counts)
        if size > 0
    }


def unit_rms(values, eps=1e-8):
    """Unit L2 direction rescaled to unit RMS; only the attention scores see it.

    Raw scene features have norm ~1, which would give scores near 0 and almost uniform
    weights for many steps; values are never rescaled.
    """
    return nn.functional.normalize(values, dim=-1, eps=eps) * math.sqrt(values.shape[-1])
