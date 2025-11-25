"""Attribution methods for bias feature detection."""

from .ig2_sae import (
    compute_ig2_for_sae_features,
    identify_bias_features,
    manipulate_features,
    compute_logit_gap
)

__all__ = [
    'compute_ig2_for_sae_features',
    'identify_bias_features',
    'manipulate_features',
    'compute_logit_gap',
]
