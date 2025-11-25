"""
Visualization utilities for korean-bias-sae project.

This module provides tools for visualizing:
- UMAP-based feature space clustering
- IG² attribution rankings
- Feature activation heatmaps
- Verification effects (suppress/amplify)
- SAE training loss curves
"""

from .font_utils import setup_korean_font
from .data_loaders import (
    load_sae_features,
    load_ig2_results,
    load_verification_results,
    load_demographics,
    load_training_logs
)
from .umap_utils import apply_umap_to_features, select_top_features_union
from .feature_selection import select_top_k_per_demographic, compute_tfidf_weights
from .plotting_utils import (
    plot_umap_clusters,
    plot_ig2_rankings,
    plot_activation_heatmap,
    plot_verification_effects,
    plot_training_loss
)

__all__ = [
    'setup_korean_font',
    'load_sae_features',
    'load_ig2_results',
    'load_verification_results',
    'load_demographics',
    'load_training_logs',
    'apply_umap_to_features',
    'select_top_features_union',
    'select_top_k_per_demographic',
    'compute_tfidf_weights',
    'plot_umap_clusters',
    'plot_ig2_rankings',
    'plot_activation_heatmap',
    'plot_verification_effects',
    'plot_training_loss',
]
