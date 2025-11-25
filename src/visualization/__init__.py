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
    load_training_logs,
    load_sae_decoder_weights,
    load_activations,
    get_demographic_labels,
    get_available_demographics
)
from .umap_utils import (
    apply_umap_to_features,
    select_top_features_union,
    prepare_umap_data,
    compute_feature_overlap,
    get_feature_frequency
)
from .feature_selection import select_top_k_per_demographic, compute_tfidf_weights
from .plotting_utils import (
    plot_umap_clusters,
    plot_ig2_rankings,
    plot_activation_heatmap,
    plot_verification_effects,
    plot_training_loss,
    plot_feature_frequency_histogram
)

__all__ = [
    # Font setup
    'setup_korean_font',
    # Data loaders
    'load_sae_features',
    'load_ig2_results',
    'load_verification_results',
    'load_demographics',
    'load_training_logs',
    'load_sae_decoder_weights',
    'load_activations',
    'get_demographic_labels',
    'get_available_demographics',
    # UMAP utilities
    'apply_umap_to_features',
    'select_top_features_union',
    'prepare_umap_data',
    'compute_feature_overlap',
    'get_feature_frequency',
    # Feature selection
    'select_top_k_per_demographic',
    'compute_tfidf_weights',
    # Plotting
    'plot_umap_clusters',
    'plot_ig2_rankings',
    'plot_activation_heatmap',
    'plot_verification_effects',
    'plot_training_loss',
    'plot_feature_frequency_histogram',
]
