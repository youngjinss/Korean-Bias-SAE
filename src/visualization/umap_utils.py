"""
UMAP dimensionality reduction utilities.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from umap import UMAP


def apply_umap_to_features(
    features: torch.Tensor,
    n_components: int = 2,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Apply UMAP dimensionality reduction to feature vectors.

    Args:
        features: Tensor of shape [N_features, feature_dim]
        n_components: Number of dimensions to reduce to
        random_state: Random seed for reproducibility
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        metric: Distance metric to use

    Returns:
        UMAP embeddings of shape [N_features, n_components]
    """
    # Convert to numpy if needed
    if isinstance(features, torch.Tensor):
        features = features.cpu().numpy()

    # Initialize UMAP
    umap_model = UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        verbose=False
    )

    # Fit and transform
    embeddings = umap_model.fit_transform(features)

    return embeddings


def select_top_features_union(
    ig2_results: Dict[str, torch.Tensor],
    top_k: int = 100
) -> Tuple[List[int], Dict[str, List[int]]]:
    """
    Select union of top-k features across all demographics.

    Args:
        ig2_results: Dictionary mapping demographic names to IG² scores
        top_k: Number of top features to select per demographic

    Returns:
        all_top_features: List of unique feature indices across all demographics
        demographic2topfeatures: Dictionary mapping demographics to their top-k indices
    """
    demographic2topfeatures = {}
    all_top_features = []

    for demographic, scores in ig2_results.items():
        # Get top-k indices
        if isinstance(scores, dict):
            scores = scores['feature_scores']

        top_indices = torch.topk(scores, k=min(top_k, len(scores))).indices
        top_indices_list = top_indices.tolist()

        demographic2topfeatures[demographic] = top_indices_list
        all_top_features.extend(top_indices_list)

    # Get unique features
    all_top_features = list(set(all_top_features))
    all_top_features.sort()

    return all_top_features, demographic2topfeatures


def create_color_mapping(
    all_features: List[int],
    selected_features: List[int],
    highlight_color: str = 'red',
    background_color: str = 'lightgray'
) -> List[str]:
    """
    Create color mapping for scatter plot.

    Args:
        all_features: List of all feature indices
        selected_features: List of features to highlight
        highlight_color: Color for selected features
        background_color: Color for non-selected features

    Returns:
        List of colors matching all_features order
    """
    selected_set = set(selected_features)
    colors = [highlight_color if fid in selected_set else background_color
              for fid in all_features]
    return colors


def prepare_umap_data(
    decoder_weights: torch.Tensor,
    ig2_results: Dict[str, torch.Tensor],
    top_k: int = 100
) -> Tuple[np.ndarray, List[int], Dict[str, List[int]]]:
    """
    Prepare data for UMAP visualization.

    Args:
        decoder_weights: Tensor of shape [100000, 4096]
        ig2_results: Dictionary of IG² scores per demographic
        top_k: Number of top features per demographic

    Returns:
        embeddings: UMAP 2D embeddings [N_selected, 2]
        all_features: List of selected feature indices
        demographic2topfeatures: Map of demographics to their top features
    """
    # Select union of top features
    all_features, demographic2topfeatures = select_top_features_union(
        ig2_results, top_k
    )

    print(f"Total features selected: {len(all_features)}")
    print(f"Features per demographic: {top_k}")

    # Extract selected features
    selected_weights = decoder_weights[all_features]

    # Apply UMAP
    embeddings = apply_umap_to_features(selected_weights)

    return embeddings, all_features, demographic2topfeatures


def compute_feature_overlap(
    demographic2topfeatures: Dict[str, List[int]]
) -> Dict[Tuple[str, str], int]:
    """
    Compute pairwise feature overlap between demographics.

    Args:
        demographic2topfeatures: Map of demographics to top features

    Returns:
        Dictionary mapping demographic pairs to overlap counts
    """
    demographics = list(demographic2topfeatures.keys())
    overlaps = {}

    for i, demo1 in enumerate(demographics):
        for demo2 in demographics[i+1:]:
            features1 = set(demographic2topfeatures[demo1])
            features2 = set(demographic2topfeatures[demo2])
            overlap = len(features1 & features2)
            overlaps[(demo1, demo2)] = overlap

    return overlaps


def get_feature_frequency(
    demographic2topfeatures: Dict[str, List[int]]
) -> Dict[int, int]:
    """
    Count how many demographics each feature appears in.

    Args:
        demographic2topfeatures: Map of demographics to top features

    Returns:
        Dictionary mapping feature indices to frequency counts
    """
    feature_freq = {}

    for features in demographic2topfeatures.values():
        for fid in features:
            feature_freq[fid] = feature_freq.get(fid, 0) + 1

    return feature_freq
