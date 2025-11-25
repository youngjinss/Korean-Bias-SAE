"""
Feature selection utilities for visualization.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def select_top_k_per_demographic(
    ig2_results: Dict[str, torch.Tensor],
    top_k: int = 100
) -> Dict[str, torch.Tensor]:
    """
    Select top-k features for each demographic.

    Args:
        ig2_results: Dictionary of IG² scores per demographic
        top_k: Number of top features to select

    Returns:
        Dictionary mapping demographics to top-k feature indices
    """
    top_features = {}

    for demographic, scores in ig2_results.items():
        if isinstance(scores, dict):
            scores = scores['feature_scores']

        # Get top-k indices
        k = min(top_k, len(scores))
        top_indices = torch.topk(scores, k=k).indices
        top_features[demographic] = top_indices

    return top_features


def compute_tfidf_weights(
    feature_activations: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute TF-IDF weights for feature activations.

    Args:
        feature_activations: Tensor of shape [N_samples, N_features]
        eps: Small constant for numerical stability

    Returns:
        TF-IDF weighted activations of same shape
    """
    n_samples = len(feature_activations)

    # Document frequency: how many samples activate each feature
    df = (feature_activations > 0).float().sum(dim=0)

    # Inverse document frequency
    idf = torch.log(n_samples / (df + 1))

    # TF-IDF
    tfidf = feature_activations * idf.unsqueeze(0)

    # L2 normalization
    tfidf_normalized = tfidf / (torch.norm(tfidf, dim=1, keepdim=True) + eps)

    return tfidf_normalized


def group_features_by_demographic(
    prompts: List[Dict],
    feature_activations: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Group feature activations by demographic dimension.

    Args:
        prompts: List of BiasPrompt dictionaries
        feature_activations: Tensor of shape [N_prompts, N_features]

    Returns:
        Dictionary mapping demographics to mean activations
    """
    demographic_groups = {}

    for i, prompt in enumerate(prompts):
        demographic = prompt['demographic_dimension']

        if demographic not in demographic_groups:
            demographic_groups[demographic] = []

        demographic_groups[demographic].append(i)

    # Compute mean activations per demographic
    demographic_means = {}
    for demographic, indices in demographic_groups.items():
        demographic_means[demographic] = feature_activations[indices].mean(dim=0)

    return demographic_means


def rank_features_by_score(
    scores: torch.Tensor,
    top_k: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank features by their scores.

    Args:
        scores: Tensor of feature scores
        top_k: If specified, return only top-k

    Returns:
        sorted_scores: Scores in descending order
        sorted_indices: Corresponding feature indices
    """
    if top_k is None:
        top_k = len(scores)

    sorted_scores, sorted_indices = torch.sort(scores, descending=True)

    return sorted_scores[:top_k], sorted_indices[:top_k]


def select_features_by_threshold(
    scores: torch.Tensor,
    threshold: float
) -> torch.Tensor:
    """
    Select features with scores above threshold.

    Args:
        scores: Tensor of feature scores
        threshold: Minimum score threshold

    Returns:
        Indices of features above threshold
    """
    indices = torch.where(scores > threshold)[0]
    return indices


def compute_feature_sparsity(
    feature_activations: torch.Tensor
) -> Tuple[float, torch.Tensor]:
    """
    Compute sparsity statistics for features.

    Args:
        feature_activations: Tensor of shape [N_samples, N_features]

    Returns:
        overall_sparsity: Fraction of zero activations
        per_feature_sparsity: Sparsity per feature
    """
    # Overall sparsity
    overall_sparsity = (feature_activations == 0).float().mean().item()

    # Per-feature sparsity
    per_feature_sparsity = (feature_activations == 0).float().mean(dim=0)

    return overall_sparsity, per_feature_sparsity


def get_shared_features(
    demographic2features: Dict[str, List[int]],
    min_demographics: int = 2
) -> List[int]:
    """
    Find features shared across multiple demographics.

    Args:
        demographic2features: Map of demographics to feature lists
        min_demographics: Minimum number of demographics to appear in

    Returns:
        List of shared feature indices
    """
    from collections import Counter

    # Count feature occurrences
    feature_counts = Counter()
    for features in demographic2features.values():
        feature_counts.update(features)

    # Filter by minimum count
    shared_features = [fid for fid, count in feature_counts.items()
                      if count >= min_demographics]

    return sorted(shared_features)


def get_unique_features(
    demographic2features: Dict[str, List[int]]
) -> Dict[str, List[int]]:
    """
    Find features unique to each demographic.

    Args:
        demographic2features: Map of demographics to feature lists

    Returns:
        Dictionary mapping demographics to their unique features
    """
    # Collect all features from other demographics
    unique_features = {}

    for target_demo, target_features in demographic2features.items():
        target_set = set(target_features)

        # Collect features from all other demographics
        other_features = set()
        for demo, features in demographic2features.items():
            if demo != target_demo:
                other_features.update(features)

        # Find unique features
        unique = target_set - other_features
        unique_features[target_demo] = sorted(list(unique))

    return unique_features


def filter_features_by_activation_strength(
    feature_activations: torch.Tensor,
    min_activation: float = 0.01,
    min_frequency: float = 0.05
) -> torch.Tensor:
    """
    Filter features by minimum activation strength and frequency.

    Args:
        feature_activations: Tensor of shape [N_samples, N_features]
        min_activation: Minimum mean activation value
        min_frequency: Minimum fraction of samples with non-zero activation

    Returns:
        Indices of features passing filters
    """
    # Mean activation strength
    mean_activation = feature_activations.mean(dim=0)

    # Activation frequency
    activation_freq = (feature_activations > 0).float().mean(dim=0)

    # Apply both filters
    valid_features = torch.where(
        (mean_activation >= min_activation) & (activation_freq >= min_frequency)
    )[0]

    return valid_features
