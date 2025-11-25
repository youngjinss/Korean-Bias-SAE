"""
Integrated Gradients (IG²) for SAE features.

Adapted from Bias-Neurons paper to work with SAE learned features instead of raw neurons.
"""

import torch
import torch.nn as nn
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def compute_ig2_for_sae_features(
    sae_features: torch.Tensor,
    linear_probe: nn.Module,
    num_steps: int = 20,
    use_squared_gap: bool = False,
    device: str = "cuda"
) -> torch.Tensor:
    """
    Compute IG² attribution scores for SAE features following Bias-Neurons paper.

    This measures how much each SAE feature contributes to the logit gap
    between two demographic predictions (e.g., male vs female).

    Implementation follows the Bias-Neurons paper:
    1. Compute IG² for each demographic class separately
    2. Take the difference between demographics

    IG²_gap(x) = IG²(x, demo1) - IG²(x, demo2)
               = x * (∫∇f_demo1(αx)dα - ∫∇f_demo2(αx)dα)

    Args:
        sae_features: SAE feature activations (batch, feature_dim)
        linear_probe: Trained BiasProbe module
        num_steps: Number of integration steps (m in the paper)
        use_squared_gap: If True, use squared difference; if False, use absolute difference
        device: Device to compute on

    Returns:
        ig2_scores: Attribution score per feature (feature_dim,)
    """
    logger.debug(f"Computing IG² with {num_steps} steps")

    # Move to device
    sae_features = sae_features.to(device)
    linear_probe = linear_probe.to(device)
    linear_probe.eval()

    batch_size, feature_dim = sae_features.shape

    # Detach features
    sae_features = sae_features.detach().clone()

    # Baseline: zero features
    baseline = torch.zeros_like(sae_features)

    # Step size for integration
    step = (sae_features - baseline) / num_steps

    # Compute IG² for demographic 1 (class 0)
    ig2_demo1 = torch.zeros(feature_dim, device=device)

    for i in range(num_steps):
        # Scaled input from baseline to actual features
        scaled_features = (baseline + step * i).requires_grad_(True)

        # Forward pass
        logits = linear_probe(scaled_features)  # (batch, output_dim)

        # Get logits for demographic 1 (class 0)
        logits_demo1 = logits[:, 0]

        # Backward to get gradients
        gradients = torch.autograd.grad(
            outputs=logits_demo1.sum(),
            inputs=scaled_features,
            create_graph=False,
            retain_graph=False
        )[0]

        # Accumulate gradients (sum across batch)
        ig2_demo1 += gradients.sum(dim=0)

    # Multiply by step and average feature value
    ig2_demo1 = (sae_features.mean(dim=0) * ig2_demo1 / num_steps)

    # Compute IG² for demographic 2 (class 1)
    ig2_demo2 = torch.zeros(feature_dim, device=device)

    for i in range(num_steps):
        # Scaled input from baseline to actual features
        scaled_features = (baseline + step * i).requires_grad_(True)

        # Forward pass
        logits = linear_probe(scaled_features)  # (batch, output_dim)

        # Get logits for demographic 2 (class 1)
        logits_demo2 = logits[:, 1]

        # Backward to get gradients
        gradients = torch.autograd.grad(
            outputs=logits_demo2.sum(),
            inputs=scaled_features,
            create_graph=False,
            retain_graph=False
        )[0]

        # Accumulate gradients (sum across batch)
        ig2_demo2 += gradients.sum(dim=0)

    # Multiply by step and average feature value
    ig2_demo2 = (sae_features.mean(dim=0) * ig2_demo2 / num_steps)

    # Compute gap: IG²(demo1) - IG²(demo2)
    ig2_gap = ig2_demo1 - ig2_demo2

    # Take absolute value or square the gap
    if use_squared_gap:
        ig2_scores = ig2_gap ** 2
    else:
        ig2_scores = torch.abs(ig2_gap)

    logger.debug(f"IG² computation complete. Score range: [{ig2_scores.min():.4f}, {ig2_scores.max():.4f}]")

    return ig2_scores


def identify_bias_features(
    ig2_scores: torch.Tensor,
    threshold_ratio: float = 0.2,
    top_k: int = None
) -> Tuple[torch.Tensor, float]:
    """
    Identify bias features based on IG² scores.

    Args:
        ig2_scores: IG² attribution scores (feature_dim,)
        threshold_ratio: Threshold as ratio of max score (e.g., 0.2 = 20%)
        top_k: Alternatively, select top K features

    Returns:
        Tuple of (bias_feature_indices, threshold_used)
    """
    if top_k is not None:
        # Select top K features
        _, indices = torch.topk(ig2_scores, k=top_k)
        threshold = ig2_scores[indices[-1]].item()
        logger.info(f"Selected top-{top_k} features, threshold: {threshold:.6f}")
    else:
        # Use threshold ratio
        threshold = threshold_ratio * ig2_scores.max().item()
        indices = torch.where(ig2_scores >= threshold)[0]
        logger.info(f"Threshold (ratio={threshold_ratio}): {threshold:.6f}")
        logger.info(f"Number of bias features: {len(indices)}")

    return indices, threshold


def compute_logit_gap(
    sae_features: torch.Tensor,
    linear_probe: nn.Module,
    use_squared: bool = False
) -> torch.Tensor:
    """
    Compute logit gap for given SAE features.

    Args:
        sae_features: SAE features (batch, feature_dim)
        linear_probe: Trained probe
        use_squared: Use squared difference

    Returns:
        Logit gaps (batch,)
    """
    with torch.no_grad():
        logits = linear_probe(sae_features)
        if use_squared:
            gap = (logits[:, 0] - logits[:, 1]) ** 2
        else:
            gap = torch.abs(logits[:, 0] - logits[:, 1])

    return gap


def manipulate_features(
    sae_features: torch.Tensor,
    feature_indices: torch.Tensor,
    manipulation_type: str = "suppress"
) -> torch.Tensor:
    """
    Manipulate specific SAE features.

    Args:
        sae_features: Original SAE features (batch, feature_dim)
        feature_indices: Indices of features to manipulate
        manipulation_type: 'suppress' (set to 0), 'amplify' (multiply by 2)

    Returns:
        Manipulated SAE features
    """
    manipulated = sae_features.clone()

    if manipulation_type == "suppress":
        manipulated[:, feature_indices] = 0
    elif manipulation_type == "amplify":
        manipulated[:, feature_indices] *= 2
    else:
        raise ValueError(f"Invalid manipulation_type: {manipulation_type}")

    return manipulated
