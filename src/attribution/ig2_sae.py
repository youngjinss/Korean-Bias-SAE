"""
Integrated Gradients (IG²) for SAE features.

Adapted from Bias-Neurons paper to work with SAE learned features instead of raw neurons.

Supports:
- Batch processing to handle large datasets without OOM
- Multi-GPU via DataParallel for faster computation
"""

import torch
import torch.nn as nn
from typing import Tuple, List, Union, Optional
import logging

logger = logging.getLogger(__name__)


def compute_ig2_for_sae_features(
    sae_features: torch.Tensor,
    linear_probe: nn.Module,
    num_steps: int = 20,
    use_squared_gap: bool = False,
    device: Union[str, torch.device] = "cuda",
    devices: Optional[List[str]] = None,
    batch_size: int = 16
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

    Supports:
    - Batch processing: Process samples in smaller batches to avoid OOM
    - Multi-GPU: Use DataParallel when multiple devices are provided

    Args:
        sae_features: SAE feature activations (num_samples, feature_dim)
        linear_probe: Trained BiasProbe module
        num_steps: Number of integration steps (m in the paper)
        use_squared_gap: If True, use squared difference; if False, use absolute difference
        device: Primary device to compute on (used if devices is None)
        devices: List of devices for multi-GPU (e.g., ["cuda:2", "cuda:3"])
        batch_size: Number of samples to process at once (default: 16)

    Returns:
        ig2_scores: Attribution score per feature (feature_dim,)
    """
    # Determine devices to use
    if devices is not None and len(devices) > 1:
        primary_device = torch.device(devices[0])
        use_multi_gpu = True
        device_ids = [int(d.split(':')[1]) for d in devices]
        logger.info(f"Multi-GPU enabled for IG²: {devices} (device_ids: {device_ids})")
    else:
        primary_device = torch.device(device if devices is None else devices[0])
        use_multi_gpu = False
        device_ids = None

    logger.info(f"Computing IG² with {num_steps} steps, batch_size={batch_size}")

    num_samples, feature_dim = sae_features.shape

    # Prepare probe for computation
    linear_probe = linear_probe.to(primary_device)
    linear_probe.eval()

    # Wrap with DataParallel if multi-GPU
    if use_multi_gpu:
        probe_parallel = nn.DataParallel(linear_probe, device_ids=device_ids)
        logger.info(f"Wrapped probe with DataParallel on devices: {device_ids}")
    else:
        probe_parallel = linear_probe

    # Accumulate IG² scores across all batches
    ig2_demo1_total = torch.zeros(feature_dim, device=primary_device)
    ig2_demo2_total = torch.zeros(feature_dim, device=primary_device)
    feature_sum = torch.zeros(feature_dim, device=primary_device)

    # Process in batches
    num_batches = (num_samples + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        # Get batch
        batch_features = sae_features[start_idx:end_idx].detach().clone().to(primary_device)
        current_batch_size = batch_features.shape[0]

        logger.debug(f"Processing batch {batch_idx + 1}/{num_batches} (samples {start_idx}-{end_idx})")

        # Accumulate feature values for final averaging
        feature_sum += batch_features.sum(dim=0)

        # Baseline: zero features
        baseline = torch.zeros_like(batch_features)

        # Step size for integration
        step = (batch_features - baseline) / num_steps

        # Compute IG² for demographic 1 (class 0)
        for i in range(num_steps):
            scaled_features = (baseline + step * i).requires_grad_(True)

            # Forward pass (uses DataParallel if multi-GPU)
            logits = probe_parallel(scaled_features)
            logits_demo1 = logits[:, 0]

            # Backward to get gradients
            gradients = torch.autograd.grad(
                outputs=logits_demo1.sum(),
                inputs=scaled_features,
                create_graph=False,
                retain_graph=False
            )[0]

            # Accumulate gradients
            ig2_demo1_total += gradients.sum(dim=0)

        # Compute IG² for demographic 2 (class 1)
        for i in range(num_steps):
            scaled_features = (baseline + step * i).requires_grad_(True)

            # Forward pass
            logits = probe_parallel(scaled_features)
            logits_demo2 = logits[:, 1]

            # Backward to get gradients
            gradients = torch.autograd.grad(
                outputs=logits_demo2.sum(),
                inputs=scaled_features,
                create_graph=False,
                retain_graph=False
            )[0]

            # Accumulate gradients
            ig2_demo2_total += gradients.sum(dim=0)

        # Clear cache after each batch to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Compute final IG² scores
    # Average feature value across all samples
    feature_mean = feature_sum / num_samples

    # Normalize accumulated gradients by number of samples and steps
    ig2_demo1 = feature_mean * ig2_demo1_total / (num_samples * num_steps)
    ig2_demo2 = feature_mean * ig2_demo2_total / (num_samples * num_steps)

    # Compute gap: IG²(demo1) - IG²(demo2)
    ig2_gap = ig2_demo1 - ig2_demo2

    # Take absolute value or square the gap
    if use_squared_gap:
        ig2_scores = ig2_gap ** 2
    else:
        ig2_scores = torch.abs(ig2_gap)

    logger.info(f"IG² computation complete. Score range: [{ig2_scores.min():.4f}, {ig2_scores.max():.4f}]")

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
