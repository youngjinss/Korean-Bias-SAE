"""
Verification methods for bias features (suppression/amplification tests).
"""

import torch
import numpy as np
from typing import List, Dict
import logging
from ..interfaces import VerificationResult
from ..attribution.ig2_sae import compute_logit_gap, manipulate_features

logger = logging.getLogger(__name__)


def verify_bias_features(
    sae_features: torch.Tensor,
    bias_feature_indices: torch.Tensor,
    linear_probe,
    num_random_controls: int = 3,
    device: str = "cuda"
) -> Dict[str, VerificationResult]:
    """
    Verify that identified bias features actually contribute to bias.

    Performs three types of tests:
    1. Suppress: Set bias features to 0
    2. Amplify: Double bias features
    3. Random control: Apply same operations to random features

    Args:
        sae_features: Original SAE features (batch, feature_dim)
        bias_feature_indices: Indices of identified bias features
        linear_probe: Trained BiasProbe
        num_random_controls: Number of random control experiments
        device: Device to compute on

    Returns:
        Dictionary of verification results
    """
    logger.info(f"Verifying {len(bias_feature_indices)} bias features...")

    sae_features = sae_features.to(device)
    linear_probe = linear_probe.to(device)

    # Compute original logit gaps
    original_gaps = compute_logit_gap(sae_features, linear_probe, use_squared=True)
    mean_original_gap = original_gaps.mean().item()

    logger.info(f"Original mean logit gap: {mean_original_gap:.4f}")

    results = {}

    # 1. Suppress bias features
    logger.info("Testing suppression...")
    suppressed_features = manipulate_features(
        sae_features,
        bias_feature_indices,
        manipulation_type="suppress"
    )
    suppressed_gaps = compute_logit_gap(suppressed_features, linear_probe, use_squared=True)
    mean_suppressed_gap = suppressed_gaps.mean().item()
    suppress_change_ratio = (mean_suppressed_gap - mean_original_gap) / mean_original_gap * 100

    results['suppress'] = VerificationResult(
        gap_before=mean_original_gap,
        gap_after=mean_suppressed_gap,
        gap_change_ratio=suppress_change_ratio,
        feature_indices=bias_feature_indices.cpu().tolist(),
        manipulation_type='suppress',
        metadata={'expected': 'decrease'}
    )

    logger.info(f"  Suppress: {mean_original_gap:.4f} -> {mean_suppressed_gap:.4f} "
                f"({suppress_change_ratio:+.2f}%)")

    # 2. Amplify bias features
    logger.info("Testing amplification...")
    amplified_features = manipulate_features(
        sae_features,
        bias_feature_indices,
        manipulation_type="amplify"
    )
    amplified_gaps = compute_logit_gap(amplified_features, linear_probe, use_squared=True)
    mean_amplified_gap = amplified_gaps.mean().item()
    amplify_change_ratio = (mean_amplified_gap - mean_original_gap) / mean_original_gap * 100

    results['amplify'] = VerificationResult(
        gap_before=mean_original_gap,
        gap_after=mean_amplified_gap,
        gap_change_ratio=amplify_change_ratio,
        feature_indices=bias_feature_indices.cpu().tolist(),
        manipulation_type='amplify',
        metadata={'expected': 'increase'}
    )

    logger.info(f"  Amplify: {mean_original_gap:.4f} -> {mean_amplified_gap:.4f} "
                f"({amplify_change_ratio:+.2f}%)")

    # 3. Random controls
    logger.info(f"Testing {num_random_controls} random controls...")
    random_results = []
    feature_dim = sae_features.shape[1]
    num_bias_features = len(bias_feature_indices)

    for i in range(num_random_controls):
        # Sample random features (same number as bias features)
        random_indices = torch.randperm(feature_dim)[:num_bias_features]

        # Suppress random features
        random_suppressed = manipulate_features(
            sae_features,
            random_indices,
            manipulation_type="suppress"
        )
        random_gaps = compute_logit_gap(random_suppressed, linear_probe, use_squared=True)
        mean_random_gap = random_gaps.mean().item()
        random_change_ratio = (mean_random_gap - mean_original_gap) / mean_original_gap * 100

        random_result = VerificationResult(
            gap_before=mean_original_gap,
            gap_after=mean_random_gap,
            gap_change_ratio=random_change_ratio,
            feature_indices=random_indices.cpu().tolist(),
            manipulation_type=f'random_suppress_{i}',
            metadata={'expected': 'minimal change'}
        )
        random_results.append(random_result)

        logger.info(f"  Random {i+1}: {mean_original_gap:.4f} -> {mean_random_gap:.4f} "
                    f"({random_change_ratio:+.2f}%)")

    # Aggregate random control results
    random_change_ratios = [r.gap_change_ratio for r in random_results]
    results['random_control'] = {
        'individual_results': random_results,
        'mean_change_ratio': np.mean(random_change_ratios),
        'std_change_ratio': np.std(random_change_ratios),
    }

    # Summary
    logger.info("\nVerification Summary:")
    logger.info(f"  Bias features suppress: {suppress_change_ratio:+.2f}%")
    logger.info(f"  Bias features amplify: {amplify_change_ratio:+.2f}%")
    logger.info(f"  Random features: {np.mean(random_change_ratios):+.2f}% ± {np.std(random_change_ratios):.2f}%")

    # Check if results are as expected
    suppress_works = suppress_change_ratio < -5  # At least 5% decrease
    amplify_works = amplify_change_ratio > 5  # At least 5% increase
    random_minimal = abs(np.mean(random_change_ratios)) < 5  # Less than 5% change

    if suppress_works and amplify_works and random_minimal:
        logger.info("✅ Verification PASSED: Bias features show expected behavior")
    else:
        logger.warning("⚠️  Verification concerns:")
        if not suppress_works:
            logger.warning("  - Suppression did not reduce gap significantly")
        if not amplify_works:
            logger.warning("  - Amplification did not increase gap significantly")
        if not random_minimal:
            logger.warning("  - Random features show unexpected large changes")

    return results
