"""
Verify bias features through suppression and amplification tests.

This script validates that identified SAE features causally influence bias predictions
by manipulating feature activations and measuring changes in bias scores.

This is adapted from korean-sparse-llm-features-open's verification approach,
but works with generation-based activations and multi-demographic support.
"""

import sys
import pickle
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.sae import GatedAutoEncoder, AutoEncoder
from src.models import BiasProbe
from src.attribution import manipulate_features, compute_logit_gap
from src.interfaces import IG2Result, VerificationResult
from src.utils import load_config, save_json
from src.utils.demographic_utils import get_demographic_info


def get_sae_features(sae, activations, device='cuda'):
    """
    Extract SAE features from activations.

    Args:
        sae: Trained SAE model
        activations: Hidden state activations
        device: Device to run on

    Returns:
        SAE features tensor
    """
    n_samples = len(activations)
    all_features = []

    sae.eval()
    print(f"Extracting SAE features for {n_samples} samples...")

    for i in tqdm(range(n_samples), desc="Extracting features"):
        sample = activations[i].unsqueeze(0).to(device)

        with torch.no_grad():
            _, features = sae(sample, output_features=True)

        # Sum over sequence dimension if needed
        if features.dim() > 2:
            features = torch.sum(features, dim=1)

        all_features.append(features[0])

    sae_features = torch.stack(all_features)
    return sae_features


def run_suppression_test(
    sae_features: torch.Tensor,
    probe: torch.nn.Module,
    bias_features: torch.Tensor,
    device: str
) -> VerificationResult:
    """
    Test if suppressing bias features reduces bias.

    Args:
        sae_features: Original SAE features (batch, feature_dim)
        probe: Trained linear probe
        bias_features: Indices of bias features to suppress
        device: Device to run on

    Returns:
        VerificationResult with gap changes
    """
    sae_features = sae_features.to(device)
    probe = probe.to(device)
    probe.eval()

    # Compute original gaps
    original_gaps = compute_logit_gap(sae_features, probe, use_squared=False)
    gap_before = original_gaps.mean().item()

    # Suppress bias features
    suppressed_features = manipulate_features(
        sae_features,
        bias_features,
        manipulation_type="suppress"
    )

    # Compute new gaps
    new_gaps = compute_logit_gap(suppressed_features, probe, use_squared=False)
    gap_after = new_gaps.mean().item()

    # Calculate change ratio
    if gap_before > 0:
        gap_change_ratio = (gap_after - gap_before) / gap_before
    else:
        gap_change_ratio = 0.0

    result = VerificationResult(
        gap_before=gap_before,
        gap_after=gap_after,
        gap_change_ratio=gap_change_ratio,
        feature_indices=bias_features.tolist(),
        manipulation_type="suppress",
        metadata={
            'num_features_manipulated': len(bias_features),
            'gap_std_before': original_gaps.std().item(),
            'gap_std_after': new_gaps.std().item(),
        }
    )

    return result


def run_amplification_test(
    sae_features: torch.Tensor,
    probe: torch.nn.Module,
    bias_features: torch.Tensor,
    device: str
) -> VerificationResult:
    """
    Test if amplifying bias features increases bias.

    Args:
        sae_features: Original SAE features (batch, feature_dim)
        probe: Trained linear probe
        bias_features: Indices of bias features to amplify
        device: Device to run on

    Returns:
        VerificationResult with gap changes
    """
    sae_features = sae_features.to(device)
    probe = probe.to(device)
    probe.eval()

    # Compute original gaps
    original_gaps = compute_logit_gap(sae_features, probe, use_squared=False)
    gap_before = original_gaps.mean().item()

    # Amplify bias features
    amplified_features = manipulate_features(
        sae_features,
        bias_features,
        manipulation_type="amplify"
    )

    # Compute new gaps
    new_gaps = compute_logit_gap(amplified_features, probe, use_squared=False)
    gap_after = new_gaps.mean().item()

    # Calculate change ratio
    if gap_before > 0:
        gap_change_ratio = (gap_after - gap_before) / gap_before
    else:
        gap_change_ratio = 0.0

    result = VerificationResult(
        gap_before=gap_before,
        gap_after=gap_after,
        gap_change_ratio=gap_change_ratio,
        feature_indices=bias_features.tolist(),
        manipulation_type="amplify",
        metadata={
            'num_features_manipulated': len(bias_features),
            'gap_std_before': original_gaps.std().item(),
            'gap_std_after': new_gaps.std().item(),
        }
    )

    return result


def run_random_control(
    sae_features: torch.Tensor,
    probe: torch.nn.Module,
    num_features: int,
    feature_dim: int,
    device: str,
    num_trials: int = 10
) -> dict:
    """
    Test random feature suppression as control.

    Args:
        sae_features: Original SAE features (batch, feature_dim)
        probe: Trained linear probe
        num_features: Number of random features to suppress (same as bias features)
        feature_dim: Total feature dimension
        device: Device to run on
        num_trials: Number of random trials

    Returns:
        Dictionary with mean and std of gap changes
    """
    sae_features = sae_features.to(device)
    probe = probe.to(device)
    probe.eval()

    # Compute original gaps
    original_gaps = compute_logit_gap(sae_features, probe, use_squared=False)
    gap_before = original_gaps.mean().item()

    gap_changes = []

    for _ in range(num_trials):
        # Sample random features
        random_indices = torch.randperm(feature_dim)[:num_features]

        # Suppress random features
        suppressed_features = manipulate_features(
            sae_features,
            random_indices,
            manipulation_type="suppress"
        )

        # Compute new gaps
        new_gaps = compute_logit_gap(suppressed_features, probe, use_squared=False)
        gap_after = new_gaps.mean().item()

        # Calculate change ratio
        if gap_before > 0:
            gap_change = (gap_after - gap_before) / gap_before
        else:
            gap_change = 0.0

        gap_changes.append(gap_change)

    return {
        'mean_gap_change': np.mean(gap_changes),
        'std_gap_change': np.std(gap_changes),
        'gap_changes': gap_changes,
        'num_trials': num_trials,
    }


def main(args):
    # Load config
    config = load_config('configs/experiment_config.yaml')
    demographic = config['data']['demographic']
    device = torch.device(config['model']['device'])

    # Get demographic info
    demo_info = get_demographic_info(demographic, data_dir='data')

    print(f"=== Verifying Bias Features ===")
    print(f"Stage: {args.stage}")
    print(f"Demographic: {demographic} ({demo_info['dimension_en']})")
    print(f"SAE type: {args.sae_type}")
    print(f"Layer: {args.layer_quantile}")

    # Load activations
    activation_path = Path(config['paths']['results_dir']) / args.stage / 'activations.pkl'

    if not activation_path.exists():
        print(f"\n❌ ERROR: Activation file not found")
        print(f"Please run 02_generate_and_extract_activations.py first")
        return

    print(f"\nLoading activations from {activation_path}")
    with open(activation_path, 'rb') as f:
        activations_dict = pickle.load(f)

    activations = activations_dict[f'{args.stage}_residual_{args.layer_quantile}']
    print(f"Loaded {len(activations)} samples")

    # Load SAE
    sae_path = Path(config['paths']['checkpoints_dir']) / f'sae-{args.sae_type}_{args.stage}_{args.layer_quantile}' / 'model.pth'

    if not sae_path.exists():
        print(f"\n❌ ERROR: SAE model not found")
        print(f"Please run 03_train_sae.py first")
        return

    print(f"\nLoading SAE from {sae_path}")
    if args.sae_type == 'standard':
        sae = AutoEncoder.from_pretrained(str(sae_path))
    elif args.sae_type == 'gated':
        sae = GatedAutoEncoder.from_pretrained(str(sae_path))
    sae.to(device)

    # Extract SAE features
    print("\nExtracting SAE features...")
    sae_features = get_sae_features(sae, activations, device)

    # Clean up
    del sae
    torch.cuda.empty_cache()

    print(f"SAE features shape: {sae_features.shape}")

    # Load probe
    probe_path = Path(config['paths']['results_dir']) / args.stage / 'probe' / 'linear_probe.pt'

    if not probe_path.exists():
        print(f"\n❌ ERROR: Linear probe not found")
        print(f"Please run 04_train_linear_probe.py first")
        return

    print(f"\nLoading probe from {probe_path}")
    checkpoint = torch.load(probe_path, map_location=device)

    probe = BiasProbe(
        input_dim=sae_features.shape[1],
        output_dim=10,
        hidden_dims=[]
    )
    probe.load_state_dict(checkpoint['probe_state_dict'])
    probe.to(device)
    probe.eval()

    print(f"Probe loaded successfully")

    # Load IG² results
    ig2_path = Path(config['paths']['results_dir']) / args.stage / 'ig2' / 'ig2_results.pt'

    if not ig2_path.exists():
        print(f"\n❌ ERROR: IG² results not found")
        print(f"Please run 05_compute_ig2.py first")
        return

    print(f"\nLoading IG² results from {ig2_path}")
    ig2_result = IG2Result.load(str(ig2_path))
    bias_features = ig2_result.bias_features.to(device)

    print(f"Loaded {len(bias_features)} bias features")
    print(f"  Threshold: {ig2_result.threshold:.6f}")
    print(f"  Top feature: {bias_features[0].item()} (score: {ig2_result.feature_scores[bias_features[0]]:.6f})")

    # Run verification tests
    print(f"\n{'='*60}")
    print(f"Running Verification Tests")
    print(f"{'='*60}\n")

    # 1. Suppression test
    print("1. Suppression Test (setting bias features to 0)")
    suppress_result = run_suppression_test(sae_features, probe, bias_features, device)

    print(f"  Gap before: {suppress_result.gap_before:.6f}")
    print(f"  Gap after:  {suppress_result.gap_after:.6f}")
    print(f"  Change:     {suppress_result.gap_change_ratio*100:+.2f}%")

    if suppress_result.gap_change_ratio < 0:
        print(f"  ✓ Suppression DECREASED bias (expected)")
    else:
        print(f"  ✗ Suppression INCREASED bias (unexpected)")

    # 2. Amplification test
    print("\n2. Amplification Test (multiplying bias features by 2)")
    amplify_result = run_amplification_test(sae_features, probe, bias_features, device)

    print(f"  Gap before: {amplify_result.gap_before:.6f}")
    print(f"  Gap after:  {amplify_result.gap_after:.6f}")
    print(f"  Change:     {amplify_result.gap_change_ratio*100:+.2f}%")

    if amplify_result.gap_change_ratio > 0:
        print(f"  ✓ Amplification INCREASED bias (expected)")
    else:
        print(f"  ✗ Amplification DECREASED bias (unexpected)")

    # 3. Random control
    print(f"\n3. Random Control (suppressing {len(bias_features)} random features)")
    print(f"   Running {args.num_random_trials} trials...")

    random_result = run_random_control(
        sae_features,
        probe,
        num_features=len(bias_features),
        feature_dim=sae_features.shape[1],
        device=device,
        num_trials=args.num_random_trials
    )

    print(f"  Mean change: {random_result['mean_gap_change']*100:+.2f}%")
    print(f"  Std change:  {random_result['std_gap_change']*100:.2f}%")

    # Compare with bias features
    suppress_change = suppress_result.gap_change_ratio
    random_mean = random_result['mean_gap_change']
    random_std = random_result['std_gap_change']

    # Z-score
    if random_std > 0:
        z_score = (suppress_change - random_mean) / random_std
        print(f"  Z-score: {z_score:.2f}")

        if abs(z_score) > 2:
            print(f"  ✓ Bias features have SIGNIFICANT effect (|z| > 2)")
        else:
            print(f"  ✗ Bias features effect not significant (|z| ≤ 2)")

    # Save results
    output_dir = Path(config['paths']['results_dir']) / args.stage / 'verification'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save suppression result
    suppress_path = output_dir / 'suppression_test.json'
    save_json(suppress_result.to_dict(), suppress_path)
    print(f"\n✓ Suppression results saved to: {suppress_path}")

    # Save amplification result
    amplify_path = output_dir / 'amplification_test.json'
    save_json(amplify_result.to_dict(), amplify_path)
    print(f"✓ Amplification results saved to: {amplify_path}")

    # Save random control result
    random_path = output_dir / 'random_control.json'
    save_json(random_result, random_path)
    print(f"✓ Random control results saved to: {random_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Bias features tested: {len(bias_features)}")
    print(f"Total feature dimension: {sae_features.shape[1]}")
    print(f"Percentage manipulated: {len(bias_features)/sae_features.shape[1]*100:.2f}%")
    print(f"\nSuppression effect: {suppress_result.gap_change_ratio*100:+.2f}%")
    print(f"Amplification effect: {amplify_result.gap_change_ratio*100:+.2f}%")
    print(f"Random control (mean): {random_result['mean_gap_change']*100:+.2f}%")

    # Validation criteria
    print(f"\nValidation:")
    criteria_met = 0
    total_criteria = 3

    # Criterion 1: Suppression reduces bias
    if suppress_result.gap_change_ratio < 0:
        print(f"  ✓ Suppression reduces bias gap")
        criteria_met += 1
    else:
        print(f"  ✗ Suppression does not reduce bias gap")

    # Criterion 2: Amplification increases bias
    if amplify_result.gap_change_ratio > 0:
        print(f"  ✓ Amplification increases bias gap")
        criteria_met += 1
    else:
        print(f"  ✗ Amplification does not increase bias gap")

    # Criterion 3: Effect is significant vs random
    if random_std > 0:
        z_score = (suppress_change - random_mean) / random_std
        if abs(z_score) > 2:
            print(f"  ✓ Effect is statistically significant (|z| = {abs(z_score):.2f} > 2)")
            criteria_met += 1
        else:
            print(f"  ✗ Effect is not statistically significant (|z| = {abs(z_score):.2f} ≤ 2)")

    print(f"\nCriteria met: {criteria_met}/{total_criteria}")

    if criteria_met == total_criteria:
        print(f"✓ All validation criteria passed!")
    else:
        print(f"⚠ Some validation criteria failed. Review bias features.")

    print(f"\n✓ Verification complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Verify bias features through suppression and amplification tests"
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='pilot',
        choices=['pilot', 'medium', 'full'],
        help='Experiment stage'
    )
    parser.add_argument(
        '--sae_type',
        type=str,
        default='gated',
        choices=['standard', 'gated'],
        help='Type of SAE used'
    )
    parser.add_argument(
        '--layer_quantile',
        type=str,
        default='q2',
        choices=['q1', 'q2', 'q3'],
        help='Which layer activations were used'
    )
    parser.add_argument(
        '--num_random_trials',
        type=int,
        default=10,
        help='Number of random control trials'
    )
    args = parser.parse_args()

    main(args)
