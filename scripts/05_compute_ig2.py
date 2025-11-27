"""
Compute IG2 attribution scores for SAE features.

This script identifies which SAE features most strongly influence bias predictions
using Integrated Gradients (IG2) attribution.

This is a NEW component compared to korean-sparse-llm-features-open, implementing
the core research contribution of applying IG2 to SAE features.

IMPORTANT: This script must use the probe trained for the specific demographic.
When using --demographic, it loads the per-demographic probe and activations.
"""

import sys
import pickle
import argparse
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.sae import GatedAutoEncoder, AutoEncoder
from src.models import BiasProbe
from src.attribution import compute_ig2_for_sae_features, identify_bias_features
from src.utils import load_config
from src.utils.demographic_utils import get_demographic_info, get_all_demographics
from src.interfaces import IG2Result


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


def main(args):
    # Load config
    config = load_config('configs/experiment_config.yaml')

    # Determine which demographic to use
    # Priority: command line arg > config file
    if args.demographic:
        demographic = args.demographic
    else:
        demographic = config['data']['demographic']

    # Get devices from config (list of devices for multi-GPU support)
    devices = config['model'].get('devices', ['cuda' if torch.cuda.is_available() else 'cpu'])
    if not isinstance(devices, list):
        devices = [devices]
    device = torch.device(devices[0])

    # Get IG2 batch size from config
    ig2_batch_size = config.get('ig2', {}).get('batch_size', 16)

    print(f"Devices: {devices}")
    print(f"IG2 batch size: {ig2_batch_size}")

    # Get demographic info
    demo_info = get_demographic_info(demographic, data_dir='data')
    demographic_values = demo_info['values']

    print(f"=== Computing IG2 Attribution ===")
    print(f"Stage: {args.stage}")
    print(f"Demographic: {demographic} ({demo_info['dimension_en']})")
    print(f"SAE type: {args.sae_type}")
    print(f"Layer: {args.layer_quantile}")

    # Determine activation path
    results_dir = Path(config['paths']['results_dir'])

    if args.demographic:
        # Use per-demographic activation file
        activation_path = results_dir / args.stage / demographic / 'activations.pkl'
    else:
        # Use default location (may be merged or single-demographic)
        activation_path = results_dir / args.stage / 'activations.pkl'

    if not activation_path.exists():
        # Fallback: try per-demographic path if default doesn't exist
        fallback_path = results_dir / args.stage / demographic / 'activations.pkl'
        if fallback_path.exists():
            activation_path = fallback_path
            print(f"Using per-demographic activation file: {activation_path}")
        else:
            print(f"\nERROR: Activation file not found at {activation_path}")
            print(f"Also checked: {fallback_path}")
            print(f"Please run 02_generate_and_extract_activations.py first")
            return

    print(f"\nLoading activations from {activation_path}")
    with open(activation_path, 'rb') as f:
        activations_dict = pickle.load(f)

    # Check if this is a merged activation file (has demographics field)
    demographics_field = f'{args.stage}_demographics'
    is_merged = demographics_field in activations_dict

    # Get activations
    all_activations = activations_dict[f'{args.stage}_residual_{args.layer_quantile}']

    # If merged file, filter to only samples from the target demographic
    if is_merged:
        print(f"Detected merged activation file, filtering for demographic: {demographic}")
        sample_demographics = activations_dict[demographics_field]

        # Find indices for this demographic
        indices = [i for i, d in enumerate(sample_demographics) if d == demographic]

        if not indices:
            print(f"\nERROR: No samples found for demographic '{demographic}' in merged file")
            print(f"Available demographics: {set(sample_demographics)}")
            return

        # Filter activations
        activations = all_activations[indices]
        print(f"Filtered to {len(indices)} samples for {demographic}")
    else:
        activations = all_activations

    print(f"Loaded {len(activations)} samples")

    # Load SAE
    sae_path = Path(config['paths']['models_dir']) / f'sae-{args.sae_type}_{args.stage}_{args.layer_quantile}' / 'model.pth'

    if not sae_path.exists():
        print(f"\nERROR: SAE model not found")
        print(f"Please run 03_train_sae.py first")
        return

    print(f"\nLoading SAE from {sae_path}")
    if args.sae_type == 'standard':
        sae = AutoEncoder.from_pretrained(str(sae_path))
    elif args.sae_type == 'gated':
        sae = GatedAutoEncoder.from_pretrained(str(sae_path))
    else:
        raise ValueError(f"Unknown SAE type: {args.sae_type}")
    sae.to(device)

    # Extract SAE features
    print("\nExtracting SAE features...")
    sae_features = get_sae_features(sae, activations, device)

    # Clean up
    del sae
    torch.cuda.empty_cache()

    print(f"SAE features shape: {sae_features.shape}")

    # Load probe - use per-demographic directory when demographic is specified
    # Use layer-specific filename for multi-layer experiments
    probe_filename = f'{args.layer_quantile}_linear_probe.pt'
    if args.demographic:
        probe_path = results_dir / args.stage / demographic / 'probe' / probe_filename
    else:
        probe_path = results_dir / args.stage / 'probe' / probe_filename

    if not probe_path.exists():
        # Fallback: try legacy non-layer-specific path for backward compatibility
        legacy_filename = 'linear_probe.pt'
        if args.demographic:
            fallback_path = results_dir / args.stage / demographic / 'probe' / legacy_filename
        else:
            fallback_path = results_dir / args.stage / 'probe' / legacy_filename

        if fallback_path.exists():
            probe_path = fallback_path
            print(f"Using legacy probe path (no layer prefix): {probe_path}")
        else:
            print(f"\nERROR: Linear probe not found at {probe_path}")
            print(f"Also checked legacy path: {fallback_path}")
            print(f"Please run 04_train_linear_probe.py --demographic {demographic} --layer_quantile {args.layer_quantile} first")
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
    print(f"  Probe accuracy: {checkpoint.get('final_accuracy', 'N/A')}")

    # Compute IG2
    print(f"\n{'='*60}")
    print(f"Computing IG2 attribution scores...")
    print(f"{'='*60}")
    print(f"Number of steps: {args.num_steps}")
    print(f"Use squared gap: {args.use_squared_gap}")
    print(f"Batch size: {ig2_batch_size}")
    print(f"Multi-GPU: {len(devices) > 1} ({devices})")

    ig2_scores = compute_ig2_for_sae_features(
        sae_features=sae_features,
        linear_probe=probe,
        num_steps=args.num_steps,
        use_squared_gap=args.use_squared_gap,
        device=device,
        devices=devices,
        batch_size=ig2_batch_size
    )

    print(f"\nIG2 scores computed!")
    print(f"  Shape: {ig2_scores.shape}")
    print(f"  Range: [{ig2_scores.min():.6f}, {ig2_scores.max():.6f}]")
    print(f"  Mean: {ig2_scores.mean():.6f}")
    print(f"  Std: {ig2_scores.std():.6f}")

    # Identify bias features
    print(f"\n{'='*60}")
    print(f"Identifying bias features...")
    print(f"{'='*60}")
    print(f"Threshold ratio: {args.threshold_ratio}")

    bias_features, threshold = identify_bias_features(
        ig2_scores,
        threshold_ratio=args.threshold_ratio
    )

    print(f"\nIdentified {len(bias_features)} bias features")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Percentage: {len(bias_features)/len(ig2_scores)*100:.2f}%")

    print(f"\nTop 20 bias features:")
    for i, feat_idx in enumerate(bias_features[:20].tolist()):
        score = ig2_scores[feat_idx].item()
        print(f"  {i+1:2d}. Feature {feat_idx:6d}: {score:.6f}")

    # Save results - use per-demographic directory when demographic is specified
    if args.demographic:
        output_dir = results_dir / args.stage / demographic / 'ig2'
    else:
        output_dir = results_dir / args.stage / 'ig2'
    output_dir.mkdir(parents=True, exist_ok=True)

    result = IG2Result(
        feature_scores=ig2_scores.cpu(),
        bias_features=bias_features.cpu(),
        threshold=threshold,
        metadata={
            'num_samples': len(sae_features),
            'num_features': sae_features.shape[1],
            'demographic': demographic,
            'demographic_values': [v.strip() for v in demographic_values],
            'sae_type': args.sae_type,
            'layer_quantile': args.layer_quantile,
            'num_steps': args.num_steps,
            'use_squared_gap': args.use_squared_gap,
            'threshold_ratio': args.threshold_ratio,
            'num_bias_features': len(bias_features),
            'probe_accuracy': checkpoint.get('final_accuracy', None),
        }
    )

    # Use layer-specific filename for multi-layer experiments
    result_filename = f'{args.layer_quantile}_ig2_results.pt'
    result_path = output_dir / result_filename
    result.save(result_path)

    print(f"\nIG2 results saved to: {result_path}")

    # Save feature importance ranking with layer-specific filename
    feature_ranking = torch.argsort(ig2_scores, descending=True)
    ranking_filename = f'{args.layer_quantile}_feature_ranking.pt'
    ranking_path = output_dir / ranking_filename
    torch.save({
        'ranking': feature_ranking,
        'scores': ig2_scores[feature_ranking],
        'layer_quantile': args.layer_quantile,
    }, ranking_path)

    print(f"Feature ranking saved to: {ranking_path}")

    # Summary statistics
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total SAE features: {len(ig2_scores)}")
    print(f"Bias features identified: {len(bias_features)}")
    print(f"Threshold: {threshold:.6f}")
    print(f"Top feature score: {ig2_scores.max():.6f}")
    print(f"Median score: {ig2_scores.median():.6f}")

    # Score distribution
    print(f"\nScore distribution:")
    percentiles = [50, 75, 90, 95, 99]
    for p in percentiles:
        val = torch.quantile(ig2_scores, p/100)
        print(f"  {p}th percentile: {val:.6f}")

    print(f"\nIG2 computation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute IG2 attribution scores for SAE features"
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
        '--num_steps',
        type=int,
        default=20,
        help='Number of integration steps for IG2'
    )
    parser.add_argument(
        '--use_squared_gap',
        action='store_true',
        default=False,
        help='Use squared gap instead of absolute difference'
    )
    parser.add_argument(
        '--threshold_ratio',
        type=float,
        default=0.2,
        help='Threshold ratio for identifying bias features (0.0-1.0)'
    )
    parser.add_argument(
        '--demographic',
        type=str,
        default=None,
        help='Demographic category to compute IG2 for (e.g., 성별, 인종). '
             'If not specified, uses config default. When specified, loads '
             'per-demographic probe and saves results to per-demographic directory.'
    )
    args = parser.parse_args()

    main(args)
