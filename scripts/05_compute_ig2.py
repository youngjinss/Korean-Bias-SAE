"""
Compute IG2 attribution scores for SAE features.

This script identifies which SAE features most strongly influence bias predictions
using Integrated Gradients (IG2) attribution.

This is a NEW component compared to korean-sparse-llm-features-open, implementing
the core research contribution of applying IG2 to SAE features.
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
from src.utils.demographic_utils import get_demographic_info
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
    demographic = config['data']['demographic']
    device = torch.device(config['model']['device'])

    # Get demographic info
    demo_info = get_demographic_info(demographic, data_dir='data')
    demographic_values = demo_info['values']

    print(f"=== Computing IG2 Attribution ===")
    print(f"Stage: {args.stage}")
    print(f"Demographic: {demographic} ({demo_info['dimension_en']})")
    print(f"SAE type: {args.sae_type}")
    print(f"Layer: {args.layer_quantile}")

    # Load activations
    activation_path = Path(config['paths']['results_dir']) / args.stage / 'activations.pkl'

    if not activation_path.exists():
        print(f"\nERROR: Activation file not found")
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

    # Load probe
    probe_path = Path(config['paths']['results_dir']) / args.stage / 'probe' / 'linear_probe.pt'

    if not probe_path.exists():
        print(f"\nERROR: Linear probe not found")
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
    print(f"  Probe accuracy: {checkpoint.get('final_accuracy', 'N/A')}")

    # Compute IG2
    print(f"\n{'='*60}")
    print(f"Computing IG2 attribution scores...")
    print(f"{'='*60}")
    print(f"Number of steps: {args.num_steps}")
    print(f"Use squared gap: {args.use_squared_gap}")

    ig2_scores = compute_ig2_for_sae_features(
        sae_features=sae_features,
        linear_probe=probe,
        num_steps=args.num_steps,
        use_squared_gap=args.use_squared_gap,
        device=device
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

    # Save results
    output_dir = Path(config['paths']['results_dir']) / args.stage / 'ig2'
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

    result_path = output_dir / 'ig2_results.pt'
    result.save(result_path)

    print(f"\nIG2 results saved to: {result_path}")

    # Save feature importance ranking
    feature_ranking = torch.argsort(ig2_scores, descending=True)
    ranking_path = output_dir / 'feature_ranking.pt'
    torch.save({
        'ranking': feature_ranking,
        'scores': ig2_scores[feature_ranking],
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
    args = parser.parse_args()

    main(args)
