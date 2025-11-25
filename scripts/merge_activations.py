"""
Merge activations from all demographic categories for gSAE training.

This script combines activations extracted for each demographic category
into a single file suitable for training a unified gSAE model.

Output structure matches korean-sparse-llm-features-open format:
- Single pickle file with merged activations from all demographics
- Metadata preserved for downstream analysis (probing, IG2)

Usage:
    python scripts/merge_activations.py --stage pilot
    python scripts/merge_activations.py --stage medium --demographics 성별 인종 종교
"""

import sys
import pickle
import argparse
import torch
import json
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils import load_config
from src.utils.demographic_utils import get_all_demographics


def merge_activations(
    stage: str,
    results_dir: Path,
    demographics: List[str],
) -> Dict:
    """
    Merge activations from multiple demographic categories.

    Args:
        stage: Experiment stage (pilot, medium, full)
        results_dir: Path to results directory
        demographics: List of demographic categories to merge

    Returns:
        Dictionary with merged activations and metadata
    """
    print(f"\n=== Merging Activations for gSAE Training ===")
    print(f"Stage: {stage}")
    print(f"Demographics: {demographics}")

    # Storage for merged data
    # Use stage-based keys to match format expected by downstream scripts (03_train_sae.py, etc.)
    merged = {
        f'{stage}_residual_q1': [],
        f'{stage}_residual_q2': [],
        f'{stage}_residual_q3': [],
        f'{stage}_input_ids': [],
        f'{stage}_labels': [],
        f'{stage}_prompts': [],
        f'{stage}_generated_texts': [],
        f'{stage}_demographics': [],  # Track which demographic each sample belongs to
    }

    # Metadata for downstream analysis
    metadata = {
        'stage': stage,
        'demographics': demographics,
        'sample_indices': {},  # {demographic: [start_idx, end_idx]}
        'num_samples_per_demographic': {},
        'total_samples': 0,
    }

    current_idx = 0

    for demo in demographics:
        demo_dir = results_dir / stage / demo
        activation_path = demo_dir / 'activations.pkl'

        if not activation_path.exists():
            print(f"  [WARNING] Skipping {demo}: {activation_path} not found")
            continue

        print(f"\n  Loading {demo}...")

        with open(activation_path, 'rb') as f:
            data = pickle.load(f)

        # Get number of samples
        num_samples = len(data[f'{stage}_labels'])
        print(f"    Samples: {num_samples}")

        # Append to merged data
        merged[f'{stage}_residual_q1'].append(data[f'{stage}_residual_q1'])
        merged[f'{stage}_residual_q2'].append(data[f'{stage}_residual_q2'])
        merged[f'{stage}_residual_q3'].append(data[f'{stage}_residual_q3'])
        merged[f'{stage}_input_ids'].append(data[f'{stage}_input_ids'])
        merged[f'{stage}_labels'].extend(data[f'{stage}_labels'])
        merged[f'{stage}_prompts'].extend(data[f'{stage}_prompts'])
        merged[f'{stage}_generated_texts'].extend(data[f'{stage}_generated_texts'])

        # Track demographic for each sample
        merged[f'{stage}_demographics'].extend([demo] * num_samples)

        # Update metadata
        metadata['sample_indices'][demo] = [current_idx, current_idx + num_samples]
        metadata['num_samples_per_demographic'][demo] = num_samples
        current_idx += num_samples

    metadata['total_samples'] = current_idx

    # Concatenate tensors
    if merged[f'{stage}_residual_q1']:
        merged[f'{stage}_residual_q1'] = torch.cat(merged[f'{stage}_residual_q1'], dim=0)
        merged[f'{stage}_residual_q2'] = torch.cat(merged[f'{stage}_residual_q2'], dim=0)
        merged[f'{stage}_residual_q3'] = torch.cat(merged[f'{stage}_residual_q3'], dim=0)
        merged[f'{stage}_input_ids'] = torch.cat(merged[f'{stage}_input_ids'], dim=0)

    return merged, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Merge activations from multiple demographics for gSAE training"
    )
    parser.add_argument(
        '--stage',
        type=str,
        default='pilot',
        choices=['pilot', 'medium', 'full'],
        help='Experiment stage'
    )
    parser.add_argument(
        '--demographics',
        type=str,
        nargs='*',
        default=None,
        help='Demographics to merge (default: all available)'
    )
    args = parser.parse_args()

    # Load config
    config = load_config('configs/experiment_config.yaml')
    results_dir = Path(config['paths']['results_dir'])

    # Get demographics to merge
    if args.demographics:
        demographics = args.demographics
    else:
        # Get all available demographics
        all_demos = get_all_demographics()
        demographics = list(all_demos.keys())

    print(f"Demographics to merge: {demographics}")

    # Check which demographics actually have data
    available_demos = []
    for demo in demographics:
        demo_dir = results_dir / args.stage / demo
        if (demo_dir / 'activations.pkl').exists():
            available_demos.append(demo)
        else:
            print(f"  [SKIP] {demo}: No activation data found")

    if not available_demos:
        print("\nERROR: No demographic activation data found!")
        print(f"Please run activation extraction first:")
        print(f"  python scripts/02_generate_and_extract_activations.py --stage {args.stage} --demographic <DEMOGRAPHIC>")
        return 1

    # Merge activations
    merged, metadata = merge_activations(
        stage=args.stage,
        results_dir=results_dir,
        demographics=available_demos,
    )

    if metadata['total_samples'] == 0:
        print("\nERROR: No samples to merge!")
        return 1

    # Save merged activations
    output_dir = results_dir / args.stage
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to default location so downstream scripts (03_train_sae.py, etc.) can find it
    output_path = output_dir / 'activations.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(merged, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n=== Merge Complete ===")
    print(f"Saved merged activations to: {output_path}")

    # Save metadata
    metadata_path = output_dir / 'activations_metadata.json'
    # Convert any non-serializable types
    metadata_serializable = {
        'stage': metadata['stage'],
        'demographics': metadata['demographics'],
        'sample_indices': metadata['sample_indices'],
        'num_samples_per_demographic': metadata['num_samples_per_demographic'],
        'total_samples': metadata['total_samples'],
        'activation_shapes': {
            'residual_q1': list(merged[f'{args.stage}_residual_q1'].shape),
            'residual_q2': list(merged[f'{args.stage}_residual_q2'].shape),
            'residual_q3': list(merged[f'{args.stage}_residual_q3'].shape),
        }
    }
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata_serializable, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata to: {metadata_path}")

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Activation shape: {merged[f'{args.stage}_residual_q2'].shape}")
    print(f"\nSamples per demographic:")
    for demo, count in metadata['num_samples_per_demographic'].items():
        pct = count / metadata['total_samples'] * 100
        print(f"  {demo}: {count} ({pct:.1f}%)")

    print(f"\n{'='*60}")
    print(f"Ready for gSAE training!")
    print(f"  python scripts/03_train_sae.py --stage {args.stage}")
    print(f"{'='*60}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
