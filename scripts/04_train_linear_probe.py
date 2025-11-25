"""
Train linear probe to predict demographic from SAE features.

This script is equivalent to korean-sparse-llm-features-open/script/train_label_recovery.py,
adapted for bias detection with multi-demographic support.

Key differences:
1. Predicts demographic values (male/female, etc.) instead of categories
2. Uses unified probe architecture with masking for all demographics
3. Works with generation-based activations (answer tokens)

IMPORTANT: Following the pattern from korean-sparse-llm-features-open, we train
SEPARATE linear probes for each demographic category. This is necessary because:
- Each demographic has different label values (gender: male/female, race: white/black/etc.)
- IG2 attribution requires a probe that classifies the specific demographic
- The SAE is shared across all demographics, but probes are demographic-specific
"""

import sys
import pickle
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.sae import GatedAutoEncoder, AutoEncoder
from src.models import BiasProbe
from src.utils import load_config
from src.utils.demographic_utils import get_demographic_info, get_demographic_mask, get_all_demographics


def get_feature_counts(sae, num_dict, activations, device='cuda'):
    """
    Extract SAE feature activations from hidden states.

    This follows the approach from korean-sparse-llm-features-open:
    - Pass activations through SAE
    - Get feature activations
    - Sum over sequence dimension (for time-aggregated features)

    Args:
        sae: Trained SAE model
        num_dict: SAE dictionary size (feature dimension)
        activations: Hidden state activations (num_samples, hidden_dim)
        device: Device to run on

    Returns:
        Feature counts: (num_samples, num_dict)
    """
    n_samples = len(activations)
    feature_counts = torch.zeros(n_samples, num_dict)

    sae.eval()
    print(f"Extracting SAE features for {n_samples} samples...")

    for s in tqdm(range(n_samples), desc="Extracting features"):
        sample = activations[s].unsqueeze(0).to(device)

        with torch.no_grad():
            # Forward through SAE
            _, features = sae(sample, output_features=True)

        # Sum over sequence dimension (if any)
        if features.dim() > 2:
            features = torch.sum(features, dim=1)  # (batch, seq, dict) -> (batch, dict)

        feature_counts[s] = features[0].detach().cpu()

    return feature_counts


def main(args):
    # Load config
    config = load_config('configs/experiment_config.yaml')

    # Determine which demographic to use
    # Priority: command line arg > config file
    if args.demographic:
        demographic = args.demographic
    else:
        demographic = config['data']['demographic']

    # Get primary device from devices list
    devices = config['model'].get('devices', ['cuda' if torch.cuda.is_available() else 'cpu'])
    device = torch.device(devices[0] if isinstance(devices, list) else devices)

    # Get demographic info
    demo_info = get_demographic_info(demographic, data_dir='data')
    demographic_values = demo_info['values']
    num_classes = len(demographic_values)

    print(f"=== Training Linear Probe ===")
    print(f"Stage: {args.stage}")
    print(f"Demographic: {demographic} ({demo_info['dimension_en']})")
    print(f"Values: {[v.strip() for v in demographic_values]}")
    print(f"Number of classes: {num_classes}")

    # Determine activation path
    # When --demographic is specified, use per-demographic activation file
    # Otherwise, use the default (possibly merged) activation file
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

    # Get activations and labels
    all_activations = activations_dict[f'{args.stage}_residual_{args.layer_quantile}']
    all_labels = activations_dict[f'{args.stage}_labels']

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

        # Filter activations and labels
        activations = all_activations[indices]
        labels = [all_labels[i] for i in indices]
        print(f"Filtered to {len(indices)} samples for {demographic}")
    else:
        activations = all_activations
        labels = all_labels

    print(f"Loaded {len(activations)} samples")
    print(f"Activation shape: {activations.shape}")

    # Load SAE
    sae_path = Path(config['paths']['models_dir']) / f'sae-{args.sae_type}_{args.stage}_{args.layer_quantile}' / 'model.pth'

    if not sae_path.exists():
        print(f"\nERROR: SAE model not found at {sae_path}")
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
    print(f"SAE loaded: dict_size={sae.dict_size}, activation_dim={sae.activation_dim}")

    # Extract SAE features
    print("\nExtracting SAE features...")
    feature_counts = get_feature_counts(sae, sae.dict_size, activations, device)

    # Clean up
    del sae
    torch.cuda.empty_cache()

    print(f"Feature counts shape: {feature_counts.shape}")

    # Prepare labels
    # Create mapping from demographic value to index
    label_to_idx = {val.strip(): idx for idx, val in enumerate(demographic_values)}

    # Convert string labels to indices
    label_indices = []
    for label in labels:
        if label not in label_to_idx:
            print(f"WARNING: Label '{label}' not in demographic_values, skipping")
            continue
        label_indices.append(label_to_idx[label])

    label_indices = torch.tensor(label_indices, dtype=torch.long)

    print(f"\nLabel distribution:")
    unique, counts = torch.unique(label_indices, return_counts=True)
    for idx, count in zip(unique, counts):
        demo_val = demographic_values[idx].strip()
        print(f"  {demo_val}: {count} ({count/len(label_indices)*100:.1f}%)")

    # Get mask for this demographic
    mask = get_demographic_mask(demographic, max_output_dim=10, data_dir='data')
    mask_tensor = torch.tensor(mask, dtype=torch.bool)

    print(f"\nMask for {demographic}: {mask[:num_classes] + ['...']}")

    # Initialize probe
    probe = BiasProbe(
        input_dim=sae.dict_size,
        output_dim=10,  # Fixed output dimension
        hidden_dims=[]   # Linear probe (no hidden layers)
    )
    probe.to(device)

    print(f"\nProbe architecture:")
    print(f"  Input dim: {sae.dict_size}")
    print(f"  Output dim: 10 (fixed)")
    print(f"  Active outputs: {num_classes}")
    print(f"  Hidden layers: None (linear)")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)

    X = feature_counts.to(device)
    y = label_indices.to(device)
    mask_tensor = mask_tensor.to(device)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training...")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"{'='*60}\n")

    losses = []
    accuracies = []
    best_accuracy = 0.0
    patience_counter = 0

    pbar = tqdm(range(1, args.epochs + 1), desc="Training probe")
    for epoch in pbar:
        # Forward pass with masking
        probe.train()
        logits = probe.forward(X, mask=mask_tensor)
        loss = criterion(logits, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging
        losses.append(loss.item())

        if epoch % args.eval_steps == 0:
            probe.eval()
            with torch.no_grad():
                logits = probe.forward(X, mask=mask_tensor)
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == y).float().mean().item()

            accuracies.append(accuracy)
            pbar.set_postfix_str(f"Acc: {accuracy:.3f}, Loss: {loss.item():.4f}")

            # Early stopping check
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

    pbar.close()

    # Final evaluation
    probe.eval()
    with torch.no_grad():
        logits = probe.forward(X, mask=mask_tensor)
        preds = torch.argmax(logits, dim=1)
        final_accuracy = (preds == y).float().mean().item()

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"{'='*60}")
    print(f"Final accuracy: {final_accuracy:.3f}")
    print(f"Best accuracy: {best_accuracy:.3f}")

    # Per-class accuracy
    print(f"\nPer-class accuracy:")
    for idx in range(num_classes):
        class_mask = (y == idx)
        if class_mask.sum() > 0:
            class_acc = (preds[class_mask] == y[class_mask]).float().mean().item()
            demo_val = demographic_values[idx].strip()
            print(f"  {demo_val}: {class_acc:.3f} ({class_mask.sum()} samples)")

    # Save probe - use per-demographic directory when demographic is specified
    if args.demographic:
        output_dir = Path(config['paths']['results_dir']) / args.stage / demographic / 'probe'
    else:
        output_dir = Path(config['paths']['results_dir']) / args.stage / 'probe'
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        'probe_state_dict': probe.state_dict(),
        'demographic': demographic,
        'demographic_values': demographic_values,
        'mask': mask,
        'num_classes': num_classes,
        'sae_type': args.sae_type,
        'layer_quantile': args.layer_quantile,
        'final_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
    }

    probe_path = output_dir / 'linear_probe.pt'
    torch.save(checkpoint, probe_path)
    print(f"\nSaved probe to: {probe_path}")

    # Save training metrics
    metrics = {
        'losses': losses,
        'accuracies': accuracies,
        'final_accuracy': final_accuracy,
        'best_accuracy': best_accuracy,
    }

    metrics_path = output_dir / 'training_metrics.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Saved metrics to: {metrics_path}")

    print(f"\nLinear probe training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train linear probe to predict demographic from SAE features"
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
        help='Which layer activations to use (q1=25%%, q2=50%%, q3=75%%)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10000,
        help='Maximum number of training epochs'
    )
    parser.add_argument(
        '--eval_steps',
        type=int,
        default=500,
        help='Evaluate every N epochs'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (in eval_steps)'
    )
    parser.add_argument(
        '--demographic',
        type=str,
        default=None,
        help='Demographic category to train probe for (e.g., 성별, 인종). '
             'If not specified, uses config default. When specified, loads '
             'per-demographic activations and saves probe to per-demographic directory.'
    )
    args = parser.parse_args()

    main(args)
