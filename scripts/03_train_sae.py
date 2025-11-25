"""
Train Sparse Autoencoder (SAE) on activations.

This script trains a Gated SAE on activations extracted from EXAONE-3.0-7.8B-Instruct.
Training logs are saved for visualization of loss curves.
"""

import torch
import pandas as pd
import argparse
import sys
from pathlib import Path
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.sae.gated_sae import GatedAutoEncoder, GatedTrainer
from src.utils.experiment_utils import load_config


def train_sae(
    activations: torch.Tensor,
    config: dict,
    save_dir: Path,
    save_interval: int = 100,
    log_interval: int = 10
):
    """
    Train Gated SAE on activations.

    Args:
        activations: Tensor of shape [N_samples, activation_dim]
        config: Training configuration
        save_dir: Directory to save model and logs
        save_interval: Steps between model checkpoints
        log_interval: Steps between logging
    """
    print("=" * 80)
    print("Training Gated Sparse Autoencoder")
    print("=" * 80)

    # Extract config (ensure numeric types - YAML may parse 1e-4 as string)
    activation_dim = int(config['sae']['activation_dim'])
    feature_dim = int(config['sae']['feature_dim'])
    total_steps = int(config['sae']['training']['total_steps'])
    batch_size = int(config['sae']['training']['batch_size'])
    lr = float(config['sae']['training']['learning_rate'])

    # Sparsity config
    sparsity_penalty = float(config['sae']['training']['sparsity_penalty'])
    p_start = float(config['sae']['training'].get('p_start', 1.0))
    p_end = float(config['sae']['training'].get('p_end', 0.0))
    anneal_start = int(config['sae']['training'].get('anneal_start', 5000))
    warmup_steps = int(config['sae']['training'].get('warmup_steps', 1000))

    # Get devices from config (list of devices for multi-GPU support)
    devices = config['model'].get('devices', ['cuda' if torch.cuda.is_available() else 'cpu'])
    if not isinstance(devices, list):
        devices = [devices]
    primary_device = torch.device(devices[0])
    multi_gpu = len(devices) > 1

    print(f"\nDevices: {devices}")
    print(f"Primary device: {primary_device}")
    print(f"Multi-GPU: {multi_gpu}")
    print(f"Activation dim: {activation_dim}")
    print(f"Feature dim: {feature_dim}")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Sparsity penalty: {sparsity_penalty}")

    # Initialize trainer with multi-GPU support
    trainer = GatedTrainer(
        dict_class=GatedAutoEncoder,
        activation_dim=activation_dim,
        dict_size=feature_dim,
        lr=lr,
        warmup_steps=warmup_steps,
        sparsity_function='Lp^p',
        initial_sparsity_penalty=sparsity_penalty,
        anneal_start=anneal_start,
        anneal_end=total_steps - 1,
        p_start=p_start,
        p_end=p_end,
        n_sparsity_updates=10,
        sparsity_queue_length=10,
        resample_steps=None,
        total_steps=total_steps,
        devices=devices  # Pass devices list for multi-GPU support
    )

    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80)

    # Training loop
    training_logs = []
    n_samples = len(activations)

    pbar = tqdm(range(total_steps), desc="Training SAE")

    for step in pbar:
        # Sample batch
        indices = torch.randint(0, n_samples, (batch_size,))
        batch = activations[indices]

        # Training step
        trainer.update(step, batch)

        # Logging
        if step % log_interval == 0:
            with torch.no_grad():
                # Get loss components
                loss_log = trainer.loss(batch.to(primary_device), step, logging=True)

                # Compute sparsity (L0 norm)
                features = loss_log.f
                sparsity_l0 = (features > 0).float().mean().item()

                # Log
                log_entry = {
                    'step': step,
                    'total_loss': loss_log.losses['loss'],
                    'recon_loss': loss_log.losses['mse_loss'],
                    'aux_loss': loss_log.losses['aux_loss'],
                    'sparsity_loss': loss_log.losses['sparsity_loss'],
                    'lp_loss': loss_log.losses['lp_loss'],
                    'sparsity_l0': sparsity_l0,
                    'p': loss_log.losses['p'],
                    'sparsity_coeff': loss_log.losses['sparsity_coeff'],
                    'lr': trainer.optimizer.param_groups[0]['lr']
                }
                training_logs.append(log_entry)

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{log_entry['total_loss']:.4f}",
                    'recon': f"{log_entry['recon_loss']:.4f}",
                    'sparsity': f"{sparsity_l0:.3f}",
                    'p': f"{log_entry['p']:.2f}"
                })

        # Save checkpoint
        if step % save_interval == 0 and step > 0:
            checkpoint_path = save_dir / f"checkpoint_step_{step}.pt"
            torch.save({
                'step': step,
                'model_state_dict': trainer.base_model.state_dict(),  # Use base_model to avoid DataParallel prefix
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': trainer.config
            }, checkpoint_path)

    pbar.close()

    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)

    # Save final model (use base_model to avoid DataParallel 'module.' prefix)
    final_model_path = save_dir / "model.pth"
    torch.save(trainer.base_model.state_dict(), final_model_path)
    print(f"✓ Saved final model to {final_model_path}")

    # Save training logs
    df_logs = pd.DataFrame(training_logs)
    log_path = save_dir / "training_logs.csv"
    df_logs.to_csv(log_path, index=False)
    print(f"✓ Saved training logs to {log_path}")

    # Print final statistics
    print(f"\nFinal Training Statistics:")
    print(f"  Total Loss:       {df_logs['total_loss'].iloc[-1]:.4f}")
    print(f"  Recon Loss:       {df_logs['recon_loss'].iloc[-1]:.4f}")
    print(f"  Sparsity Loss:    {df_logs['sparsity_loss'].iloc[-1]:.4f}")
    print(f"  Sparsity (L0):    {df_logs['sparsity_l0'].iloc[-1]:.4f}")
    print(f"  Target Sparsity:  0.0500")
    print(f"  Deviation:        {abs(df_logs['sparsity_l0'].iloc[-1] - 0.05):.4f}")

    return trainer.ae, df_logs


def main():
    parser = argparse.ArgumentParser(description="Train SAE on activations")
    parser.add_argument("--config", type=str, default="configs/experiment_config.yaml",
                       help="Path to config file")
    parser.add_argument("--stage", type=str, default="pilot",
                       choices=["pilot", "medium", "full"],
                       help="Data stage to use")
    parser.add_argument("--sae_type", type=str, default="gated",
                       choices=["gated", "standard"],
                       help="SAE type to train")
    parser.add_argument("--layer_quantile", type=str, default="q2",
                       choices=["q1", "q2", "q3"],
                       help="Layer quantile to use")
    parser.add_argument("--activations", type=str, default=None,
                       help="Path to activations file (default: results/{stage}/activations.pkl)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: checkpoints/sae-{sae_type}_{stage}_{layer_quantile}/)")
    parser.add_argument("--steps", type=int, default=None,
                       help="Override total training steps")
    parser.add_argument("--batch-size", type=int, default=None,
                       help="Override batch size")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override config if specified
    if args.steps is not None:
        config['sae']['training']['total_steps'] = args.steps
    if args.batch_size is not None:
        config['sae']['training']['batch_size'] = args.batch_size

    # Setup paths
    if args.activations is None:
        activations_path = PROJECT_ROOT / "results" / args.stage / "activations.pkl"
    else:
        activations_path = Path(args.activations)

    if args.output is None:
        output_dir = PROJECT_ROOT / "checkpoints" / f"sae-{args.sae_type}_{args.stage}_{args.layer_quantile}"
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading activations from: {activations_path}")

    # Load activations from pickle file
    import pickle
    with open(activations_path, 'rb') as f:
        data = pickle.load(f)

    # Extract the correct quantile activations
    quantile_key = f"{args.stage}_residual_{args.layer_quantile}"
    if quantile_key not in data:
        raise KeyError(f"Quantile key '{quantile_key}' not found. Available keys: {list(data.keys())}")

    activations = data[quantile_key]

    if not isinstance(activations, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(activations)}")

    print(f"Activations shape: {activations.shape}")
    print(f"Number of samples: {len(activations)}")

    # Train SAE
    model, logs = train_sae(
        activations=activations,
        config=config,
        save_dir=output_dir,
        save_interval=config['sae']['training'].get('save_interval', 500),
        log_interval=config['sae']['training'].get('log_interval', 10)
    )

    print(f"\n✓ Training complete!")
    print(f"✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
