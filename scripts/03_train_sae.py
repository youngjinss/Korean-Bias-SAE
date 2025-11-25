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
from src.utils.experiment_utils import setup_experiment, load_config


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

    # Extract config
    activation_dim = config['sae']['activation_dim']
    feature_dim = config['sae']['feature_dim']
    total_steps = config['sae']['training']['total_steps']
    batch_size = config['sae']['training']['batch_size']
    lr = config['sae']['training']['learning_rate']

    # Sparsity config
    sparsity_penalty = config['sae']['training']['sparsity_penalty']
    p_start = config['sae']['training'].get('p_start', 1.0)
    p_end = config['sae']['training'].get('p_end', 0.0)
    anneal_start = config['sae']['training'].get('anneal_start', 5000)
    warmup_steps = config['sae']['training'].get('warmup_steps', 1000)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    print(f"Activation dim: {activation_dim}")
    print(f"Feature dim: {feature_dim}")
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Sparsity penalty: {sparsity_penalty}")

    # Initialize trainer
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
        device=device
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
                loss_log = trainer.loss(batch.to(device), step, logging=True)

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
                'model_state_dict': trainer.ae.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'config': trainer.config
            }, checkpoint_path)

    pbar.close()

    print("\n" + "=" * 80)
    print("Training Complete")
    print("=" * 80)

    # Save final model
    final_model_path = save_dir / "sae_model.pt"
    torch.save(trainer.ae.state_dict(), final_model_path)
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
    parser.add_argument("--activations", type=str, required=True,
                       help="Path to activations file (.pt)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: results/{stage}/)")
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

    # Setup output directory
    if args.output is None:
        output_dir = PROJECT_ROOT / "results" / args.stage
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading activations from: {args.activations}")
    activations = torch.load(args.activations, map_location='cpu')

    # Handle different activation formats
    if isinstance(activations, dict):
        # Extract activations tensor
        if 'activations' in activations:
            activations = activations['activations']
        elif 'features' in activations:
            activations = activations['features']
        else:
            raise KeyError("Could not find activations in loaded data")

    print(f"Activations shape: {activations.shape}")

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
