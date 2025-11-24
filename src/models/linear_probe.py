"""
Linear Probe for mapping SAE features to demographic predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class BiasProbe(nn.Module):
    """
    Linear probe that maps SAE features to demographic prediction logits.

    Architecture:
        SAE features (100K dims) -> Linear -> output_dim logits

    Supports masking for demographics with fewer than output_dim values.
    For example, with output_dim=10:
        - 성별 (2 values): mask=[T,T,F,F,F,F,F,F,F,F]
        - 나이 (4 values): mask=[T,T,T,T,F,F,F,F,F,F]
        - 인종 (10 values): mask=[T,T,T,T,T,T,T,T,T,T]

    Can be extended to MLP by specifying hidden_dims.
    """

    def __init__(
        self,
        input_dim: int = 100000,
        output_dim: int = 10,
        hidden_dims: Optional[List[int]] = None
    ):
        """
        Initialize bias probe.

        Args:
            input_dim: SAE feature dimension
            output_dim: Fixed output dimension (max across all demographics)
            hidden_dims: Optional hidden layer dimensions for MLP
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims or []

        # Build network
        layers = []
        in_dim = input_dim

        # Add hidden layers if specified
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        sae_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional masking.

        Args:
            sae_features: SAE feature activations (batch, input_dim)
            mask: Boolean mask for valid positions (output_dim,)
                  True = valid, False = padding

        Returns:
            Logits (batch, output_dim)
            If mask provided, masked positions are set to -inf
        """
        logits = self.network(sae_features)

        # Apply mask if provided
        if mask is not None:
            # Expand mask to batch dimension: (output_dim,) -> (1, output_dim)
            mask = mask.unsqueeze(0)
            # Set masked (invalid) positions to -inf
            logits = logits.masked_fill(~mask, float('-inf'))

        return logits

    def predict_probs(
        self,
        sae_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get probability predictions with masking.

        Args:
            sae_features: SAE feature activations (batch, input_dim)
            mask: Boolean mask for valid positions (output_dim,)

        Returns:
            Probabilities (batch, output_dim)
            Masked positions will have probability 0.0
        """
        logits = self.forward(sae_features, mask=mask)
        probs = F.softmax(logits, dim=-1)
        return probs


class ProbeTrainer:
    """
    Trainer for the bias probe.
    """

    def __init__(
        self,
        probe: BiasProbe,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: str = "cuda"
    ):
        """
        Initialize probe trainer.

        Args:
            probe: BiasProbe model
            learning_rate: Learning rate
            weight_decay: L2 regularization
            device: Device to train on
        """
        self.probe = probe.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            probe.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.best_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(
        self,
        dataloader: DataLoader,
        loss_type: str = "kl",
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Train for one epoch with optional masking.

        Args:
            dataloader: Training data loader
            loss_type: 'kl' for KL divergence or 'ce' for cross-entropy
            mask: Boolean mask for valid demographic positions (output_dim,)

        Returns:
            Dictionary of metrics
        """
        self.probe.train()
        total_loss = 0.0
        num_batches = 0

        # Move mask to device if provided
        if mask is not None:
            mask = mask.to(self.device)

        for batch in dataloader:
            features, labels = batch
            features = features.to(self.device)
            labels = labels.to(self.device)

            # Forward pass with mask
            logits = self.probe(features, mask=mask)

            # Compute loss
            if loss_type == "kl":
                # KL divergence for soft labels
                log_probs = F.log_softmax(logits, dim=-1)
                loss = F.kl_div(log_probs, labels, reduction='batchmean')
            elif loss_type == "ce":
                # Cross-entropy for hard labels
                loss = F.cross_entropy(logits, labels)
            else:
                raise ValueError(f"Invalid loss_type: {loss_type}")

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    def evaluate(
        self,
        dataloader: DataLoader,
        loss_type: str = "kl",
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Evaluate on validation set with optional masking.

        Args:
            dataloader: Validation data loader
            loss_type: Loss type
            mask: Boolean mask for valid demographic positions (output_dim,)

        Returns:
            Dictionary of metrics
        """
        self.probe.eval()
        total_loss = 0.0
        num_batches = 0

        # Move mask to device if provided
        if mask is not None:
            mask = mask.to(self.device)

        with torch.no_grad():
            for batch in dataloader:
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)

                # Forward pass with mask
                logits = self.probe(features, mask=mask)

                if loss_type == "kl":
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = F.kl_div(log_probs, labels, reduction='batchmean')
                elif loss_type == "ce":
                    loss = F.cross_entropy(logits, labels)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    def check_early_stopping(
        self,
        val_loss: float,
        patience: int = 5
    ) -> bool:
        """
        Check if should stop early.

        Args:
            val_loss: Validation loss
            patience: Patience for early stopping

        Returns:
            True if should stop
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered (patience={patience})")
                return True
            return False

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'probe_state_dict': self.probe.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.probe.load_state_dict(checkpoint['probe_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        logger.info(f"Checkpoint loaded: {path}")


class SAEFeatureDataset(Dataset):
    """
    Dataset for SAE features and labels.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Initialize dataset.

        Args:
            features: SAE features (num_samples, feature_dim)
            labels: Labels (num_samples, num_classes) for soft labels
                    or (num_samples,) for hard labels
        """
        assert features.shape[0] == labels.shape[0], "Feature and label count mismatch"
        self.features = features
        self.labels = labels

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
