"""
Sparse Autoencoder (SAE) wrapper for feature extraction.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SAEWrapper:
    """
    Wrapper for gSAE (Gated Sparse Autoencoder).

    Provides convenient methods for:
    - Loading pre-trained gSAE
    - Encoding activations to sparse features
    - Decoding sparse features back to activations
    """

    def __init__(
        self,
        sae_path: str,
        sae_type: str = "gated",
        device: str = "cuda"
    ):
        """
        Initialize SAE wrapper.

        Args:
            sae_path: Path to pre-trained SAE weights
            sae_type: 'gated' or 'standard'
            device: Device to load SAE on
        """
        self.sae_path = Path(sae_path)
        self.sae_type = sae_type
        self.device = device

        logger.info(f"Loading {sae_type} SAE from: {sae_path}")

        # Import SAE module from local implementation
        if sae_type == "gated":
            from .sae import GatedAutoEncoder
            self.sae = GatedAutoEncoder.from_pretrained(str(self.sae_path), device=device)
        elif sae_type == "standard":
            from .sae import AutoEncoder
            self.sae = AutoEncoder.from_pretrained(str(self.sae_path), device=device)
        else:
            raise ValueError(f"Invalid sae_type: {sae_type}. Must be 'gated' or 'standard'")

        self.sae.eval()  # Set to evaluation mode
        logger.info(f"SAE loaded successfully")
        logger.info(f"  Activation dim: {self.sae.activation_dim}")
        logger.info(f"  Feature dim: {self.sae.dict_size}")

        self.activation_dim = self.sae.activation_dim
        self.feature_dim = self.sae.dict_size

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Encode activations to sparse features.

        Args:
            activations: Input activations (batch, activation_dim)

        Returns:
            Sparse features (batch, feature_dim)
        """
        with torch.no_grad():
            features = self.sae.encode(activations)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to activations.

        Args:
            features: Sparse features (batch, feature_dim)

        Returns:
            Reconstructed activations (batch, activation_dim)
        """
        with torch.no_grad():
            reconstructed = self.sae.decode(features)
        return reconstructed

    def forward(self, activations: torch.Tensor) -> tuple:
        """
        Full forward pass: encode and decode.

        Args:
            activations: Input activations (batch, activation_dim)

        Returns:
            Tuple of (reconstructed_activations, features)
        """
        with torch.no_grad():
            reconstructed, features = self.sae(activations, output_features=True)
        return reconstructed, features

    def get_feature_sparsity(self, features: torch.Tensor) -> float:
        """
        Compute feature sparsity (fraction of zero features).

        Args:
            features: Feature tensor (batch, feature_dim)

        Returns:
            Sparsity ratio (0 to 1)
        """
        sparsity = (features == 0).float().mean().item()
        return sparsity

    def get_reconstruction_error(
        self,
        activations: torch.Tensor
    ) -> float:
        """
        Compute reconstruction error (L2 distance).

        Args:
            activations: Original activations (batch, activation_dim)

        Returns:
            Mean L2 reconstruction error
        """
        reconstructed, _ = self.forward(activations)
        error = torch.nn.functional.mse_loss(reconstructed, activations)
        return error.item()

    def __repr__(self):
        return (f"SAEWrapper(type={self.sae_type}, "
                f"activation_dim={self.activation_dim}, "
                f"feature_dim={self.feature_dim})")
