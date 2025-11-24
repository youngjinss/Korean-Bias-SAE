"""
Sparse Autoencoder implementations.
Integrated from korean-sparse-llm-features-open for standalone usage.
"""

from .gated_sae import GatedAutoEncoder, GatedTrainer
from .standard_sae import AutoEncoder, StandardTrainer

__all__ = [
    'GatedAutoEncoder',
    'GatedTrainer',
    'AutoEncoder',
    'StandardTrainer',
]
