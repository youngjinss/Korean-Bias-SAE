"""
Plotting utilities for various visualization types.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path


def plot_umap_clusters(
    embeddings: np.ndarray,
    all_features: List[int],
    demographic2topfeatures: Dict[str, List[int]],
    demographic_labels_ko: List[str],
    demographic_labels_en: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 18),
    top_k: int = 100
) -> plt.Figure:
    """
    Plot UMAP feature space clusters for multiple demographics.

    Args:
        embeddings: UMAP 2D embeddings [N_features, 2]
        all_features: List of all feature indices
        demographic2topfeatures: Map of demographics to top features
        demographic_labels_ko: Korean demographic labels
        demographic_labels_en: English demographic labels
        save_path: Path to save figure
        figsize: Figure size
        top_k: Number of top features (for title)

    Returns:
        Figure object
    """
    n_demographics = len(demographic_labels_ko)
    n_cols = 3
    n_rows = (n_demographics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Main title
    fig.suptitle(
        f"편향 특성 벡터 분포 (Demographics × Top-{top_k} Features)\n"
        f"Bias Feature Distribution across Demographics",
        fontsize=22,
        y=0.995
    )

    for i, (demo_ko, demo_en) in enumerate(zip(demographic_labels_ko, demographic_labels_en)):
        if i >= len(axes):
            break

        ax = axes[i]
        top_features = demographic2topfeatures.get(demo_ko, [])

        # Create color mapping
        colors = ['red' if fid in top_features else 'lightgray' for fid in all_features]

        # Scatter plot
        sns.scatterplot(
            x=embeddings[:, 0],
            y=embeddings[:, 1],
            hue=colors,
            palette=['red', 'lightgray'],
            hue_order=['red', 'lightgray'],
            alpha=0.8,
            s=30,
            ax=ax,
            legend=False
        )

        # Styling
        ax.set_title(f"{demo_ko} ({demo_en})", fontsize=18, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)

    # Hide extra subplots
    for i in range(n_demographics, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_ig2_rankings(
    ig2_results: Dict[str, torch.Tensor],
    demographic_labels_ko: List[str],
    demographic_labels_en: List[str],
    save_path: Optional[Path] = None,
    top_k: int = 20,
    figsize: Tuple[int, int] = (20, 15)
) -> plt.Figure:
    """
    Plot IG² attribution rankings for each demographic.

    Args:
        ig2_results: Dictionary of IG² scores per demographic
        demographic_labels_ko: Korean demographic labels
        demographic_labels_en: English demographic labels
        save_path: Path to save figure
        top_k: Number of top features to show
        figsize: Figure size

    Returns:
        Figure object
    """
    n_demographics = len(demographic_labels_ko)
    n_cols = 3
    n_rows = (n_demographics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    fig.suptitle(
        f"IG² 특성 중요도 순위 (Top-{top_k} Features)\n"
        f"IG² Feature Importance Rankings",
        fontsize=22,
        y=0.995
    )

    for i, (demo_ko, demo_en) in enumerate(zip(demographic_labels_ko, demographic_labels_en)):
        if i >= len(axes):
            break

        ax = axes[i]

        if demo_ko not in ig2_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f"{demo_ko} ({demo_en})", fontsize=16)
            continue

        scores = ig2_results[demo_ko]
        if isinstance(scores, dict):
            scores = scores['feature_scores']

        # Get top-k
        k = min(top_k, len(scores))
        top_scores, top_indices = torch.topk(scores, k=k)

        # Convert to numpy
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        # Color gradient
        colors = plt.cm.Reds(np.linspace(0.4, 0.9, k))

        # Horizontal bar chart
        y_pos = np.arange(k)
        ax.barh(y_pos, top_scores, color=colors)

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels([f"F{idx}" for idx in top_indices], fontsize=10)
        ax.set_xlabel("IG² Score", fontsize=12)
        ax.set_title(f"{demo_ko} ({demo_en})", fontsize=16, pad=10)
        ax.invert_yaxis()

    # Hide extra subplots
    for i in range(n_demographics, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_activation_heatmap(
    feature_activations: torch.Tensor,
    selected_features: List[int],
    prompt_labels: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (20, 12),
    cmap: str = 'viridis'
) -> plt.Figure:
    """
    Plot heatmap of feature activations.

    Args:
        feature_activations: Tensor [N_prompts, N_features]
        selected_features: List of feature indices to visualize
        prompt_labels: Optional labels for prompts
        save_path: Path to save figure
        figsize: Figure size
        cmap: Colormap name

    Returns:
        Figure object
    """
    # Select features
    selected_activations = feature_activations[:, selected_features].cpu().numpy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Heatmap
    im = ax.imshow(selected_activations, aspect='auto', cmap=cmap, interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Activation Magnitude', fontsize=14)

    # Labels
    ax.set_xlabel(f'Bias Features (Top {len(selected_features)})', fontsize=14)
    ax.set_ylabel('Prompts', fontsize=14)
    ax.set_title('편향 특성 활성화 패턴\nBias Feature Activation Patterns',
                 fontsize=18, pad=20)

    # Y-axis labels
    if prompt_labels:
        step = max(1, len(prompt_labels) // 20)
        y_ticks = np.arange(0, len(prompt_labels), step)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([prompt_labels[i] for i in y_ticks], fontsize=8)

    # X-axis
    x_ticks = np.linspace(0, len(selected_features)-1, min(10, len(selected_features)), dtype=int)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"F{selected_features[i]}" for i in x_ticks],
                       fontsize=10, rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_verification_effects(
    verification_results: Dict[str, Dict],
    demographic_labels_ko: List[str],
    demographic_labels_en: List[str],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 15)
) -> plt.Figure:
    """
    Plot verification effects (suppress/amplify/random).

    Args:
        verification_results: Dictionary with verification results
        demographic_labels_ko: Korean demographic labels
        demographic_labels_en: English demographic labels
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    n_demographics = len(demographic_labels_ko)
    n_cols = 3
    n_rows = (n_demographics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    fig.suptitle(
        '편향 특성 조작 효과 (Suppress / Amplify / Random)\n'
        'Bias Feature Manipulation Effects',
        fontsize=22,
        y=0.995
    )

    for i, (demo_ko, demo_en) in enumerate(zip(demographic_labels_ko, demographic_labels_en)):
        if i >= len(axes):
            break

        ax = axes[i]

        if demo_ko not in verification_results:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            ax.set_title(f"{demo_ko} ({demo_en})", fontsize=14)
            continue

        results = verification_results[demo_ko]

        # Extract means and stds
        x_labels = ['Baseline', 'Suppress', 'Amplify', 'Random']
        means = [
            results.get('baseline_gap_mean', 0),
            results.get('suppress_gap_mean', 0),
            results.get('amplify_gap_mean', 0),
            results.get('random_gap_mean', 0)
        ]
        stds = [
            results.get('baseline_gap_std', 0),
            results.get('suppress_gap_std', 0),
            results.get('amplify_gap_std', 0),
            results.get('random_gap_std', 0)
        ]

        # Bar colors
        colors = ['gray', 'blue', 'red', 'orange']

        # Bar chart
        x_pos = np.arange(len(x_labels))
        ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)

        # Baseline reference line
        ax.axhline(y=means[0], linestyle='--', color='gray', alpha=0.5, linewidth=1)

        # Labels
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_ylabel('Logit Gap', fontsize=12)
        ax.set_title(f"{demo_ko}\n({demo_en})", fontsize=14, pad=10)

        # Grid
        ax.grid(axis='y', alpha=0.3)

    # Hide extra subplots
    for i in range(n_demographics, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_training_loss(
    training_logs: pd.DataFrame,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Plot SAE training loss curves.

    Args:
        training_logs: DataFrame with training metrics
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    fig.suptitle('SAE 학습 손실 곡선\nSAE Training Loss Curves',
                 fontsize=18, y=0.995)

    # Total loss
    if 'total_loss' in training_logs.columns:
        ax = axes[0]
        ax.plot(training_logs['step'], training_logs['total_loss'],
                linewidth=2, color='blue')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Total Loss', fontsize=12)
        ax.set_title('Total Loss', fontsize=14)
        ax.grid(alpha=0.3)

    # Reconstruction loss
    if 'recon_loss' in training_logs.columns:
        ax = axes[1]
        ax.plot(training_logs['step'], training_logs['recon_loss'],
                linewidth=2, color='green')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Reconstruction Loss', fontsize=12)
        ax.set_title('Reconstruction Loss', fontsize=14)
        ax.grid(alpha=0.3)

    # Sparsity loss
    if 'sparsity_loss' in training_logs.columns:
        ax = axes[2]
        ax.plot(training_logs['step'], training_logs['sparsity_loss'],
                linewidth=2, color='red')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Sparsity Loss', fontsize=12)
        ax.set_title('Sparsity Loss', fontsize=14)
        ax.grid(alpha=0.3)

    # Sparsity (L0)
    if 'sparsity_l0' in training_logs.columns:
        ax = axes[3]
        ax.plot(training_logs['step'], training_logs['sparsity_l0'],
                linewidth=2, color='purple')
        ax.set_xlabel('Step', fontsize=12)
        ax.set_ylabel('Sparsity (L0)', fontsize=12)
        ax.set_title('Feature Sparsity', fontsize=14)
        ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


def plot_feature_frequency_histogram(
    demographic2topfeatures: Dict[str, List[int]],
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """
    Plot histogram of feature frequency across demographics.

    Args:
        demographic2topfeatures: Map of demographics to top features
        save_path: Path to save figure
        figsize: Figure size

    Returns:
        Figure object
    """
    from collections import Counter

    # Count feature occurrences
    feature_counts = Counter()
    for features in demographic2topfeatures.values():
        feature_counts.update(features)

    # Sort by count
    sorted_counts = sorted(feature_counts.values(), reverse=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(sorted_counts))
    ax.bar(x, sorted_counts, color='gray', alpha=0.7)

    # Reference line for number of demographics
    n_demographics = len(demographic2topfeatures)
    ax.axhline(y=n_demographics, color='red', linestyle='--',
               label=f'All {n_demographics} Demographics')

    ax.set_xlabel('Feature Index (sorted by frequency)', fontsize=12)
    ax.set_ylabel('Number of Demographics', fontsize=12)
    ax.set_title('편향 특성 빈도 분포\nBias Feature Frequency Distribution',
                 fontsize=16, pad=15)
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig
