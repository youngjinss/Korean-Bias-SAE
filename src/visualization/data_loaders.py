"""
Data loading utilities for visualization.
"""

import json
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any


def load_sae_features(
    results_dir: Path,
    stage: str = "pilot"
) -> Tuple[torch.Tensor, List[str]]:
    """
    Load SAE feature activations.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')

    Returns:
        features: Tensor of shape [N_prompts, 100000]
        prompt_ids: List of prompt identifiers
    """
    feature_path = results_dir / stage / "sae_features.pt"

    if not feature_path.exists():
        raise FileNotFoundError(f"SAE features not found at {feature_path}")

    data = torch.load(feature_path, map_location='cpu')

    if isinstance(data, dict):
        features = data['features']
        prompt_ids = data.get('prompt_ids', [f"prompt_{i}" for i in range(len(features))])
    else:
        features = data
        prompt_ids = [f"prompt_{i}" for i in range(len(features))]

    return features, prompt_ids


def load_ig2_results(
    results_dir: Path,
    stage: str = "pilot",
    demographic: Optional[str] = None
) -> Dict[str, torch.Tensor]:
    """
    Load IG² attribution results.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')
        demographic: If specified, load only for this demographic

    Returns:
        Dictionary mapping demographic names to IG² scores
    """
    ig2_path = results_dir / stage / "ig2_results.pt"

    if not ig2_path.exists():
        raise FileNotFoundError(f"IG² results not found at {ig2_path}")

    data = torch.load(ig2_path, map_location='cpu')

    if demographic is not None:
        if demographic not in data:
            raise ValueError(f"Demographic '{demographic}' not found in IG² results")
        return {demographic: data[demographic]}

    return data


def load_verification_results(
    results_dir: Path,
    stage: str = "pilot"
) -> Dict[str, Dict[str, Any]]:
    """
    Load verification (suppress/amplify) results.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')

    Returns:
        Dictionary with verification results per demographic
    """
    verif_path = results_dir / stage / "verification_results.json"

    if not verif_path.exists():
        raise FileNotFoundError(f"Verification results not found at {verif_path}")

    with open(verif_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    return results


def load_demographics(data_dir: Path) -> Dict[str, Dict]:
    """
    Load demographic dictionary.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary with demographic dimensions and values
    """
    demo_path = data_dir / "demographic_dict_ko.json"

    if not demo_path.exists():
        raise FileNotFoundError(f"Demographics file not found at {demo_path}")

    with open(demo_path, 'r', encoding='utf-8') as f:
        demographics = json.load(f)

    return demographics


def load_training_logs(
    results_dir: Path,
    stage: str = "pilot"
) -> pd.DataFrame:
    """
    Load SAE training logs.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')

    Returns:
        DataFrame with training metrics (step, loss, recon_loss, sparsity_loss, etc.)
    """
    log_path = results_dir / stage / "training_logs.csv"

    if not log_path.exists():
        # Try alternate naming
        log_path = results_dir / stage / "sae_training_logs.csv"

    if not log_path.exists():
        raise FileNotFoundError(f"Training logs not found at {log_path}")

    df = pd.read_csv(log_path)
    return df


def load_sae_decoder_weights(
    results_dir: Path,
    stage: str = "pilot"
) -> torch.Tensor:
    """
    Load SAE decoder weights for UMAP visualization.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')

    Returns:
        Decoder weights tensor of shape [100000, 4096]
    """
    model_path = results_dir / stage / "sae_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"SAE model not found at {model_path}")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract decoder weights
    if 'decoder.weight' in checkpoint:
        decoder_weights = checkpoint['decoder.weight']
    elif 'model_state_dict' in checkpoint:
        decoder_weights = checkpoint['model_state_dict']['decoder.weight']
    elif 'state_dict' in checkpoint:
        decoder_weights = checkpoint['state_dict']['decoder.weight']
    else:
        raise KeyError("Could not find decoder weights in checkpoint")

    # Transpose to [num_features, latent_dim]
    if decoder_weights.shape[0] == 4096:
        decoder_weights = decoder_weights.T

    return decoder_weights


def load_bias_prompts(
    data_dir: Path,
    stage: str = "pilot"
) -> List[Dict[str, Any]]:
    """
    Load generated bias prompts.

    Args:
        data_dir: Path to data directory
        stage: Data stage ('pilot', 'medium', or 'full')

    Returns:
        List of BiasPrompt dictionaries
    """
    import jsonlines

    prompt_path = data_dir / "generated" / f"{stage}_prompts.jsonl"

    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompts not found at {prompt_path}")

    prompts = []
    with jsonlines.open(prompt_path, 'r') as reader:
        for obj in reader:
            prompts.append(obj)

    return prompts


def get_demographic_labels(
    demographics: Dict[str, Dict]
) -> Tuple[List[str], List[str]]:
    """
    Extract Korean and English demographic labels.

    Args:
        demographics: Demographic dictionary

    Returns:
        korean_labels: List of Korean demographic names
        english_labels: List of English demographic names
    """
    korean_labels = list(demographics.keys())
    english_labels = [demo['dimension_en'] for demo in demographics.values()]

    return korean_labels, english_labels
