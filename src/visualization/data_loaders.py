"""
Data loading utilities for visualization.

Updated to support per-demographic directory structure:
- results/{stage}/{demographic}/ig2/ig2_results.pt
- results/{stage}/{demographic}/probe/linear_probe.pt
- results/{stage}/{demographic}/verification/
- results/models/sae-{type}_{stage}_{layer}/model.pth
"""

import json
import pickle
import torch
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union


def load_sae_features(
    results_dir: Path,
    stage: str = "pilot",
    demographic: Optional[str] = None
) -> Tuple[torch.Tensor, List[str]]:
    """
    Load SAE feature activations.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')
        demographic: If specified, load from per-demographic directory

    Returns:
        features: Tensor of shape [N_prompts, 100000]
        prompt_ids: List of prompt identifiers
    """
    if demographic:
        feature_path = results_dir / stage / demographic / "sae_features.pt"
    else:
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
) -> Dict[str, Any]:
    """
    Load IG² attribution results.

    With the new per-demographic structure, this function can:
    1. Load for a specific demographic from: results/{stage}/{demographic}/ig2/ig2_results.pt
    2. Load all demographics by scanning all subdirectories

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')
        demographic: If specified, load only for this demographic

    Returns:
        Dictionary mapping demographic names to IG² result data
    """
    results = {}

    if demographic is not None:
        # Load for specific demographic
        ig2_path = results_dir / stage / demographic / "ig2" / "ig2_results.pt"
        if not ig2_path.exists():
            # Try old path structure
            ig2_path = results_dir / stage / "ig2" / "ig2_results.pt"

        if not ig2_path.exists():
            raise FileNotFoundError(f"IG² results not found at {ig2_path}")

        data = torch.load(ig2_path, map_location='cpu')
        results[demographic] = data
    else:
        # Try to load from all demographic subdirectories
        stage_dir = results_dir / stage

        if not stage_dir.exists():
            raise FileNotFoundError(f"Stage directory not found at {stage_dir}")

        # Find all demographic directories with ig2 results
        for subdir in stage_dir.iterdir():
            if subdir.is_dir():
                ig2_path = subdir / "ig2" / "ig2_results.pt"
                if ig2_path.exists():
                    demo_name = subdir.name
                    data = torch.load(ig2_path, map_location='cpu')
                    results[demo_name] = data

        # If no per-demographic results found, try old structure
        if not results:
            old_ig2_path = stage_dir / "ig2" / "ig2_results.pt"
            if old_ig2_path.exists():
                data = torch.load(old_ig2_path, map_location='cpu')
                # Old format might have all demographics in one file
                if isinstance(data, dict) and 'feature_scores' in data:
                    # Single demographic result
                    results['default'] = data
                else:
                    results = data

    if not results:
        raise FileNotFoundError(f"No IG² results found in {results_dir / stage}")

    return results


def load_verification_results(
    results_dir: Path,
    stage: str = "pilot",
    demographic: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load verification (suppress/amplify) results.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')
        demographic: If specified, load only for this demographic

    Returns:
        Dictionary with verification results per demographic
    """
    results = {}

    if demographic is not None:
        # Load for specific demographic
        verif_dir = results_dir / stage / demographic / "verification"
        if verif_dir.exists():
            results[demographic] = _load_verification_from_dir(verif_dir)
        else:
            raise FileNotFoundError(f"Verification results not found at {verif_dir}")
    else:
        # Try to load from all demographic subdirectories
        stage_dir = results_dir / stage

        for subdir in stage_dir.iterdir():
            if subdir.is_dir():
                verif_dir = subdir / "verification"
                if verif_dir.exists():
                    demo_name = subdir.name
                    results[demo_name] = _load_verification_from_dir(verif_dir)

        # If no per-demographic results found, try old structure
        if not results:
            old_verif_path = stage_dir / "verification_results.json"
            if old_verif_path.exists():
                with open(old_verif_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)

    if not results:
        raise FileNotFoundError(f"No verification results found in {results_dir / stage}")

    return results


def _load_verification_from_dir(verif_dir: Path) -> Dict[str, Any]:
    """Load verification results from a directory with JSON files."""
    result = {}

    # Load suppression test
    suppress_path = verif_dir / "suppression_test.json"
    if suppress_path.exists():
        with open(suppress_path, 'r', encoding='utf-8') as f:
            suppress_data = json.load(f)
            result['baseline_gap_mean'] = suppress_data.get('gap_before', 0)
            result['suppress_gap_mean'] = suppress_data.get('gap_after', 0)
            result['suppress_gap_std'] = suppress_data.get('metadata', {}).get('gap_std_after', 0)

    # Load amplification test
    amplify_path = verif_dir / "amplification_test.json"
    if amplify_path.exists():
        with open(amplify_path, 'r', encoding='utf-8') as f:
            amplify_data = json.load(f)
            result['amplify_gap_mean'] = amplify_data.get('gap_after', 0)
            result['amplify_gap_std'] = amplify_data.get('metadata', {}).get('gap_std_after', 0)

    # Load random control
    random_path = verif_dir / "random_control.json"
    if random_path.exists():
        with open(random_path, 'r', encoding='utf-8') as f:
            random_data = json.load(f)
            result['random_gap_mean'] = random_data.get('mean_gap_change', 0) * result.get('baseline_gap_mean', 1) + result.get('baseline_gap_mean', 0)
            result['random_gap_std'] = random_data.get('std_gap_change', 0)

    return result


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
    stage: str = "pilot",
    sae_type: str = "gated",
    layer_quantile: str = "q2"
) -> pd.DataFrame:
    """
    Load SAE training logs.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')
        sae_type: SAE type ('gated' or 'standard')
        layer_quantile: Layer quantile ('q1', 'q2', 'q3')

    Returns:
        DataFrame with training metrics (step, loss, recon_loss, sparsity_loss, etc.)
    """
    # New path structure: results/models/sae-{type}_{stage}_{layer}/training_logs.csv
    model_dir = results_dir / "models" / f"sae-{sae_type}_{stage}_{layer_quantile}"
    log_path = model_dir / "training_logs.csv"

    if not log_path.exists():
        # Try old path structure
        log_path = results_dir / stage / "training_logs.csv"

    if not log_path.exists():
        log_path = results_dir / stage / "sae_training_logs.csv"

    if not log_path.exists():
        raise FileNotFoundError(f"Training logs not found. Tried:\n"
                               f"  - {model_dir / 'training_logs.csv'}\n"
                               f"  - {results_dir / stage / 'training_logs.csv'}")

    df = pd.read_csv(log_path)
    return df


def load_sae_decoder_weights(
    results_dir: Path,
    stage: str = "pilot",
    sae_type: str = "gated",
    layer_quantile: str = "q2"
) -> torch.Tensor:
    """
    Load SAE decoder weights for UMAP visualization.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')
        sae_type: SAE type ('gated' or 'standard')
        layer_quantile: Layer quantile ('q1', 'q2', 'q3')

    Returns:
        Decoder weights tensor of shape [100000, 4096]
    """
    # New path structure: results/models/sae-{type}_{stage}_{layer}/model.pth
    model_dir = results_dir / "models" / f"sae-{sae_type}_{stage}_{layer_quantile}"
    model_path = model_dir / "model.pth"

    if not model_path.exists():
        # Try old path structure
        model_path = results_dir / stage / "sae_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"SAE model not found. Tried:\n"
                               f"  - {model_dir / 'model.pth'}\n"
                               f"  - {results_dir / stage / 'sae_model.pt'}")

    checkpoint = torch.load(model_path, map_location='cpu')

    # Extract decoder weights
    if 'decoder.weight' in checkpoint:
        decoder_weights = checkpoint['decoder.weight']
    elif 'model_state_dict' in checkpoint:
        decoder_weights = checkpoint['model_state_dict']['decoder.weight']
    elif 'state_dict' in checkpoint:
        decoder_weights = checkpoint['state_dict']['decoder.weight']
    else:
        # Try direct state dict format (from model.state_dict())
        if isinstance(checkpoint, dict):
            for key in checkpoint.keys():
                if 'decoder' in key and 'weight' in key:
                    decoder_weights = checkpoint[key]
                    break
            else:
                raise KeyError(f"Could not find decoder weights in checkpoint. Keys: {list(checkpoint.keys())}")
        else:
            raise KeyError("Could not find decoder weights in checkpoint")

    # Transpose to [num_features, latent_dim]
    if decoder_weights.shape[0] == 4096:
        decoder_weights = decoder_weights.T

    return decoder_weights


def load_activations(
    results_dir: Path,
    stage: str = "pilot",
    demographic: Optional[str] = None,
    layer_quantile: str = "q2"
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Load activation data from pickle file.

    Args:
        results_dir: Path to results directory
        stage: Data stage ('pilot', 'medium', or 'full')
        demographic: If specified, load from per-demographic directory
        layer_quantile: Layer quantile ('q1', 'q2', 'q3')

    Returns:
        activations: Tensor of activations
        labels: List of demographic labels
        prompts: List of prompts
    """
    if demographic:
        activation_path = results_dir / stage / demographic / "activations.pkl"
    else:
        activation_path = results_dir / stage / "activations.pkl"

    if not activation_path.exists():
        raise FileNotFoundError(f"Activations not found at {activation_path}")

    with open(activation_path, 'rb') as f:
        data = pickle.load(f)

    activations = data[f'{stage}_residual_{layer_quantile}']
    labels = data[f'{stage}_labels']
    prompts = data.get(f'{stage}_prompts', [f"prompt_{i}" for i in range(len(labels))])

    return activations, labels, prompts


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


def get_available_demographics(
    results_dir: Path,
    stage: str = "pilot"
) -> List[str]:
    """
    Get list of demographics that have results available.

    Args:
        results_dir: Path to results directory
        stage: Data stage

    Returns:
        List of demographic names with results
    """
    stage_dir = results_dir / stage
    demographics = []

    if stage_dir.exists():
        for subdir in stage_dir.iterdir():
            if subdir.is_dir() and subdir.name not in ['models', 'probe', 'ig2', 'verification']:
                # Check if this looks like a demographic directory
                if (subdir / 'activations.pkl').exists() or (subdir / 'ig2').exists():
                    demographics.append(subdir.name)

    return demographics
