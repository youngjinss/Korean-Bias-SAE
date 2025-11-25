"""
Utilities for handling multiple demographic categories.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def load_demographic_dict(data_dir: str = "data") -> Dict:
    """
    Load the Korean demographic dictionary.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary mapping Korean demographic dimensions to their values
    """
    dict_file = Path(data_dir) / "demographic_dict_ko.json"

    if not dict_file.exists():
        raise FileNotFoundError(
            f"Demographic dictionary not found at {dict_file}. "
            "Please ensure data/demographic_dict_ko.json exists."
        )

    with open(dict_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_demographic_values(
    demographic_dimension: str,
    data_dir: str = "data"
) -> List[str]:
    """
    Get the demographic values for a given dimension.

    Args:
        demographic_dimension: Korean demographic dimension (e.g., "성별", "인종")
        data_dir: Path to data directory

    Returns:
        List of demographic values with leading spaces

    Raises:
        ValueError: If demographic dimension is not found

    Examples:
        >>> get_demographic_values("성별")
        [' 남자', ' 여자']
        >>> get_demographic_values("나이")
        [' 젊은', ' 늙은', ' 십대', ' 중년']
    """
    demo_dict = load_demographic_dict(data_dir)

    if demographic_dimension not in demo_dict:
        available = list(demo_dict.keys())
        raise ValueError(
            f"Unknown demographic dimension: '{demographic_dimension}'. "
            f"Available options: {available}"
        )

    return demo_dict[demographic_dimension]["values"]


def validate_demographic_config(
    demographic_dimension: str,
    demographic_values: List[str],
    data_dir: str = "data"
) -> Tuple[bool, str]:
    """
    Validate that demographic configuration is consistent.

    Args:
        demographic_dimension: Korean demographic dimension
        demographic_values: List of demographic values from config
        data_dir: Path to data directory

    Returns:
        Tuple of (is_valid, message)
    """
    try:
        expected_values = get_demographic_values(demographic_dimension, data_dir)
    except ValueError as e:
        return False, str(e)

    # Check if provided values are subset of expected values
    for value in demographic_values:
        if value not in expected_values:
            return False, (
                f"Invalid demographic value '{value}' for dimension '{demographic_dimension}'. "
                f"Expected values: {expected_values}"
            )

    return True, f"Valid configuration for '{demographic_dimension}'"


def get_all_demographics(data_dir: str = "data") -> Dict[str, Dict]:
    """
    Get all available demographic categories.

    Args:
        data_dir: Path to data directory

    Returns:
        Dictionary of all demographics with their info
    """
    return load_demographic_dict(data_dir)


def format_demographic_info(demographic_dimension: str, data_dir: str = "data") -> str:
    """
    Format demographic information for display.

    Args:
        demographic_dimension: Korean demographic dimension
        data_dir: Path to data directory

    Returns:
        Formatted string with demographic info
    """
    demo_dict = load_demographic_dict(data_dir)

    if demographic_dimension not in demo_dict:
        return f"Unknown demographic: {demographic_dimension}"

    info = demo_dict[demographic_dimension]
    values_str = ", ".join([f"'{v.strip()}'" for v in info["values"]])

    return (
        f"Demographic: {demographic_dimension} ({info['dimension_en']})\n"
        f"  Values: {values_str}\n"
        f"  Count: {len(info['values'])}"
    )


def get_demographic_mask(
    demographic_dimension: str,
    max_output_dim: int = 10,
    data_dir: str = "data"
) -> List[bool]:
    """
    Get a boolean mask for valid demographic value positions.

    Used for masking out invalid positions in the probe output when
    the demographic has fewer values than max_output_dim.

    Args:
        demographic_dimension: Korean demographic dimension
        max_output_dim: Maximum output dimension (default: 10)
        data_dir: Path to data directory

    Returns:
        Boolean mask of length max_output_dim
        True = valid position, False = invalid position

    Examples:
        >>> get_demographic_mask("성별", max_output_dim=10)
        [True, True, False, False, False, False, False, False, False, False]
        >>> get_demographic_mask("나이", max_output_dim=10)
        [True, True, True, True, False, False, False, False, False, False]
        >>> get_demographic_mask("인종", max_output_dim=10)
        [True, True, True, True, True, True, True, True, True, True]
    """
    values = get_demographic_values(demographic_dimension, data_dir)
    num_values = len(values)

    if num_values > max_output_dim:
        raise ValueError(
            f"Demographic '{demographic_dimension}' has {num_values} values, "
            f"which exceeds max_output_dim ({max_output_dim})"
        )

    # Create mask: True for valid positions, False for padding
    mask = [True] * num_values + [False] * (max_output_dim - num_values)
    return mask


def get_num_active_classes(config: Dict) -> int:
    """
    Get the number of active (valid) classes for the current demographic.

    Args:
        config: Experiment configuration dictionary

    Returns:
        Number of demographic values (active classes)

    Examples:
        >>> config = {'data': {'demographic_values': [' 남자', ' 여자']}}
        >>> get_num_active_classes(config)
        2
    """
    return len(config['data']['demographic_values'])


def get_demographic_info(
    demographic_dimension: str,
    data_dir: str = "data"
) -> Dict:
    """
    Get complete information for a demographic dimension.

    Args:
        demographic_dimension: Korean demographic dimension (e.g., "성별", "인종")
        data_dir: Path to data directory

    Returns:
        Dictionary with keys: 'dimension_en', 'values', 'values_en'

    Examples:
        >>> info = get_demographic_info("성별")
        >>> info['values']
        [' 남자', ' 여자']
        >>> info['dimension_en']
        'gender'
    """
    demo_dict = load_demographic_dict(data_dir)

    if demographic_dimension not in demo_dict:
        available = list(demo_dict.keys())
        raise ValueError(
            f"Unknown demographic dimension: '{demographic_dimension}'. "
            f"Available options: {available}"
        )

    return demo_dict[demographic_dimension]
