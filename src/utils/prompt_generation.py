"""
Utilities for generating bias prompts from templates and modifiers.
"""

import json
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def load_modifiers(stage: str, data_dir: str = "data") -> Dict[str, List[str]]:
    """
    Load modifiers for a given stage.

    Args:
        stage: 'pilot', 'medium', or 'full'
        data_dir: Path to data directory

    Returns:
        Dictionary with 'negative' and 'positive' modifier lists
    """
    data_path = Path(data_dir) / "modifiers"

    negative_file = data_path / f"{stage}_negative_ko.json"
    positive_file = data_path / f"{stage}_positive_ko.json"

    # Load files
    with open(negative_file, 'r', encoding='utf-8') as f:
        negative = json.load(f)

    with open(positive_file, 'r', encoding='utf-8') as f:
        positive = json.load(f)

    return {
        'negative': negative,
        'positive': positive
    }


def load_templates(stage: str, data_dir: str = "data") -> List[str]:
    """
    Load templates for a given stage.

    Args:
        stage: 'pilot', 'medium', or 'full'
        data_dir: Path to data directory

    Returns:
        List of template strings
    """
    templates_file = Path(data_dir) / "templates" / "korean_templates.json"

    with open(templates_file, 'r', encoding='utf-8') as f:
        templates_data = json.load(f)

    key = f"{stage}_templates"
    if key not in templates_data:
        raise ValueError(f"No templates found for stage '{stage}'")

    return templates_data[key]


def generate_prompts(
    modifiers: List[str],
    templates: List[str],
    demographic_dimension: str = "성별"
) -> List[str]:
    """
    Generate prompts by filling in templates with modifiers.

    Args:
        modifiers: List of modifier words (e.g., ["게으른", "똑똑한"])
        templates: List of template strings with {Modifier} and {Demographic_Dimension}
        demographic_dimension: Demographic dimension (default: "성별")

    Returns:
        List of generated prompts
    """
    prompts = []

    for modifier in modifiers:
        for template in templates:
            prompt = template.replace("{Modifier}", modifier)
            prompt = prompt.replace("{Demographic_Dimension}", demographic_dimension)
            prompts.append(prompt)

    return prompts


def generate_stage_prompts(
    stage: str,
    data_dir: str = "data",
    demographic_dimension: str = "성별",
    include_negative: bool = True,
    include_positive: bool = True
) -> List[str]:
    """
    Generate all prompts for a given stage.

    Args:
        stage: 'pilot', 'medium', or 'full'
        data_dir: Path to data directory
        demographic_dimension: Demographic dimension (default: "성별")
        include_negative: Include negative modifiers
        include_positive: Include positive modifiers

    Returns:
        List of all generated prompts
    """
    logger.info(f"Generating prompts for stage: {stage}")

    # Load modifiers and templates
    modifiers_dict = load_modifiers(stage, data_dir)
    templates = load_templates(stage, data_dir)

    all_prompts = []

    if include_negative:
        negative_prompts = generate_prompts(
            modifiers_dict['negative'],
            templates,
            demographic_dimension
        )
        all_prompts.extend(negative_prompts)
        logger.info(f"  Generated {len(negative_prompts)} negative prompts")

    if include_positive:
        positive_prompts = generate_prompts(
            modifiers_dict['positive'],
            templates,
            demographic_dimension
        )
        all_prompts.extend(positive_prompts)
        logger.info(f"  Generated {len(positive_prompts)} positive prompts")

    logger.info(f"  Total prompts: {len(all_prompts)}")

    return all_prompts
