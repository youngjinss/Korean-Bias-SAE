"""
Utilities for generating bias prompts from templates and modifiers.

Supports both original generation format and QA multiple choice format.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def format_options(demographic_values: List[str]) -> str:
    """
    Format demographic values as numbered options (0-indexed).

    Args:
        demographic_values: List of values like [" 남자", " 여자"]

    Returns:
        Formatted string like "0. 남자\n1. 여자"

    Example:
        >>> format_options([" 남자", " 여자"])
        "0. 남자\\n1. 여자"
    """
    options = []
    for i, value in enumerate(demographic_values):  # 0-indexed
        options.append(f"{i}. {value.strip()}")
    return "\n".join(options)


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


def load_templates(stage: str, data_dir: str = "data", qa_format: bool = False) -> List[str]:
    """
    Load templates for a given stage.

    Args:
        stage: 'pilot', 'medium', or 'full'
        data_dir: Path to data directory
        qa_format: If True, load QA format templates with {Options} placeholder

    Returns:
        List of template strings
    """
    templates_file = Path(data_dir) / "templates" / "korean_templates.json"

    with open(templates_file, 'r', encoding='utf-8') as f:
        templates_data = json.load(f)

    suffix = "_qa" if qa_format else ""
    key = f"{stage}_templates{suffix}"

    if key not in templates_data:
        raise ValueError(f"No templates found for key '{key}'")

    return templates_data[key]


def generate_prompts(
    modifiers: List[str],
    templates: List[str],
    demographic_dimension: str = "성별",
    demographic_values: Optional[List[str]] = None,
    qa_format: bool = False
) -> List[str]:
    """
    Generate prompts by filling in templates with modifiers.

    Args:
        modifiers: List of modifier words (e.g., ["게으른", "똑똑한"])
        templates: List of template strings with {Modifier} and {Demographic_Dimension}
        demographic_dimension: Demographic dimension (default: "성별")
        demographic_values: List of possible values (required for qa_format)
        qa_format: If True, also replace {Options} placeholder

    Returns:
        List of generated prompts
    """
    prompts = []

    # Pre-compute options string if QA format
    options_str = None
    if qa_format:
        if demographic_values is None:
            raise ValueError("demographic_values required for QA format")
        options_str = format_options(demographic_values)

    for modifier in modifiers:
        for template in templates:
            prompt = template.replace("{Modifier}", modifier)
            prompt = prompt.replace("{Demographic_Dimension}", demographic_dimension)

            # Replace {Options} placeholder for QA format
            if qa_format and options_str is not None:
                prompt = prompt.replace("{Options}", options_str)

            prompts.append(prompt)

    return prompts


def generate_qa_prompts(
    modifiers: List[str],
    templates: List[str],
    demographic_dimension: str,
    demographic_values: List[str]
) -> List[str]:
    """
    Generate QA-format prompts with numbered options.

    This is a convenience wrapper around generate_prompts with qa_format=True.

    Args:
        modifiers: List of modifier words (e.g., ["게으른", "똑똑한"])
        templates: List of QA template strings with {Options} placeholder
        demographic_dimension: Demographic dimension (e.g., "성별")
        demographic_values: List of possible values (e.g., [" 남자", " 여자"])

    Returns:
        List of generated QA prompts

    Example:
        >>> templates = ["{Modifier} 사람의 {Demographic_Dimension}은?\\n{Options}\\n답:"]
        >>> prompts = generate_qa_prompts(
        ...     ["게으른"], templates, "성별", [" 남자", " 여자"]
        ... )
        >>> print(prompts[0])
        게으른 사람의 성별은?
        0. 남자
        1. 여자
        답:
    """
    return generate_prompts(
        modifiers=modifiers,
        templates=templates,
        demographic_dimension=demographic_dimension,
        demographic_values=demographic_values,
        qa_format=True
    )


def generate_stage_prompts(
    stage: str,
    data_dir: str = "data",
    demographic_dimension: str = "성별",
    demographic_values: Optional[List[str]] = None,
    include_negative: bool = True,
    include_positive: bool = True,
    qa_format: bool = False
) -> List[str]:
    """
    Generate all prompts for a given stage.

    Args:
        stage: 'pilot', 'medium', or 'full'
        data_dir: Path to data directory
        demographic_dimension: Demographic dimension (default: "성별")
        demographic_values: List of possible values (required for qa_format)
        include_negative: Include negative modifiers
        include_positive: Include positive modifiers
        qa_format: If True, use QA format templates with numbered options

    Returns:
        List of all generated prompts
    """
    logger.info(f"Generating prompts for stage: {stage} (QA format: {qa_format})")

    # Load modifiers and templates
    modifiers_dict = load_modifiers(stage, data_dir)
    templates = load_templates(stage, data_dir, qa_format=qa_format)

    all_prompts = []

    if include_negative:
        negative_prompts = generate_prompts(
            modifiers_dict['negative'],
            templates,
            demographic_dimension,
            demographic_values=demographic_values,
            qa_format=qa_format
        )
        all_prompts.extend(negative_prompts)
        logger.info(f"  Generated {len(negative_prompts)} negative prompts")

    if include_positive:
        positive_prompts = generate_prompts(
            modifiers_dict['positive'],
            templates,
            demographic_dimension,
            demographic_values=demographic_values,
            qa_format=qa_format
        )
        all_prompts.extend(positive_prompts)
        logger.info(f"  Generated {len(positive_prompts)} positive prompts")

    logger.info(f"  Total prompts: {len(all_prompts)}")

    return all_prompts
