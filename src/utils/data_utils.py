"""
Data utilities: loading, saving, preprocessing.
"""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any


def save_jsonl(data: List[Dict[str, Any]], filepath: str):
    """
    Save list of dictionaries to JSONL file.

    Args:
        data: List of dictionaries
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(filepath, 'w') as writer:
        for item in data:
            writer.write(item)


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file to list of dictionaries.

    Args:
        filepath: Input file path

    Returns:
        List of dictionaries
    """
    data = []
    with jsonlines.open(filepath, 'r') as reader:
        for item in reader:
            data.append(item)
    return data


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file.

    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file to dictionary.

    Args:
        filepath: Input file path

    Returns:
        Dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
