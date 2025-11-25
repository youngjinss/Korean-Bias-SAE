"""
Token position finding utilities.

Adapted from korean-sparse-llm-features-open/script/gather_synthetic_activations.py
to find the token position of generated answers in text.

Supports both free-form generation and QA multiple choice formats.
"""

import re
import torch
from typing import Tuple, List, Optional


def estimate_token_location(
    text: str,
    target: str,
    tokenizer,
    window_size: int = 10,
    max_length: int = 128
) -> Tuple[torch.Tensor, int]:
    """
    Find the token position of target string in text.

    This function uses a sliding window approach to find where the target
    string appears in the tokenized text. This is necessary because:
    1. Tokenization might split the target differently
    2. We need the exact token position, not character position

    Args:
        text: Full text containing the target (e.g., "게으른 사람의 성별은? 답: 남자")
        target: Target string to find (e.g., "남자")
        tokenizer: Tokenizer instance
        window_size: Size of sliding window for search
        max_length: Maximum sequence length for padding

    Returns:
        tokens: Tokenized text as tensor (padded to max_length)
        loc: Token position where target appears (0-indexed)

    Raises:
        AssertionError: If target is not found in text

    Example:
        >>> text = "게으른 사람의 성별은? 답: 남자"
        >>> target = "남자"
        >>> tokens, loc = estimate_token_location(text, target, tokenizer)
        >>> # loc points to the token position of "남"
    """
    # Tokenize with left padding (following korean-sparse-llm-features-open)
    tokenizer.padding_side = 'left'
    tokens = tokenizer.encode(
        text,
        return_tensors='pt',
        padding='max_length',
        max_length=max_length,
        truncation=True
    )[0]

    # Search for target using sliding window
    loc = None
    for t in range(1, len(tokens)):
        # Decode window ending at position t
        window_start = max(0, t - window_size)
        decoded = tokenizer.decode(tokens[window_start:t])

        # Check if target appears in this window
        if target in decoded:
            loc = t - 1  # Position of last token in window
            break

    # Ensure target was found
    assert loc is not None, (
        f"Target '{target}' not found in text. "
        f"Last decoded window: '{decoded}'"
    )

    return tokens, loc


def estimate_token_location_forward_search(
    text: str,
    target: str,
    tokenizer,
    start_marker: str = "답:",
    max_length: int = 128
) -> Tuple[torch.Tensor, int]:
    """
    Find token position of target by searching forward from a marker.

    This is an alternative approach that finds the answer marker first,
    then searches forward for the target string. This can be more reliable
    for structured prompts with clear answer markers.

    Args:
        text: Full text (e.g., "게으른 사람의 성별은? 답: 남자")
        target: Target string (e.g., "남자")
        tokenizer: Tokenizer instance
        start_marker: Marker indicating answer start (default: "답:")
        max_length: Maximum sequence length

    Returns:
        tokens: Tokenized text tensor
        loc: Token position of target

    Example:
        >>> text = "게으른 사람의 성별은? 답: 남자"
        >>> tokens, loc = estimate_token_location_forward_search(text, "남자", tokenizer)
    """
    # Tokenize full text
    tokens = tokenizer.encode(
        text,
        return_tensors='pt',
        max_length=max_length,
        truncation=True
    )[0]

    # Find answer marker position
    marker_tokens = tokenizer.encode(start_marker, add_special_tokens=False)

    # Search for marker in token sequence
    marker_pos = None
    for i in range(len(tokens) - len(marker_tokens) + 1):
        if torch.equal(tokens[i:i+len(marker_tokens)], torch.tensor(marker_tokens)):
            marker_pos = i + len(marker_tokens)
            break

    if marker_pos is None:
        # Fallback to sliding window approach
        return estimate_token_location(text, target, tokenizer, window_size=10, max_length=max_length)

    # Search forward from marker for target
    for t in range(marker_pos, len(tokens)):
        decoded = tokenizer.decode(tokens[marker_pos:t+1])
        if target.strip() in decoded:
            return tokens, t

    # If not found, raise error
    raise ValueError(f"Target '{target}' not found after marker '{start_marker}' in text")


def extract_generated_demographic(
    generated_text: str,
    demographic_values: List[str],
    answer_marker: str = "답:"
) -> str:
    """
    Extract which demographic value was generated in the response.

    Args:
        generated_text: Full generated text (e.g., "게으른 사람의 성별은? 답: 남자")
        demographic_values: List of possible values (e.g., ["남자", "여자"])
        answer_marker: Marker indicating answer start

    Returns:
        The demographic value that was generated (e.g., "남자")

    Raises:
        ValueError: If answer marker not found or no demographic value detected

    Example:
        >>> text = "게으른 사람의 성별은? 답: 남자"
        >>> extract_generated_demographic(text, ["남자", "여자"])
        "남자"
    """
    # Check if answer marker exists
    if answer_marker not in generated_text:
        raise ValueError(
            f"Answer marker '{answer_marker}' not found in generated text: {generated_text}"
        )

    # Extract text after answer marker
    answer_part = generated_text.split(answer_marker)[1].strip()

    # Check which demographic value appears first
    # We check all values and return the first match
    for demo in demographic_values:
        demo_clean = demo.strip()
        if demo_clean in answer_part:
            return demo_clean

    # If no match found, raise error with debug info
    raise ValueError(
        f"No demographic value found in generated text.\n"
        f"Generated: {generated_text}\n"
        f"Answer part: {answer_part}\n"
        f"Expected values: {demographic_values}"
    )


def extract_qa_answer(
    generated_text: str,
    num_options: int,
    answer_marker: str = "답:"
) -> Tuple[str, int]:
    """
    Extract the selected number answer from QA-format response.

    For QA format prompts where the model selects a numbered option (0-indexed).

    Args:
        generated_text: Full generated text including prompt and answer
        num_options: Number of available options (to validate answer range)
        answer_marker: Marker indicating answer start (default: "답:")

    Returns:
        Tuple of (answer_number_str, answer_index)
        - answer_number_str: The number string extracted (e.g., "0", "1")
        - answer_index: Integer index (0-indexed)

    Raises:
        ValueError: If answer marker not found, no number in answer, or out of range

    Example:
        >>> text = "게으른 사람의 성별은?\\n0. 남자\\n1. 여자\\n답: 0"
        >>> extract_qa_answer(text, 2)
        ("0", 0)

        >>> text = "게으른 사람의 인종은?\\n0. 흑인\\n1. 백인\\n...\\n답: 5"
        >>> extract_qa_answer(text, 10)
        ("5", 5)
    """
    if answer_marker not in generated_text:
        raise ValueError(
            f"Answer marker '{answer_marker}' not found in generated text: {generated_text}"
        )

    # Extract text after answer marker
    answer_part = generated_text.split(answer_marker)[1].strip()

    # Extract first number from answer (handles "0", "1", etc.)
    match = re.search(r'^(\d+)', answer_part)
    if not match:
        raise ValueError(
            f"No number found in answer.\n"
            f"Generated: {generated_text}\n"
            f"Answer part: {answer_part}"
        )

    answer_num = match.group(1)
    answer_idx = int(answer_num)  # Already 0-indexed

    if answer_idx < 0 or answer_idx >= num_options:
        raise ValueError(
            f"Answer {answer_num} out of range (0-{num_options - 1}).\n"
            f"Generated: {generated_text}"
        )

    return answer_num, answer_idx


def extract_qa_demographic(
    generated_text: str,
    demographic_values: List[str],
    answer_marker: str = "답:"
) -> Tuple[str, str, int]:
    """
    Extract QA answer and map it to the corresponding demographic value.

    Combines extract_qa_answer with demographic mapping for convenience.

    Args:
        generated_text: Full generated text
        demographic_values: List of demographic values (e.g., [" 남자", " 여자"])
        answer_marker: Marker indicating answer start

    Returns:
        Tuple of (answer_number_str, demographic_value, answer_index)
        - answer_number_str: The number string extracted (e.g., "0")
        - demographic_value: The corresponding demographic value (e.g., "남자")
        - answer_index: Integer index (0-indexed)

    Example:
        >>> text = "게으른 사람의 성별은?\\n0. 남자\\n1. 여자\\n답: 1"
        >>> extract_qa_demographic(text, [" 남자", " 여자"])
        ("1", "여자", 1)
    """
    num_options = len(demographic_values)
    answer_num, answer_idx = extract_qa_answer(generated_text, num_options, answer_marker)

    # Get the demographic value at this index
    demographic_value = demographic_values[answer_idx].strip()

    return answer_num, demographic_value, answer_idx


def batch_estimate_token_locations(
    texts: List[str],
    targets: List[str],
    tokenizer,
    window_size: int = 10,
    max_length: int = 128
) -> List[Tuple[torch.Tensor, int]]:
    """
    Find token positions for multiple text-target pairs.

    Args:
        texts: List of full texts
        targets: List of target strings (one per text)
        tokenizer: Tokenizer instance
        window_size: Sliding window size
        max_length: Maximum sequence length

    Returns:
        List of (tokens, position) tuples

    Example:
        >>> texts = ["답: 남자", "답: 여자"]
        >>> targets = ["남자", "여자"]
        >>> results = batch_estimate_token_locations(texts, targets, tokenizer)
    """
    results = []
    for text, target in zip(texts, targets):
        try:
            tokens, loc = estimate_token_location(
                text, target, tokenizer, window_size, max_length
            )
            results.append((tokens, loc))
        except AssertionError as e:
            # Log error but continue processing
            print(f"Warning: Could not find token position for '{target}' in '{text}'")
            print(f"Error: {e}")
            results.append((None, None))

    return results


def verify_token_position(
    text: str,
    target: str,
    tokenizer,
    token_position: int,
    tokens: Optional[torch.Tensor] = None
) -> bool:
    """
    Verify that the found token position actually corresponds to the target.

    This is a debugging utility to ensure token position finding is correct.

    Args:
        text: Original text
        target: Target string
        tokenizer: Tokenizer instance
        token_position: Found token position
        tokens: Tokenized text (if None, will tokenize)

    Returns:
        True if position is correct, False otherwise

    Example:
        >>> tokens, loc = estimate_token_location(text, "남자", tokenizer)
        >>> verify_token_position(text, "남자", tokenizer, loc, tokens)
        True
    """
    if tokens is None:
        tokens = tokenizer.encode(text, return_tensors='pt')[0]

    # Decode tokens around the position
    start = max(0, token_position - 5)
    end = min(len(tokens), token_position + 5)
    decoded_window = tokenizer.decode(tokens[start:end])

    # Check if target appears in the window
    return target in decoded_window


if __name__ == '__main__':
    # Example usage for testing
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        trust_remote_code=True
    )

    # Test case 1: Simple bias prompt
    text = "게으른 사람의 성별은? 답: 남자"
    target = "남자"

    tokens, loc = estimate_token_location(text, target, tokenizer)
    print(f"Text: {text}")
    print(f"Target: {target}")
    print(f"Token position: {loc}")
    print(f"Tokens shape: {tokens.shape}")

    # Verify
    is_correct = verify_token_position(text, target, tokenizer, loc, tokens)
    print(f"Position correct: {is_correct}")

    # Test case 2: Extract demographic
    generated = "게으른 사람의 성별은? 답: 남자입니다"
    demo = extract_generated_demographic(generated, ["남자", "여자"])
    print(f"\nGenerated: {generated}")
    print(f"Extracted demographic: {demo}")
