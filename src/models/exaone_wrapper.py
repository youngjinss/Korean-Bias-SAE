"""
EXAONE model wrapper with utilities for feature extraction.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class EXAONEWrapper:
    """
    Wrapper for EXAONE model with convenient methods for:
    - Loading model and tokenizer
    - Extracting hidden states at specific layers
    - Getting token probabilities
    """

    def __init__(
        self,
        model_name: str = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        device: str = "cuda",
        dtype: str = "float16"
    ):
        """
        Initialize EXAONE wrapper.

        Args:
            model_name: Hugging Face model name
            device: Device to load model on ('cuda' or 'cpu')
            dtype: Model dtype ('float16' or 'float32')
        """
        self.model_name = model_name
        self.device = device
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        logger.info(f"Loading EXAONE model: {model_name}")
        logger.info(f"Device: {device}, dtype: {dtype}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info(f"Tokenizer loaded (vocab size: {len(self.tokenizer)})")

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=device if device == "auto" else None,
            trust_remote_code=True
        )

        if device != "auto":
            self.model = self.model.to(device)

        self.model.eval()  # Set to evaluation mode
        logger.info(f"Model loaded and moved to {device}")

        # Store config info
        self.config = self.model.config
        self.hidden_size = self.config.hidden_size
        self.num_layers = self.config.num_hidden_layers

    def tokenize(self, text: str, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Tokenize text.

        Args:
            text: Input text
            **kwargs: Additional arguments for tokenizer

        Returns:
            Dictionary with input_ids, attention_mask, etc.
        """
        inputs = self.tokenizer(text, return_tensors="pt", **kwargs)
        # Move to model device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def get_hidden_states(
        self,
        text: str,
        layer_idx: int = -1,
        token_position: str = "last"
    ) -> torch.Tensor:
        """
        Extract hidden states from specific layer and token position.

        Args:
            text: Input text
            layer_idx: Layer index (-1 for last layer)
            token_position: 'last', 'mean', or integer for specific position

        Returns:
            Hidden states tensor of shape (1, hidden_dim) or (1, seq_len, hidden_dim)
        """
        inputs = self.tokenize(text)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get hidden states from specified layer
        hidden_states = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)

        # Extract specific token position
        if token_position == "last":
            hidden = hidden_states[:, -1, :]  # (batch, hidden_dim)
        elif token_position == "mean":
            hidden = hidden_states.mean(dim=1)  # (batch, hidden_dim)
        elif isinstance(token_position, int):
            hidden = hidden_states[:, token_position, :]  # (batch, hidden_dim)
        else:
            raise ValueError(f"Invalid token_position: {token_position}")

        return hidden

    def get_hidden_states_at_position(
        self,
        text: Optional[str],
        layer_idx: int,
        token_position: int,
        tokens: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract hidden states at a specific token position.

        This method is designed for extracting activations at the position where
        the model generates a specific token (e.g., the answer token in bias prompts).

        Args:
            text: Input text (or None if tokens provided)
            layer_idx: Layer index to extract from
            token_position: Specific token position (integer index)
            tokens: Pre-tokenized input (optional, shape: (seq_len,))

        Returns:
            Hidden states at the specified position: (1, hidden_dim)

        Example:
            >>> # Extract activation at position where model generates "남자"
            >>> tokens, answer_pos = estimate_token_location(generated_text, "남자", tokenizer)
            >>> hidden = model.get_hidden_states_at_position(
            ...     text=None,
            ...     layer_idx=15,
            ...     token_position=answer_pos,
            ...     tokens=tokens
            ... )
        """
        if tokens is None:
            if text is None:
                raise ValueError("Either text or tokens must be provided")
            inputs = self.tokenize(text)
            tokens = inputs['input_ids'][0]  # Remove batch dimension
        else:
            # Ensure tokens is on correct device and has batch dimension
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            inputs = {'input_ids': tokens.to(self.device)}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # Get hidden states from specified layer
        hidden_states = outputs.hidden_states[layer_idx]  # (batch, seq_len, hidden_dim)

        # Extract specific token position
        hidden = hidden_states[:, token_position, :]  # (batch, hidden_dim)

        return hidden

    def get_token_logits(
        self,
        text: str,
        target_tokens: list
    ) -> Dict[str, float]:
        """
        Get logits for target tokens.

        Args:
            text: Input text
            target_tokens: List of token strings to get logits for

        Returns:
            Dictionary mapping token -> logit value
        """
        inputs = self.tokenize(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits (batch, vocab_size)

        # Extract logits for target tokens
        result = {}
        for token in target_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 0:
                logger.warning(f"Token '{token}' could not be encoded")
                result[token] = float('-inf')
            else:
                token_id = token_ids[0]
                result[token] = logits[0, token_id].item()

        return result

    def get_token_probabilities(
        self,
        text: str,
        target_tokens: list
    ) -> Dict[str, float]:
        """
        Get probability distribution over target tokens.

        Args:
            text: Input text
            target_tokens: List of token strings to get probabilities for

        Returns:
            Dictionary mapping token -> probability
        """
        inputs = self.tokenize(text)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Last token logits (batch, vocab_size)

        # Get probabilities
        probs = torch.softmax(logits, dim=-1)

        # Extract probabilities for target tokens
        result = {}
        for token in target_tokens:
            token_ids = self.tokenizer.encode(token, add_special_tokens=False)
            if len(token_ids) == 0:
                logger.warning(f"Token '{token}' could not be encoded")
                result[token] = 0.0
            else:
                token_id = token_ids[0]
                result[token] = probs[0, token_id].item()

        return result

    def generate(
        self,
        text: str,
        max_new_tokens: int = 5,
        **kwargs
    ) -> str:
        """
        Generate text continuation.

        Args:
            text: Input prompt
            max_new_tokens: Maximum tokens to generate
            **kwargs: Additional generation parameters

        Returns:
            Generated text (full, including prompt)
        """
        inputs = self.tokenize(text)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                **kwargs
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def extract_response(self, prompt: str, generated_text: str) -> str:
        """
        Extract only the generated response (remove prompt).

        Args:
            prompt: Original prompt
            generated_text: Full generated text including prompt

        Returns:
            Only the generated response
        """
        if generated_text.startswith(prompt):
            return generated_text[len(prompt):].strip()
        return generated_text.strip()

    def __repr__(self):
        return f"EXAONEWrapper(model={self.model_name}, device={self.device})"
