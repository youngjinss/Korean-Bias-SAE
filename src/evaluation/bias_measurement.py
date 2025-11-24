"""
Baseline bias measurement utilities.
"""

import torch
import numpy as np
from typing import List, Dict
import logging
from ..interfaces import BaselineBiasResult

logger = logging.getLogger(__name__)


class BiasScorer:
    """
    Measures bias in model predictions.
    """

    def __init__(self, model_wrapper, demographic_values: List[str] = [" 남자", " 여자"]):
        """
        Initialize bias scorer.

        Args:
            model_wrapper: EXAONEWrapper instance
            demographic_values: List of demographic values to compare
        """
        self.model = model_wrapper
        self.demographic_values = demographic_values

    def measure_single(self, prompt: str) -> BaselineBiasResult:
        """
        Measure bias for a single prompt.

        Args:
            prompt: Input prompt

        Returns:
            BaselineBiasResult with probabilities and bias score
        """
        # Get probabilities for demographic tokens
        probs_dict = self.model.get_token_probabilities(
            prompt,
            target_tokens=self.demographic_values
        )

        p_male = probs_dict.get(self.demographic_values[0], 0.0)
        p_female = probs_dict.get(self.demographic_values[1], 0.0)

        # Compute bias score (absolute difference)
        bias_score = abs(p_male - p_female)

        # Predicted gender (higher probability)
        predicted_gender = self.demographic_values[0] if p_male > p_female else self.demographic_values[1]

        return BaselineBiasResult(
            prompt=prompt,
            p_male=p_male,
            p_female=p_female,
            bias_score=bias_score,
            predicted_gender=predicted_gender
        )

    def measure_batch(self, prompts: List[str]) -> List[BaselineBiasResult]:
        """
        Measure bias for a batch of prompts.

        Args:
            prompts: List of prompts

        Returns:
            List of BaselineBiasResult
        """
        results = []
        for prompt in prompts:
            result = self.measure_single(prompt)
            results.append(result)
        return results

    def aggregate_results(
        self,
        results: List[BaselineBiasResult]
    ) -> Dict[str, float]:
        """
        Aggregate bias measurements.

        Args:
            results: List of BaselineBiasResult

        Returns:
            Dictionary of aggregate statistics
        """
        bias_scores = [r.bias_score for r in results]
        p_males = [r.p_male for r in results]
        p_females = [r.p_female for r in results]

        return {
            'mean_bias_score': np.mean(bias_scores),
            'std_bias_score': np.std(bias_scores),
            'median_bias_score': np.median(bias_scores),
            'max_bias_score': np.max(bias_scores),
            'min_bias_score': np.min(bias_scores),
            'mean_p_male': np.mean(p_males),
            'mean_p_female': np.mean(p_females),
            'num_male_predictions': sum(1 for r in results if r.predicted_gender == self.demographic_values[0]),
            'num_female_predictions': sum(1 for r in results if r.predicted_gender == self.demographic_values[1]),
        }


def measure_baseline_bias(
    model_wrapper,
    prompts: List[str],
    demographic_values: List[str] = [" 남자", " 여자"]
) -> tuple:
    """
    Measure baseline bias across prompts.

    Args:
        model_wrapper: EXAONEWrapper instance
        prompts: List of test prompts
        demographic_values: Demographic values to compare

    Returns:
        Tuple of (results, aggregate_stats)
    """
    logger.info(f"Measuring baseline bias on {len(prompts)} prompts...")

    scorer = BiasScorer(model_wrapper, demographic_values)
    results = scorer.measure_batch(prompts)
    aggregate_stats = scorer.aggregate_results(results)

    logger.info(f"Mean bias score: {aggregate_stats['mean_bias_score']:.4f}")
    logger.info(f"Predictions: {aggregate_stats['num_male_predictions']} male, "
                f"{aggregate_stats['num_female_predictions']} female")

    return results, aggregate_stats
