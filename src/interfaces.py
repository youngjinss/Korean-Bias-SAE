"""
Data interfaces and contracts for the bias detection pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import torch


@dataclass
class BiasPrompt:
    """Standard format for bias prompts"""
    prompt: str
    modifier: str
    modifier_type: str  # N, Ner, Nest, P, Per, Pest
    demographic_dimension: str  # 성별
    demographic_pair: Tuple[str, str]  # (남자, 여자)
    template_id: int
    jut_id: str  # Gender-N, Gender-P, etc.

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'prompt': self.prompt,
            'modifier': self.modifier,
            'modifier_type': self.modifier_type,
            'demographic_dimension': self.demographic_dimension,
            'demographic_pair': list(self.demographic_pair),
            'template_id': self.template_id,
            'jut_id': self.jut_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BiasPrompt':
        """Create from dictionary"""
        return cls(
            prompt=data['prompt'],
            modifier=data['modifier'],
            modifier_type=data['modifier_type'],
            demographic_dimension=data['demographic_dimension'],
            demographic_pair=tuple(data['demographic_pair']),
            template_id=data['template_id'],
            jut_id=data['jut_id']
        )


@dataclass
class SAEFeatures:
    """SAE feature activations with metadata"""
    features: torch.Tensor  # (batch, feature_dim) e.g., (32, 100000)
    layer_idx: int
    token_position: str  # 'last', 'mean', or 'specific'
    prompt_ids: List[str] = field(default_factory=list)

    def save(self, path: str):
        """Save features to disk"""
        torch.save({
            'features': self.features,
            'layer_idx': self.layer_idx,
            'token_position': self.token_position,
            'prompt_ids': self.prompt_ids
        }, path)

    @classmethod
    def load(cls, path: str) -> 'SAEFeatures':
        """Load features from disk"""
        data = torch.load(path)
        return cls(
            features=data['features'],
            layer_idx=data['layer_idx'],
            token_position=data['token_position'],
            prompt_ids=data.get('prompt_ids', [])
        )


@dataclass
class IG2Result:
    """IG² attribution results"""
    feature_scores: torch.Tensor  # (feature_dim,) e.g., (100000,)
    bias_features: torch.Tensor  # Indices of top features
    threshold: float
    metadata: Dict = field(default_factory=dict)

    def save(self, path: str):
        """Save results to disk"""
        torch.save({
            'feature_scores': self.feature_scores,
            'bias_features': self.bias_features,
            'threshold': self.threshold,
            'metadata': self.metadata
        }, path)

    @classmethod
    def load(cls, path: str) -> 'IG2Result':
        """Load results from disk"""
        data = torch.load(path)
        return cls(
            feature_scores=data['feature_scores'],
            bias_features=data['bias_features'],
            threshold=data['threshold'],
            metadata=data.get('metadata', {})
        )


@dataclass
class VerificationResult:
    """Suppression/amplification test results"""
    gap_before: float
    gap_after: float
    gap_change_ratio: float
    feature_indices: List[int]
    manipulation_type: str  # suppress, amplify, random
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'gap_before': self.gap_before,
            'gap_after': self.gap_after,
            'gap_change_ratio': self.gap_change_ratio,
            'feature_indices': self.feature_indices,
            'manipulation_type': self.manipulation_type,
            'metadata': self.metadata
        }


@dataclass
class BaselineBiasResult:
    """Results from baseline bias measurement"""
    prompt: str
    p_male: float
    p_female: float
    bias_score: float  # P(max_logit) - P(min_logit)
    predicted_gender: str  # Predicted demographic value

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'prompt': self.prompt,
            'p_male': self.p_male,
            'p_female': self.p_female,
            'bias_score': self.bias_score,
            'predicted_gender': self.predicted_gender
        }
