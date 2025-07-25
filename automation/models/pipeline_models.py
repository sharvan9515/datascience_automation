from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DatasetProfile:
    """Structured representation of dataset profiling results."""

    statistical_summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    missing_patterns: Dict[str, Any] = field(default_factory=dict)
    outlier_detection: Dict[str, int] = field(default_factory=dict)
    skewness: Dict[str, float] = field(default_factory=dict)
    correlations: Dict[str, Any] = field(default_factory=dict)
    domain_insights: Dict[str, Any] = field(default_factory=dict)
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PreprocessingPlan:
    """Instructions and code for preprocessing the dataset."""

    logs: List[str] = field(default_factory=list)
    code: str = ""
    rationale: str = ""
    dropped_columns: List[str] = field(default_factory=list)
    encoded_columns: Dict[str, str] = field(default_factory=dict)


@dataclass
class FeatureEngineeringPlan:
    """LLM-proposed features and implementation code."""

    features: List[Dict[str, str]] = field(default_factory=list)
    code: str = ""
    rationale: str = ""


@dataclass
class ModelTrainingPlan:
    """Model choice and training parameters."""

    algorithm: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    cross_validation_folds: int = 5
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class PipelineDecision:
    """Decision from the orchestrator about the next pipeline step."""

    proceed: bool = True
    reason: Optional[str] = None
    next_step: Optional[str] = None


@dataclass
class FinalPipeline:
    """Assembled pipeline code and final model results."""

    code_blocks: Dict[str, List[str]] = field(default_factory=dict)
    model_path: Optional[str] = None
    score: Optional[float] = None
    logs: List[str] = field(default_factory=list)

