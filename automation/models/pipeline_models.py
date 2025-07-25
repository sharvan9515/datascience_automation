from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    """Structured representation of dataset profiling results."""

    statistical_summary: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    missing_patterns: Dict[str, Any] = Field(default_factory=dict)
    outlier_detection: Dict[str, int] = Field(default_factory=dict)
    skewness: Dict[str, float] = Field(default_factory=dict)
    correlations: Dict[str, Any] = Field(default_factory=dict)
    domain_insights: Dict[str, Any] = Field(default_factory=dict)
    complexity_metrics: Dict[str, Any] = Field(default_factory=dict)


class PreprocessingPlan(BaseModel):
    """Instructions and code for preprocessing the dataset."""

    logs: List[str] = Field(default_factory=list)
    code: str = ""
    rationale: str = ""
    dropped_columns: List[str] = Field(default_factory=list)
    encoded_columns: Dict[str, str] = Field(default_factory=dict)


class FeatureEngineeringPlan(BaseModel):
    """LLM-proposed features and implementation code."""

    features: List[Dict[str, str]] = Field(default_factory=list)
    code: str = ""
    rationale: str = ""


class ModelTrainingPlan(BaseModel):
    """Model choice and training parameters."""

    algorithm: str = ""
    parameters: Dict[str, Any] = Field(default_factory=dict)
    cross_validation_folds: int = 5
    metrics: Dict[str, float] = Field(default_factory=dict)


class PipelineDecision(BaseModel):
    """Decision from the orchestrator about the next pipeline step."""

    proceed: bool = True
    reason: Optional[str] = None
    next_step: Optional[str] = None


class FinalPipeline(BaseModel):
    """Assembled pipeline code and final model results."""

    code_blocks: Dict[str, List[str]] = Field(default_factory=dict)
    model_path: Optional[str] = None
    score: Optional[float] = None
    logs: List[str] = Field(default_factory=list)
