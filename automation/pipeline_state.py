from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

@dataclass
class PipelineState:
    df: pd.DataFrame
    target: str
    task_type: Optional[str] = None
    iterate: bool = False
    log: List[str] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    feature_formulas: dict[str, str] = field(default_factory=dict)
    feature_ideas: list[dict] = field(default_factory=list)
    known_features: set[str] = field(default_factory=set)
    pending_code: dict[str, list[str]] = field(default_factory=dict)
    code_blocks: dict[str, list[str]] = field(default_factory=dict)
    current_score: float | None = None
    iteration_history: list[dict] = field(default_factory=list)
    snippet_history: list[dict] = field(default_factory=list)
    best_score: Optional[float] = None
    best_df: Optional[pd.DataFrame] = None
    best_code_blocks: dict[str, list[str]] = field(default_factory=dict)
    best_features: List[str] = field(default_factory=list)
    best_params: dict[str, object] = field(default_factory=dict)
    patience: int = 5
    no_improve_rounds: int = 0
    iteration: int = 0
    max_iter: int = 0

    # Track all trained models for ensembling
    trained_models: list = field(default_factory=list)

    def append_log(self, message: str) -> None:
        """Append a human-readable message to the pipeline log."""
        self.log.append(str(message))

    def append_code(self, stage: str, code: str) -> None:
        """Store code snippets for later assembly."""
        self.code_blocks.setdefault(stage, []).append(code)

    def append_pending_code(self, stage: str, code: str) -> None:
        """Store code snippets awaiting validation or execution."""
        self.pending_code.setdefault(stage, []).append(code)

    def add_trained_model(self, model, name, model_type, predictions=None, score=None):
        """Add a trained model and its metadata for ensembling."""
        self.trained_models.append({
            'model': model,
            'name': name,
            'type': model_type,
            'predictions': predictions,
            'score': score
        })

    def get_trained_models(self):
        """Return all trained models and their metadata."""
        return self.trained_models
