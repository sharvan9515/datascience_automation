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
    code_blocks: dict[str, List[str]] = field(default_factory=dict)
    iteration_history: List[dict] = field(default_factory=list)
    best_score: Optional[float] = None
    no_improve_rounds: int = 0
    iteration: int = 0
    max_iter: int = 0

    def append_log(self, message: str) -> None:
        """Append a human-readable message to the pipeline log."""
        self.log.append(str(message))

    def append_code(self, stage: str, code: str) -> None:
        """Store code snippets for later assembly."""
        self.code_blocks.setdefault(stage, []).append(code)
