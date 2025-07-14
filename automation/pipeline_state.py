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

    def append_log(self, message: str) -> None:
        """Append a human-readable message to the pipeline log."""
        self.log.append(str(message))
