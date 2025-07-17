from __future__ import annotations

from abc import ABC, abstractmethod

from automation.pipeline_state import PipelineState

class BaseAgent(ABC):
    """Base class for all automation agents."""

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        """Execute the agent and return the updated state."""
        raise NotImplementedError
