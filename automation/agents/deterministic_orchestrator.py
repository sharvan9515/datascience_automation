from __future__ import annotations

from automation.pipeline_state import PipelineState
from .task_identification import TaskIdentificationAgent
from .timeseries_detection import TimeseriesDetectionAgent
from .preprocessing import PreprocessingAgent
from .correlation_eda import CorrelationEDAAgent
from .feature_ideation import FeatureIdeationAgent
from .feature_implementation import FeatureImplementationAgent
from .feature_selection import FeatureSelectionAgent
from .feature_reduction import FeatureReductionAgent
from .model_training import ModelTrainingAgent
from .model_evaluation import ModelEvaluationAgent
from .. import code_assembler


class DeterministicOrchestrator:
    """Run a fixed sequence of agents until no further iterations are requested."""

    AGENT_ORDER = [
        TimeseriesDetectionAgent(),
        TaskIdentificationAgent(),
        PreprocessingAgent(),
        CorrelationEDAAgent(),
        FeatureIdeationAgent(),
        FeatureImplementationAgent(),
        FeatureSelectionAgent(),
        FeatureReductionAgent(),
        ModelTrainingAgent(),
        ModelEvaluationAgent(),
    ]

    def run_pipeline(self, state: PipelineState, max_iter: int = 3) -> PipelineState:
        state.append_log("DeterministicOrchestrator: starting pipeline")
        iteration = 0
        state.iterate = True
        while state.iterate and iteration < max_iter:
            state.append_log(f"DeterministicOrchestrator: iteration {iteration}")
            for agent in self.AGENT_ORDER:
                state.append_log(f"Running {agent.__class__.__name__}")
                try:
                    state = agent.run(state)
                except Exception as exc:  # noqa: BLE001
                    state.append_log(f"{agent.__class__.__name__} failed: {exc}")
                    state.iterate = False
                    break
            iteration += 1
        state = code_assembler.run(state)
        state.append_log(
            f"DeterministicOrchestrator finished after {iteration} iterations with best score {state.best_score}"
        )
        return state


def run(state: PipelineState, max_iter: int = 3) -> PipelineState:
    """Convenience function for backward compatibility."""
    return DeterministicOrchestrator().run_pipeline(state, max_iter=max_iter)
