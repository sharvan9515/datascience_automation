from automation.pipeline_state import PipelineState
from . import (
    task_identification,
    preprocessing,
    correlation_eda,
    feature_ideation,
    feature_implementation,
    feature_selection,
    feature_reduction,
    model_training,
    model_evaluation,
)

AGENT_SEQUENCE = [
    task_identification,
    preprocessing,
    correlation_eda,
    feature_ideation,
    feature_implementation,
    feature_selection,
    feature_reduction,
    model_training,
    model_evaluation,
]


def run(state: PipelineState) -> PipelineState:
    for agent in AGENT_SEQUENCE:
        state = agent.run(state)
    return state
