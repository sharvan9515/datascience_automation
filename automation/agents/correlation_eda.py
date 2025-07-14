import pandas as pd
from automation.pipeline_state import PipelineState


def run(state: PipelineState) -> PipelineState:
    if state.task_type != 'classification':
        corr = state.df.corr()[state.target].drop(state.target).abs().sort_values(ascending=False)
    else:
        # For classification, use numeric encoding for correlation
        df = state.df.copy()
        df[state.target] = df[state.target].astype('category').cat.codes
        corr = df.corr()[state.target].drop(state.target).abs().sort_values(ascending=False)
    top = corr.head(5)
    state.append_log(f"CorrelationEDA: top correlated features: {', '.join(top.index)}")
    return state
