from automation.pipeline_state import PipelineState


def run(state: PipelineState) -> PipelineState:
    target_series = state.df[state.target]
    if target_series.dtype == 'O' or target_series.nunique() < 20:
        state.task_type = 'classification'
    else:
        state.task_type = 'regression'
    state.log.append(f"TaskIdentification: determined task_type={state.task_type}")
    return state
