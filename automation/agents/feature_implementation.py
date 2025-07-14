from automation.pipeline_state import PipelineState


def run(state: PipelineState) -> PipelineState:
    df = state.df.copy()
    for feat in state.features:
        if '_over_' in feat:
            num1, num2 = feat.split('_over_')
            if num1 in df.columns and num2 in df.columns:
                df[feat] = df[num1] / (df[num2] + 1e-6)
                state.append_log(f"FeatureImplementation: created {feat}")
    state.df = df
    return state
