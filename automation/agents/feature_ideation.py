from automation.pipeline_state import PipelineState


def run(state: PipelineState) -> PipelineState:
    numeric_cols = [c for c in state.df.columns if state.df[c].dtype != 'O' and c != state.target]
    if len(numeric_cols) >= 2:
        feat_name = f"{numeric_cols[0]}_over_{numeric_cols[1]}"
        formula = f"{numeric_cols[0]} / ({numeric_cols[1]} + 1e-6)"
        state.log.append(f"FeatureIdeation: proposing {feat_name} = {formula}")
        state.features.append(feat_name)
    return state
