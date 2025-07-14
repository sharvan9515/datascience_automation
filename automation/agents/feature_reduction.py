import pandas as pd
from automation.pipeline_state import PipelineState
from sklearn.decomposition import PCA


def run(state: PipelineState) -> PipelineState:
    df = state.df
    feature_cols = [c for c in df.columns if c != state.target]
    if len(feature_cols) > 15:
        pca = PCA(n_components=0.9)
        components = pca.fit_transform(df[feature_cols])
        comp_cols = [f"pc{i+1}" for i in range(components.shape[1])]
        state.df = df[[state.target]].join(
            pd.DataFrame(components, columns=comp_cols, index=df.index)
        )
        state.log.append(f"FeatureReduction: applied PCA to {len(feature_cols)} features -> {len(comp_cols)} components")
    return state
