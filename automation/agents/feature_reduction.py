"""Decide on and apply dimensionality reduction."""

from __future__ import annotations

import json
import pandas as pd
from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from sklearn.decomposition import PCA


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt)


def run(state: PipelineState) -> PipelineState:
    """Consult an LLM on PCA usage and queue PCA code if recommended."""

    df = state.df
    stage_name = "feature_reduction"
    feature_cols = [c for c in df.columns if c != state.target]
    if not feature_cols:
        return state

    # measure simple multicollinearity
    numeric_df = df[feature_cols].select_dtypes(exclude="object")
    corr_matrix = numeric_df.corr().abs()
    high_corr_pairs = 0
    cols = list(corr_matrix.columns)
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            if corr_matrix.loc[c1, c2] > 0.85:
                high_corr_pairs += 1
    total_pairs = len(cols) * (len(cols) - 1) / 2 or 1
    high_corr_ratio = high_corr_pairs / total_pairs

    prompt = (
        "We have a dataset with "
        f"{len(feature_cols)} features. "
        f"About {high_corr_pairs} of {int(total_pairs)} feature pairs "
        f"have correlation above 0.85 (ratio {high_corr_ratio:.2f}). "
        "Should we apply PCA for dimensionality reduction? "
        "Respond in JSON with keys 'apply_pca' (yes/no) and 'reason'."
    )

    llm_raw = _query_llm(prompt)
    try:
        parsed = json.loads(llm_raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

    if not isinstance(parsed, dict) or "apply_pca" not in parsed:
        raise RuntimeError("LLM response missing 'apply_pca'")

    decision = str(parsed.get("apply_pca", "")).lower()
    apply_pca = decision.startswith("y")
    reason = parsed.get("reason", "")

    if apply_pca:
        state.append_log(f"FeatureReduction: PCA recommended - {reason}")
    else:
        state.append_log(f"FeatureReduction: skipped PCA - {reason}")

    code_snippet = (
        "pca = PCA(n_components=0.9)\n"
        f"components = pca.fit_transform(df[{feature_cols!r}])\n"
        "comp_cols = [f'pc{i+1}' for i in range(components.shape[1])]\n"
        "df = df[[target]].join(pd.DataFrame(components, columns=comp_cols, index=df.index))"
    )
    state.append_pending_code(stage_name, code_snippet)
    return state
