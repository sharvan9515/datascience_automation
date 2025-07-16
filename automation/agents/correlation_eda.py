import os
import pandas as pd
from automation.pipeline_state import PipelineState


def _query_llm(prompt: str) -> str:
    """Return raw LLM response or raise ``RuntimeError`` on failure."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    try:
        import openai
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("openai package is required") from exc

    client = openai.OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def run(state: PipelineState) -> PipelineState:
    """Compute correlations/outliers and summarize via LLM."""
    df = state.df.copy()

    # Prepare numeric dataframe for correlation calculations
    numeric_df = df.select_dtypes(exclude="object").copy()
    if state.task_type == "classification" and state.target in df.columns:
        numeric_df[state.target] = df[state.target].astype("category").cat.codes
    elif state.target in df.columns:
        numeric_df[state.target] = df[state.target]

    corr_matrix = numeric_df.corr()
    if state.target in corr_matrix:
        target_corr = (
            corr_matrix[state.target]
            .drop(state.target)
            .abs()
            .sort_values(ascending=False)
        )
    else:
        target_corr = pd.Series(dtype=float)
    top_corr = target_corr.head(5)

    # Identify highly correlated feature pairs (possible redundancy)
    redundant = []
    cols = [c for c in numeric_df.columns if c != state.target]
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1 :]:
            if abs(corr_matrix.loc[c1, c2]) > 0.85:
                redundant.append((c1, c2))

    # Simple outlier detection using z-score
    outlier_counts = {}
    for col in cols:
        col_z = ((numeric_df[col] - numeric_df[col].mean()) / (numeric_df[col].std() + 1e-6)).abs()
        outlier_counts[col] = int((col_z > 3).sum())

    # Build prompt for LLM summarization
    prompt = (
        "You are an expert data analyst. "
        "Given the following correlation information and outlier counts, "
        "suggest any features that may be redundant due to high correlation or "
        "particularly predictive of the target. "
        "Respond in a short paragraph.\n"
        f"Top target correlations: {top_corr.to_dict()}\n"
        f"Highly correlated feature pairs: {redundant}\n"
        f"Outlier counts: {outlier_counts}"
    )

    summary = _query_llm(prompt)
    if not summary:
        raise RuntimeError("LLM did not return a summary for correlation EDA")
    state.append_log(f"CorrelationEDA: {summary}")
    return state
