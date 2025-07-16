import json
import os
from automation.pipeline_state import PipelineState


def _query_llm(prompt: str) -> str | None:
    """Return raw LLM response or None if call fails."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
    except Exception:
        return None

    client = openai.OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


def run(state: PipelineState) -> PipelineState:
    df = state.df

    schema = {col: str(df[col].dtype) for col in df.columns}
    stats = df.describe(include="all").to_dict()

    prompt = (
        "Given the following dataset schema and summary statistics, "
        "determine whether predicting the target column should be "
        "treated as classification or regression. "
        "Also mention any immediate data quality issues. "
        "Respond in JSON with keys 'task_type' and optional 'issues'.\n"
        f"Target: {state.target}\n"
        f"Schema: {json.dumps(schema)}\n"
        f"Stats: {json.dumps(stats, default=str)}"
    )

    llm_raw = _query_llm(prompt)
    parsed = None
    if llm_raw:
        try:
            parsed = json.loads(llm_raw)
        except json.JSONDecodeError:
            parsed = None

    if parsed and isinstance(parsed, dict) and parsed.get("task_type"):
        state.task_type = parsed["task_type"].lower()
        if issues := parsed.get("issues"):
            state.append_log(f"TaskIdentification issues: {issues}")
        state.append_log(
            f"TaskIdentification: determined task_type={state.task_type} via LLM"
        )
        return state

    # Fallback heuristic if LLM response is missing or malformed
    target_series = df[state.target]
    if target_series.dtype == "O" or target_series.nunique() < 20:
        fallback = "classification"
    else:
        fallback = "regression"
    state.task_type = fallback
    state.append_log(
        f"TaskIdentification: LLM unclear, used heuristic task_type={fallback}"
    )
    return state
