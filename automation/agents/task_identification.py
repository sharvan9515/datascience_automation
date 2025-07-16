import json
import os
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
    try:
        parsed = json.loads(llm_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

    if not isinstance(parsed, dict) or "task_type" not in parsed:
        raise RuntimeError("LLM response missing required 'task_type'")

    state.task_type = parsed["task_type"].lower()
    if issues := parsed.get("issues"):
        state.append_log(f"TaskIdentification issues: {issues}")
    state.append_log(
        f"TaskIdentification: determined task_type={state.task_type} via LLM"
    )
    return state
