"""Generate feature implementation code via LLM and queue it for later execution."""

from __future__ import annotations

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
    """Generate pandas code for each feature and queue it for validation."""

    stage_name = "feature_implementation"

    if not state.features:
        # Nothing to implement
        return state

    schema = {col: str(state.df[col].dtype) for col in state.df.columns}

    base_prompt = (
        "You are a pandas expert. Given a DataFrame `df` with columns "
        f"{json.dumps(schema)}, implement the following new features: "
        f"{state.features}. Return JSON with 'code' (Python code modifying df in"
        " place) and 'logs' (one message per feature describing the action)."
    )

    llm_resp = _query_llm(base_prompt)
    try:
        parsed: dict[str, object] = json.loads(llm_resp)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

    if "code" not in parsed:
        raise RuntimeError("LLM response missing 'code'")

    code = parsed.get("code", "")
    logs = parsed.get("logs", [])
    for msg in logs or []:
        state.append_log(f"FeatureImplementation: {msg}")

    state.append_pending_code(stage_name, code)
    return state
