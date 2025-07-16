"""Implement proposed features using LLM generated pandas code."""

from __future__ import annotations

import json
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
    """Generate pandas code for each feature and execute it safely."""

    df = state.df.copy()
    stage_name = "feature_implementation"

    if not state.features:
        # Nothing to implement
        return state

    schema = {col: str(df[col].dtype) for col in df.columns}

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
    success = False

    # allow two retries with error feedback
    for attempt in range(3):
        local_env = {"df": df, "pd": pd}
        try:
            exec(code, {}, local_env)
            df = local_env["df"]
            success = True
            break
        except Exception as e:  # noqa: BLE001
            error_prompt = (
                base_prompt
                + f" The previous code failed with: {e}. "
                "Please return corrected JSON with 'code' and 'logs'."
            )
            llm_resp = _query_llm(error_prompt)
            if not llm_resp:
                break
            try:
                parsed = json.loads(llm_resp)
            except json.JSONDecodeError:
                parsed = None
            if not parsed or "code" not in parsed:
                break
            code = parsed.get("code", "")
            logs = parsed.get("logs", [])
            state.append_log("FeatureImplementation: retrying after error")

    if not success:
        raise RuntimeError("LLM-provided feature implementation code failed")

    for msg in logs or []:
        state.append_log(f"FeatureImplementation: {msg}")
    state.append_code(stage_name, code)
    state.df = df
    return state
