"""Implement proposed features using LLM generated pandas code."""

from __future__ import annotations

import json
import os
import pandas as pd

from automation.pipeline_state import PipelineState


def _query_llm(prompt: str) -> str | None:
    """Return raw LLM response or ``None`` if the call fails."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
    except Exception:
        return None

    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message["content"].strip()
    except Exception:
        return None


def run(state: PipelineState) -> PipelineState:
    """Generate pandas code for each feature and execute it safely."""

    df = state.df.copy()

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
    parsed: dict[str, object] | None = None
    if llm_resp:
        try:
            parsed = json.loads(llm_resp)
        except json.JSONDecodeError:
            parsed = None

    # Simple fallback for _over_ style features if LLM is unavailable
    if not parsed or "code" not in parsed:
        for feat in state.features:
            if "_over_" in feat:
                num1, num2 = feat.split("_over_")
                if num1 in df.columns and num2 in df.columns:
                    df[feat] = df[num1] / (df[num2] + 1e-6)
                    state.append_log(f"FeatureImplementation fallback: created {feat}")
        state.df = df
        return state

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

    if success:
        for msg in logs or []:
            state.append_log(f"FeatureImplementation: {msg}")
        state.df = df
        return state

    # Final fallback if retries failed
    for feat in state.features:
        if "_over_" in feat and feat not in df.columns:
            num1, num2 = feat.split("_over_")
            if num1 in df.columns and num2 in df.columns:
                df[feat] = df[num1] / (df[num2] + 1e-6)
                state.append_log(f"FeatureImplementation fallback: created {feat}")

    state.df = df
    return state
