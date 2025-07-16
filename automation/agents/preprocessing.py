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
    """Use an LLM to determine preprocessing steps and apply them safely."""
    df = state.df.copy()
    stage_name = "preprocessing"

    schema = {col: str(df[col].dtype) for col in df.columns}
    missing = df.isnull().sum().to_dict()

    base_prompt = (
        "You are a data preprocessing assistant. "
        "Given a pandas DataFrame `df` with schema "
        f"{json.dumps(schema)} and missing counts {json.dumps(missing)}, "
        "suggest preprocessing steps for machine learning. "
        "Return JSON with keys 'logs' (list of messages describing each step) "
        "and 'code' (Python pandas code that modifies df in place)."
    )

    llm_resp = _query_llm(base_prompt)
    try:
        parsed = json.loads(llm_resp)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

    if 'code' not in parsed:
        raise RuntimeError("LLM response missing 'code' field")

    code = parsed.get('code', '')
    logs = parsed.get('logs', [])

    success = False
    for attempt in range(2):
        local_env = {'df': df, 'pd': pd}
        try:
            exec(code, {}, local_env)
            df = local_env['df']
            success = True
            break
        except Exception as e:
            error_prompt = (
                base_prompt
                + f" The previous code failed with: {e}. Please return corrected JSON."
            )
            llm_resp = _query_llm(error_prompt)
            try:
                parsed = json.loads(llm_resp)
            except json.JSONDecodeError:
                raise RuntimeError("Failed to parse corrected LLM response")
            if 'code' not in parsed:
                raise RuntimeError("Corrected LLM response missing 'code'")
            code = parsed.get('code', '')
            logs = parsed.get('logs', [])

    if not success:
        raise RuntimeError("LLM-provided preprocessing code failed")

    for msg in logs:
        state.append_log(f"Preprocessing: {msg}")
    if rationale := parsed.get('rationale'):
        state.append_log(f"Preprocessing rationale: {rationale}")
    state.append_code(stage_name, code)
    state.df = df
    return state
