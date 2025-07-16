import json
import os
import pandas as pd
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
    """Use an LLM to determine preprocessing steps and apply them safely."""
    df = state.df.copy()

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
    parsed = None
    if llm_resp:
        try:
            parsed = json.loads(llm_resp)
        except json.JSONDecodeError:
            parsed = None

    if not parsed or 'code' not in parsed:
        # fallback to simple mean/mode imputation
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'O':
                    fill_value = df[col].mode().iloc[0]
                else:
                    fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
                state.append_log(
                    f"Preprocessing fallback: filled missing values in {col} with {fill_value}"
                )
        state.df = df
        return state

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
            if not llm_resp:
                break
            try:
                parsed = json.loads(llm_resp)
            except json.JSONDecodeError:
                parsed = None
            if not parsed or 'code' not in parsed:
                break
            code = parsed.get('code', '')
            logs = parsed.get('logs', [])

    if success:
        for msg in logs:
            state.append_log(f"Preprocessing: {msg}")
        if rationale := parsed.get('rationale'):
            state.append_log(f"Preprocessing rationale: {rationale}")
        state.df = df
        return state

    # final fallback if LLM execution failed
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'O':
                fill_value = df[col].mode().iloc[0]
            else:
                fill_value = df[col].mean()
            df[col] = df[col].fillna(fill_value)
            state.append_log(
                f"Preprocessing fallback: filled missing values in {col} with {fill_value}"
            )
    state.df = df
    return state
