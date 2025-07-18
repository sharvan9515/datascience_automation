import json
import pandas as pd
from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt)


def ensure_numeric_features(df, target, state=None):
    for col in df.columns:
        if col == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            if state:
                state.append_log(f"Preprocessing: Encoding non-numeric column '{col}' as categorical codes.")
            df[col] = df[col].astype('category').cat.codes
        if df[col].isnull().any():
            fill_value = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]
            if state:
                state.append_log(f"Preprocessing: Filling missing values in column '{col}' with {fill_value}.")
            df[col] = df[col].fillna(fill_value)
    return df


class Agent(BaseAgent):
    """Preprocessing agent."""

    def run(self, state: PipelineState) -> PipelineState:
        """Query the LLM for preprocessing code and store it for later execution."""
        state.append_log("Preprocessing supervisor: starting")

        df = state.df.copy()
        stage_name = "preprocessing"

        key_features = ['Sex', 'Embarked', 'Pclass', 'Fare', 'Age']
        schema = {col: str(df[col].dtype) for col in df.columns}
        missing = df.isnull().sum().to_dict()
        unique_counts = {col: int(df[col].nunique(dropna=False)) for col in df.columns}
        base_prompt = (
            "You are a data preprocessing assistant. "
            "Given a pandas DataFrame `df` with schema "
            f"{json.dumps(schema)} and missing counts {json.dumps(missing)}, "
            f"and unique value counts {json.dumps(unique_counts)}, "
            "suggest preprocessing steps for machine learning. "
            "Encode all non-numeric columns (do not drop unless they are constant or truly unencodable free-text). "
            "Prefer one-hot encoding for low-cardinality categoricals, ordinal for binary, and frequency/target encoding for high-cardinality. "
            "Handle missing values in both numeric and categorical columns. "
            "Log all dropped columns and encoding choices. "
            "Return JSON with keys 'logs' (list of messages describing each step), 'code' (Python pandas code that modifies df in place), and 'rationale' (explanation)."
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

        for msg in logs:
            state.append_log(f"Preprocessing: {msg}")
        if rationale := parsed.get('rationale'):
            state.append_log(f"Preprocessing rationale: {rationale}")
        # Ensure all features are numeric and have no missing values
        try:
            exec_globals = {'pd': pd}
            local_vars = {'df': state.df.copy(), 'target': state.target}
            exec(code, exec_globals, local_vars)
            local_vars['df'] = ensure_numeric_features(local_vars['df'], state.target, state)
            # Post-LLM validation: check key features
            missing_features = [f for f in key_features if f in df.columns and f not in local_vars['df'].columns]
            if missing_features:
                raise RuntimeError(f"LLM preprocessing dropped key features: {missing_features}")
            # Check that all key features are numeric or encoded
            for f in key_features:
                if f in local_vars['df'].columns and not pd.api.types.is_numeric_dtype(local_vars['df'][f]):
                    raise RuntimeError(f"LLM preprocessing did not encode key feature '{f}' as numeric.")
        except Exception as e:
            state.append_log(f"Preprocessing: LLM code failed with error: {e}")
            raise
        state.append_pending_code(stage_name, code)
        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state)
