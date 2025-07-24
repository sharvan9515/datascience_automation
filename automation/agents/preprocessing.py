import json
import pandas as pd
from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm, create_context_aware_prompt
from .base import BaseAgent
from automation.utils.sandbox import safe_exec
from automation.validators import DataValidator


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
        snapshot_version = state.create_snapshot()
        stage_name = "preprocessing"

        # Use must_keep from state if present, else default to []
        must_keep = getattr(state, 'must_keep', [])
        schema = {col: str(df[col].dtype) for col in df.columns}
        missing = df.isnull().sum().to_dict()
        unique_counts = {col: int(df[col].nunique(dropna=False)) for col in df.columns}
        # Dynamically build the prompt
        context = create_context_aware_prompt(
            state.profile,
            state.task_type or 'classification',
            stage_name,
            state.recommended_algorithms,
        )
        base_prompt = (
            f"{context}\n"
            f"You are a data preprocessing assistant.\n"
            f"Given a pandas DataFrame `df` with the following:\n"
            f"schema = {json.dumps(schema)}\n"
            f"missing = {json.dumps(missing)}\n"
            f"unique_counts = {json.dumps(unique_counts)}\n"
            f"must_keep = {json.dumps(must_keep)}\n"
            "\nIMPORTANT: Do NOT use the example column names ('A', 'B', etc.) in your code. Only use columns from the provided schema above.\n"
            "\n**Your tasks:**\n"
            "1. Handle missing values: Numeric → fill with mean, Categorical → fill with mode.\n"
            "2. Encode non-numeric columns (do NOT drop unless constant or unencodable free-text):\n"
            "   - Binary (unique_counts == 2) → ordinal (0/1)\n"
            "   - Low-cardinality (≤10 uniques) → one-hot\n"
            "   - High-cardinality (>10 uniques) → frequency or target encoding\n"
            "3. Never drop any column in must_keep unless it is constant or completely missing.\n"
            "   - If you drop any column, log the reason (e.g. 'constant', 'all missing', 'unencodable free-text') in the logs.\n"
            "4. Log every dropped column and each imputation or encoding action.\n"
            "5. Return a JSON object with:\n"
            "   - 'logs': list of messages describing each step\n"
            "   - 'code': pandas code (as one string) that modifies df in place (use df.loc or assignment)\n"
            "   - 'rationale': a brief explanation of your overall strategy\n"
            "\n---\nEXAMPLE (for a different dataset, do NOT use these column names):\n"
            "schema = {'A':'float64','B':'object','C':'int64','D':'object'}\n"
            "missing = {'A':2,'B':0,'C':0,'D':1}\n"
            "unique_counts = {'A':10,'B':2,'C':10,'D':20}\n"
            "must_keep = []\n"
            "\nExample output:\n"
            "{\n  'logs': [\n    'Filled 2 missing values in column A with mean.',\n    'Column B is binary categorical, encoded as ordinal (0/1).',\n    'Column D is high-cardinality categorical, encoded with frequency encoding.',\n    'Filled 1 missing value in column D with mode.'\n  ],\n  'code': \"df['A'] = df['A'].fillna(df['A'].mean())\\ndf['B'] = df['B'].astype('category').cat.codes\\ndf['D'] = df['D'].map(df['D'].value_counts())\\ndf['D'] = df['D'].fillna(df['D'].mode()[0])\",\n  'rationale': 'Numeric columns: mean imputation. Binary categoricals: ordinal encoding. High-cardinality categoricals: frequency encoding. All steps logged.'\n}\n"
            "---\nNow, using the provided schema, missing, unique_counts, and must_keep, produce your JSON response."
        )
        try:
            llm_resp = _query_llm(base_prompt)
        except RuntimeError as exc:
            state.append_log(f"Preprocessing: LLM query failed: {exc}")
            return state
        try:
            parsed = json.loads(llm_resp)
        except json.JSONDecodeError as exc:
            state.append_log(f"Preprocessing: Failed to parse LLM response: {exc}. Raw response: {llm_resp}")
            # Try to auto-correct common JSON issues (single to double quotes)
            fixed_resp = llm_resp.replace("'", '"')
            try:
                parsed = json.loads(fixed_resp)
                state.append_log("Preprocessing: Successfully parsed LLM response after auto-correction.")
            except Exception as exc2:
                state.append_log(f"Preprocessing: Still failed to parse LLM response after auto-correction: {exc2}. Skipping this step.")
                return state

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
            local_vars = safe_exec(
                code,
                state=state,
                extra_globals=exec_globals,
                local_vars=local_vars,
                allowed_modules={'pandas'},
            )
            local_vars['df'] = ensure_numeric_features(local_vars['df'], state.target, state)
            # Post-LLM validation: check must_keep features
            missing_features = [f for f in must_keep if f in df.columns and f not in local_vars['df'].columns]
            if missing_features:
                raise RuntimeError(f"LLM preprocessing dropped must_keep features: {missing_features}")
            # Check that all must_keep features are numeric or encoded
            for f in must_keep:
                if f in local_vars['df'].columns and not pd.api.types.is_numeric_dtype(local_vars['df'][f]):
                    raise RuntimeError(f"LLM preprocessing did not encode must_keep feature '{f}' as numeric.")
            # --- PATCH: Guarantee all non-target columns are numeric ---
            non_numeric_cols = [col for col in local_vars['df'].columns if col != state.target and not pd.api.types.is_numeric_dtype(local_vars['df'][col])]
            for col in non_numeric_cols:
                state.append_log(f"Preprocessing: Forcing encoding of non-numeric column '{col}' as categorical codes (guaranteed patch).")
                local_vars['df'][col] = local_vars['df'][col].astype('category').cat.codes
            # After forced encoding, check again
            still_non_numeric = [col for col in local_vars['df'].columns if col != state.target and not pd.api.types.is_numeric_dtype(local_vars['df'][col])]
            if still_non_numeric:
                raise RuntimeError(f"Preprocessing: Columns remain non-numeric after forced encoding: {still_non_numeric}")

            ok, reason = DataValidator.validate_transformation(df, local_vars['df'], state.target)
            if not ok:
                state.append_log(f"Preprocessing: validation failed - {reason}")
                state.rollback_to(snapshot_version)
                return state

            # Overwrite state.df with fully numeric DataFrame
            state.df = local_vars['df']
        except Exception as e:
            state.append_log(f"Preprocessing: LLM code failed with error: {e}")
            raise
        final_code = f"{code}\n# Ensure all columns are numeric\ndf = ensure_numeric_features(df, target)"
        state.append_pending_code(stage_name, final_code)
        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state)
