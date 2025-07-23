"""Generate feature implementation code via LLM and queue it for later execution."""

from __future__ import annotations

import json
import re
import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier, XGBRegressor
from automation.utils.sandbox import safe_exec

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent



def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt, expect_json=True)


def inject_missing_imports(code: str) -> str:
    imports = []
    # Unconditionally inject import re if 're' is used anywhere and not already imported
    if 're' in code and 'import re' not in code:
        imports.append('import re')
    if 'np' in code and 'import numpy as np' not in code:
        imports.append('import numpy as np')
    if 'pd' in code and 'import pandas as pd' not in code:
        imports.append('import pandas as pd')
    if 'sklearn' in code and 'import sklearn' not in code:
        imports.append('import sklearn')
    if 'LabelEncoder' in code and 'from sklearn.preprocessing import LabelEncoder' not in code:
        imports.append('from sklearn.preprocessing import LabelEncoder')
    if 'OneHotEncoder' in code and 'from sklearn.preprocessing import OneHotEncoder' not in code:
        imports.append('from sklearn.preprocessing import OneHotEncoder')
    if 'SimpleImputer' in code and 'from sklearn.impute import SimpleImputer' not in code:
        imports.append('from sklearn.impute import SimpleImputer')
    if 'XGBClassifier' in code and 'from xgboost import XGBClassifier' not in code:
        imports.append('from xgboost import XGBClassifier')
    if 'XGBRegressor' in code and 'from xgboost import XGBRegressor' not in code:
        imports.append('from xgboost import XGBRegressor')
    if imports:
        code = '\n'.join(imports) + '\n' + code
    return code


def ensure_numeric_features(df, target, state=None):
    for col in df.columns:
        if col == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            if state:
                state.append_log(f"FeatureImplementation: Encoding non-numeric column '{col}' as categorical codes.")
            df[col] = df[col].astype('category').cat.codes
        if df[col].isnull().any():
            fill_value = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]
            if state:
                state.append_log(f"FeatureImplementation: Filling missing values in column '{col}' with {fill_value}.")
            df[col] = df[col].fillna(fill_value)
    return df


class Agent(BaseAgent):
    """Feature implementation agent."""

    def run(self, state: PipelineState) -> PipelineState:
        """Generate pandas code for each feature and queue it for validation."""
        state.append_log("Feature engineering supervisor: implementation start")

        stage_name = "feature_implementation"

        if not state.features:
            # Nothing to implement
            return state

        # Ensure all non-numeric columns are encoded as numeric before feature engineering
        for col in state.df.columns:
            if col == state.target:
                continue
            if not pd.api.types.is_numeric_dtype(state.df[col]):
                state.df[col] = state.df[col].astype('category').cat.codes
                state.append_log(f"FeatureImplementation: Encoded non-numeric column '{col}' as category codes before feature engineering.")

        schema = {col: str(state.df[col].dtype) for col in state.df.columns}

        feature_descriptions = [
            f"{name} = {state.feature_formulas.get(name, '')}"
            for name in state.features
        ]

        # Main prompt for the current dataset
        base_prompt = (
            "You are a pandas expert. Given a DataFrame `df` with columns "
            f"{json.dumps(schema)}, implement the following new features: "
            f"{'; '.join(feature_descriptions)}. "
            "Assume all categorical columns are already encoded as numeric. Only use numeric columns in formulas. "
            "Avoid chained assignments and use df.loc for setting values. "
            "If you need to use .str or regex operations on a column, always check and convert the column to string first (e.g., df['col'] = df['col'].astype(str)). "
            "Return JSON with 'code' (Python code modifying df in place) and 'logs' (one message per feature describing the action)."
        )
        try:
            llm_resp = _query_llm(base_prompt)
        except RuntimeError as exc:
            state.append_log(f"FeatureImplementation: LLM query failed: {exc}")
            return state
        try:
            parsed: dict[str, object] = json.loads(llm_resp)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

        if "code" not in parsed:
            raise RuntimeError("LLM response missing 'code'")

        code = parsed.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        code = inject_missing_imports(code)
        logs = parsed.get("logs", [])
        if not isinstance(logs, list):
            logs = [str(logs)]
        for msg in logs:
            state.append_log(f"FeatureImplementation: {msg}")

        exec_globals = {
            're': re,
            'np': np,
            'pd': pd,
            'sklearn': sklearn,
            'LabelEncoder': LabelEncoder,
            'OneHotEncoder': OneHotEncoder,
            'SimpleImputer': SimpleImputer,
            'XGBClassifier': XGBClassifier,
            'XGBRegressor': XGBRegressor,
        }
        allowed = {'re', 'numpy', 'pandas', 'sklearn', 'xgboost'}
        try:
            local_vars = {'df': state.df.copy(), 'target': state.target}
            local_vars = safe_exec(
                code,
                state=state,
                extra_globals=exec_globals,
                local_vars=local_vars,
                allowed_modules=allowed,
            )
            # Ensure all features are numeric and have no missing values
            local_vars['df'] = ensure_numeric_features(local_vars['df'], state.target, state)
        except (TypeError, AttributeError) as e:
            state.append_log(f"FeatureImplementation: LLM code failed with error: {e}. Attempting to coerce only relevant columns to string and retry.")
            # Parse code to find columns used with .str or regex
            import re as _re
            str_cols = set()
            # Find df['col'].str or df["col"].str patterns
            for match in _re.finditer(r"df\[['\"]([\w_]+)['\"]\]\.str", code):
                str_cols.add(match.group(1))
            # Find regex usage: df['col'].apply(lambda x: re.search(...))
            for match in _re.finditer(r"df\[['\"]([\w_]+)['\"]\]\.apply\(lambda", code):
                str_cols.add(match.group(1))
            local_vars = {'df': state.df.copy(), 'target': state.target}
            for col in str_cols:
                if col in local_vars['df'].columns:
                    local_vars['df'][col] = local_vars['df'][col].astype(str)
            try:
                local_vars = safe_exec(
                    code,
                    state=state,
                    extra_globals=exec_globals,
                    local_vars=local_vars,
                    allowed_modules=allowed,
                )
                local_vars['df'] = ensure_numeric_features(local_vars['df'], state.target, state)
                state.append_log("FeatureImplementation: Retry after selective coercion succeeded.")
            except Exception as e2:
                state.append_log(f"FeatureImplementation: Retry after selective coercion failed with error: {e2}. Skipping this feature.")
                return state
        except KeyError as e:
            state.append_log(f"FeatureImplementation: Missing column error: {e}. Skipping this feature.")
            return state
        except Exception as e:
            state.append_log(f"FeatureImplementation: LLM code failed with error: {e}")
            # Retry: prompt LLM for a fix
            fix_prompt = (
                f"The previous code for implementing features failed with error: {e}. "
                f"Here is the code that failed:\n{code}\n"
                "Please provide corrected Python code for the same feature implementation as a JSON object with a single key 'code'. "
                "If you use .str or regex, always convert the column to string first."
            )
            try:
                fixed_code_json = _query_llm(fix_prompt)
            except RuntimeError as exc2:
                state.append_log(f"FeatureImplementation: fix query failed: {exc2}")
                raise RuntimeError(
                    "LLM did not return valid code for feature implementation"
                ) from exc2
            try:
                parsed = json.loads(fixed_code_json)
                fixed_code = parsed.get("code", "")
                if not isinstance(fixed_code, str):
                    fixed_code = str(fixed_code)
                fixed_code = inject_missing_imports(fixed_code)
            except Exception as e_json:
                state.append_log(f"FeatureImplementation: LLM code retry JSON parse failed: {e_json}. Raw response: {fixed_code_json}")
                raise RuntimeError(f"LLM did not return valid code for feature implementation. Last response: {fixed_code_json}")
            try:
                local_vars = safe_exec(
                    fixed_code,
                    state=state,
                    extra_globals=exec_globals,
                    local_vars=local_vars,
                    allowed_modules=allowed,
                )
                # Ensure all features are numeric and have no missing values
                local_vars['df'] = ensure_numeric_features(local_vars['df'], state.target, state)
                state.append_log("FeatureImplementation: LLM code retry succeeded.")
                code = fixed_code
            except Exception as e2:
                state.append_log(f"FeatureImplementation: LLM code retry failed with error: {e2}")
                raise RuntimeError(f"LLM code retry failed with error: {e2}")
        # After successful execution, always append the code to pending_code for tracking
        state.append_pending_code(stage_name, code)
        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state)
