from __future__ import annotations

import json
import pandas as pd
from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent
import re

def inject_missing_imports(code: str) -> str:
    # Add imports for common libraries if used in code
    imports = []
    if re.search(r'\bre\.', code) and 'import re' not in code:
        imports.append('import re')
    if re.search(r'\bnp\.', code) and 'import numpy as np' not in code:
        imports.append('import numpy as np')
    if re.search(r'\bpd\.', code) and 'import pandas as pd' not in code:
        imports.append('import pandas as pd')
    if re.search(r'\bsklearn\.', code) and 'import sklearn' not in code:
        imports.append('import sklearn')
    if re.search(r'\bLabelEncoder\b', code) and 'from sklearn.preprocessing import LabelEncoder' not in code:
        imports.append('from sklearn.preprocessing import LabelEncoder')
    if re.search(r'\bOneHotEncoder\b', code) and 'from sklearn.preprocessing import OneHotEncoder' not in code:
        imports.append('from sklearn.preprocessing import OneHotEncoder')
    if re.search(r'\bSimpleImputer\b', code) and 'from sklearn.impute import SimpleImputer' not in code:
        imports.append('from sklearn.impute import SimpleImputer')
    if re.search(r'\bXGBClassifier\b', code) and 'from xgboost import XGBClassifier' not in code:
        imports.append('from xgboost import XGBClassifier')
    if re.search(r'\bXGBRegressor\b', code) and 'from xgboost import XGBRegressor' not in code:
        imports.append('from xgboost import XGBRegressor')
    if imports:
        code = '\n'.join(imports) + '\n' + code
    return code

class BaselineAgent(BaseAgent):
    """Agentic, dataset-agnostic baseline pipeline generator and executor."""

    def run(self, state: PipelineState) -> PipelineState:
        state.append_log("BaselineAgent: generating robust baseline pipeline via LLM.")
        df = state.df.copy()
        schema = {col: str(df[col].dtype) for col in df.columns}
        sample = df.head(5).to_dict(orient='list')
        prompt = (
            "You are a data science assistant. Given a pandas DataFrame `df` with schema "
            f"{json.dumps(schema)} and a sample {json.dumps(sample)}, and a target column '{state.target}', "
            "generate Python code to:\n"
            "- Drop or encode all non-numeric columns (including IDs, free-text, high-cardinality categoricals).\n"
            "- Handle missing values in both numeric and categorical columns.\n"
            "- Encode the target column if it is non-numeric.\n"
            "- Use XGBoost (or another strong model) for classification or regression, as appropriate.\n"
            "- Evaluate the model using the best metrics for the task.\n"
            "- Do not assume any specific column names. The code should work for any dataset.\n"
            "Return your answer as a JSON object with a single key code containing the Python code as a string."
        )
        llm_code_json = query_llm(prompt)
        try:
            parsed = json.loads(llm_code_json)
            llm_code = parsed.get("code", "")
            if not isinstance(llm_code, str):
                llm_code = str(llm_code)
        except Exception as e_json:
            state.append_log(f"BaselineAgent: LLM code JSON parse failed: {e_json}. Raw response: {llm_code_json}")
            raise RuntimeError(f"LLM did not return valid code for baseline agent. Last response: {llm_code_json}")
        state.append_log("BaselineAgent: LLM-generated code:\n" + llm_code)
        # Auto-import and error handling for LLM code
        llm_code = inject_missing_imports(llm_code)
        local_vars = {'df': df.copy(), 'target': state.target}
        try:
            exec(llm_code, {}, local_vars)
            state.append_log("BaselineAgent: baseline code executed successfully.")
        except Exception as e:
            state.append_log(f"BaselineAgent: baseline code failed with error: {e}")
            # Optionally, prompt the LLM for a fix here
        # Optionally, update state.df/model/score from local_vars if needed
        return state

def run(state: PipelineState) -> PipelineState:
    return BaselineAgent().run(state) 