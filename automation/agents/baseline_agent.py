from __future__ import annotations

import json
import pandas as pd
from automation.pipeline_state import PipelineState
from automation.basic_template import run_basic_template
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
        state.append_log("BaselineAgent: running basic_template.py logic as baseline.")
        df, model, score = run_basic_template(
            state.df.copy(),
            state.target,
            state.task_type,
            time_col=state.time_col if state.timeseries_mode else None,
        )
        state.df = df
        state.best_score = score
        # Optionally, store model, logs, etc. in state
        return state

def run(state: PipelineState) -> PipelineState:
    return BaselineAgent().run(state) 