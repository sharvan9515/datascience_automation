from __future__ import annotations

import json
import re
import pandas as pd
from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent

def inject_missing_imports(code: str) -> str:
    imports = []
    if re.search(r'\bre\.', code) and 'import re' not in code:
        imports.append('import re')
    if re.search(r'\bnp\.', code) and 'import numpy as np' not in code:
        imports.append('import numpy as np')
    if re.search(r'\bpd\.', code) and 'import pandas as pd' not in code:
        imports.append('import pandas as pd')
    if re.search(r'\bsklearn\.', code) and 'import sklearn' not in code:
        imports.append('import sklearn')
    if re.search(r'\bVotingClassifier\b', code) and 'from sklearn.ensemble import VotingClassifier' not in code:
        imports.append('from sklearn.ensemble import VotingClassifier')
    if re.search(r'\bVotingRegressor\b', code) and 'from sklearn.ensemble import VotingRegressor' not in code:
        imports.append('from sklearn.ensemble import VotingRegressor')
    if re.search(r'\bXGBClassifier\b', code) and 'from xgboost import XGBClassifier' not in code:
        imports.append('from xgboost import XGBClassifier')
    if re.search(r'\bXGBRegressor\b', code) and 'from xgboost import XGBRegressor' not in code:
        imports.append('from xgboost import XGBRegressor')
    if imports:
        code = '\n'.join(imports) + '\n' + code
    return code

class EnsembleAgent(BaseAgent):
    """Agentic ensembling: combine predictions from multiple models."""
    def run(self, state: PipelineState) -> PipelineState:
        state.append_log("EnsembleAgent: proposing ensemble model via LLM.")
        # For now, just mention the models used so far (could be tracked in state)
        prompt = (
            "Given trained models (e.g., RandomForest, XGBoost, LogisticRegression) and their predictions, "
            "generate Python code to combine their predictions using majority voting (for classification) or averaging (for regression). "
            "Evaluate the ensemble using the best metric for the task. "
            "Assume you have X_train, X_test, y_train, y_test, and trained models named model1, model2, model3. "
            "Return your answer as a JSON object with a single key code containing the Python code as a string."
        )
        llm_code_json = query_llm(prompt)
        try:
            parsed = json.loads(llm_code_json)
            llm_code = parsed.get("code", "")
            if not isinstance(llm_code, str):
                llm_code = str(llm_code)
        except Exception as e_json:
            state.append_log(f"EnsembleAgent: LLM code JSON parse failed: {e_json}. Raw response: {llm_code_json}")
            raise RuntimeError(f"LLM did not return valid code for ensemble agent. Last response: {llm_code_json}")
        state.append_log("EnsembleAgent: LLM-generated code:\n" + llm_code)
        llm_code = inject_missing_imports(llm_code)
        local_vars = {}
        try:
            exec(llm_code, {}, local_vars)
            state.append_log("EnsembleAgent: ensemble code executed successfully.")
        except Exception as e:
            state.append_log(f"EnsembleAgent: ensemble code failed with error: {e}")
        # Optionally, update state with ensemble results if improved
        return state

def run(state: PipelineState) -> PipelineState:
    return EnsembleAgent().run(state) 