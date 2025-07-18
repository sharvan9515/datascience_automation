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
        state.append_log("EnsembleAgent: building ensemble from tracked models.")
        trained_models = state.get_trained_models()
        if len(trained_models) < 2:
            state.append_log("EnsembleAgent: Not enough models for ensembling. Skipping.")
            return state
        import numpy as np
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        from sklearn.metrics import accuracy_score, r2_score
        # Prepare data
        df = state.df
        X = df.drop(columns=[state.target]).copy()
        y = df[state.target]
        X = X.fillna(0)
        # Build ensemble
        estimators = [(m['name'], m['model']) for m in trained_models]
        if state.task_type == 'classification':
            ensemble = VotingClassifier(estimators=estimators, voting='soft' if hasattr(estimators[0][1], 'predict_proba') else 'hard')
            try:
                ensemble.fit(X, y)
                preds = ensemble.predict(X)
                score = accuracy_score(y, preds)
                state.append_log(f"EnsembleAgent: VotingClassifier accuracy={score:.4f}")
            except Exception as e:
                state.append_log(f"EnsembleAgent: VotingClassifier failed: {e}")
                return state
        else:
            ensemble = VotingRegressor(estimators=estimators)
            try:
                ensemble.fit(X, y)
                preds = ensemble.predict(X)
                score = r2_score(y, preds)
                state.append_log(f"EnsembleAgent: VotingRegressor r2={score:.4f}")
            except Exception as e:
                state.append_log(f"EnsembleAgent: VotingRegressor failed: {e}")
                return state
        # Update best score if improved
        if state.best_score is None or score > state.best_score:
            state.best_score = score
            state.append_log("EnsembleAgent: Ensemble improved the best score.")
        return state

def run(state: PipelineState) -> PipelineState:
    return EnsembleAgent().run(state) 