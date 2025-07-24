"""Train a model selected via an LLM recommendation."""

from __future__ import annotations

import json
import os
import joblib

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm, create_context_aware_prompt
from ..intelligent_model_selector import IntelligentModelSelector
from .base import BaseAgent
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt, expect_json=True)


MODEL_MAP = {
    "logisticregression": LogisticRegression,
    "linearregression": LinearRegression,
    "randomforestclassifier": RandomForestClassifier,
    "randomforestregressor": RandomForestRegressor,
    "svc": SVC,
    "svr": SVR,
    "xgbclassifier": XGBClassifier,
    "xgbregressor": XGBRegressor,
}
if HAS_LGBM:
    MODEL_MAP["lgbmclassifier"] = LGBMClassifier
    MODEL_MAP["lgbmregressor"] = LGBMRegressor


def _normalize(name: str) -> str:
    return name.replace(" ", "").lower()


class ModelTrainingAgent(BaseAgent):
    """Train a selected model and record metrics."""

    def run(self, state: PipelineState) -> PipelineState:
        """Train a model suggested by the LLM and log metrics."""
        state.append_log("Model training supervisor: starting")

        df = state.df
        stage_name = "model_training"
        X = df.drop(columns=[state.target]).copy()
        y = df[state.target]
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category").cat.codes
        X = X.fillna(0)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Ask the LLM for an appropriate algorithm and params
        context = create_context_aware_prompt(
            state.profile,
            state.task_type or 'classification',
            'model training',
            state.recommended_algorithms,
        )
        recommended = state.recommended_algorithms or IntelligentModelSelector.select_optimal_algorithms(
            state.profile,
            state.task_type or 'classification',
        )
        rec_text = f"Recommended algorithms: {', '.join(recommended)}." if recommended else ''
        prompt = (
            f"{context}\n{rec_text}\n"
            "Select an appropriate model (choose from: LogisticRegression, RandomForestClassifier, SVC, XGBClassifier, LGBMClassifier for classification; LinearRegression, RandomForestRegressor, SVR, XGBRegressor, LGBMRegressor for regression) and basic hyperparameters "
            f"for a {state.task_type} task with {len(df)} samples and "
            f"{X.shape[1]} features. Respond in JSON with keys 'model' and 'params'."
        )

        try:
            llm_raw = _query_llm(prompt)
        except RuntimeError as exc:
            state.append_log(f"ModelTraining: LLM query failed: {exc}")
            return state
        try:
            parsed = json.loads(llm_raw)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

        model_name = _normalize(str(parsed.get("model")))
        if not model_name:
            raise RuntimeError("LLM response missing model name")
        params = parsed.get("params", {}) or {}
        model_cls = MODEL_MAP.get(model_name)
        if model_cls is None:
            raise RuntimeError(f"Unsupported model suggested by LLM: {model_name}")

        if model_cls is LogisticRegression:
            params.setdefault("max_iter", 500)

        state.append_log(
            f"ModelTraining: selected {model_cls.__name__} with params {params}"
        )

        model = model_cls(**params)
        model.fit(X_train, y_train)

        if state.task_type == "classification":
            score = accuracy_score(y_test, model.predict(X_test))
            state.append_log(
                f"ModelTraining: {model_cls.__name__} accuracy={score:.4f}"
            )
        else:
            score = r2_score(y_test, model.predict(X_test))
            state.append_log(
                f"ModelTraining: {model_cls.__name__} r2={score:.4f}"
            )

        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(model, "artifacts/model.pkl")
        code_snippet = (
            f"X = df.drop(columns=['{state.target}'])\n"
            f"y = df['{state.target}']\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
            f"model = {model_cls.__name__}(**{params})\n"
            "model.fit(X_train, y_train)\n"
            "joblib.dump(model, 'artifacts/model.pkl')"
        )
        state.append_pending_code(stage_name, code_snippet)

        # Track the trained model for ensembling
        y_pred = model.predict(X_test)
        state.add_trained_model(
            model=model,
            name=model_cls.__name__,
            model_type=state.task_type,
            predictions=y_pred,
            score=score if 'score' in locals() else None
        )

        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    """Backwards compatible function API."""
    return ModelTrainingAgent().run(state)
