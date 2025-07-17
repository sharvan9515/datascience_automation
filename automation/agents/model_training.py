"""Train a model selected via an LLM recommendation."""

from __future__ import annotations

import json
import os
import joblib

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


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
}


def _normalize(name: str) -> str:
    return name.replace(" ", "").lower()


class Agent(BaseAgent):
    """Model training agent."""

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
        prompt = (
            "Select an appropriate scikit-learn model and basic hyperparameters "
            f"for a {state.task_type} task with {len(df)} samples and "
            f"{X.shape[1]} features. Respond in JSON with keys 'model' and 'params'."
        )

        llm_raw = _query_llm(prompt)
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
        state.append_code(stage_name, code_snippet)

        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state)
