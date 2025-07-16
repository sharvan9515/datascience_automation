"""Train a model selected via an LLM recommendation."""

from __future__ import annotations

import json
import os

from automation.pipeline_state import PipelineState
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


def _query_llm(prompt: str) -> str | None:
    """Return raw LLM response or ``None`` if the call fails."""

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


def run(state: PipelineState) -> PipelineState:
    """Train a model suggested by the LLM and log metrics."""

    df = state.df
    X = df.drop(columns=[state.target])
    y = df[state.target]
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
    model_cls = None
    params: dict[str, object] = {}
    if llm_raw:
        try:
            parsed = json.loads(llm_raw)
            model_name = _normalize(str(parsed.get("model", "")))
            params = parsed.get("params", {}) or {}
            model_cls = MODEL_MAP.get(model_name)
        except Exception:
            model_cls = None

    if model_cls is None:
        # fallback based on task type
        model_cls = LogisticRegression if state.task_type == "classification" else LinearRegression
        params = {}
        state.append_log(
            f"ModelTraining: LLM unavailable, using fallback {model_cls.__name__}"
        )
    else:
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

    return state
