"""Evaluate a simple model and decide if another iteration is needed."""

from __future__ import annotations

import json
import os
from automation.pipeline_state import PipelineState
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression

__all__ = ["compute_score", "run"]


def _query_llm(prompt: str) -> str:
    """Return raw LLM response or raise ``RuntimeError`` on failure."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is required")
    try:
        import openai
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("openai package is required") from exc

    client = openai.OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LLM call failed: {exc}") from exc


def compute_score(df: pd.DataFrame, target: str, task_type: str) -> float:
    """Return a simple train/test score for the given dataset."""

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if task_type == "classification":
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average="weighted")

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)


def run(state: PipelineState) -> PipelineState:
    """Train a quick model, analyze metrics, and consult the LLM for advice."""

    df = state.df
    X = df.drop(columns=[state.target])
    y = df[state.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if state.task_type == "classification":
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds)

        metrics_str = (
            f"accuracy={acc:.4f}\nconfusion_matrix={cm.tolist()}\n{report}"
        )
        state.append_log(f"ModelEvaluation metrics:\n{metrics_str}")
        new_score = acc
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        metrics_str = (
            f"r2={r2:.4f}, mse={mse:.4f}, mae={mae:.4f}"
        )
        state.append_log(f"ModelEvaluation metrics: {metrics_str}")
        new_score = r2

    # Ask the LLM for refinement suggestions
    prompt = (
        "Given these evaluation metrics, suggest ways to improve the model. "
        "Mention new features, transformations, or model adjustments. "
        "Should we iterate further? Respond in JSON with keys 'iterate', 'reason',"
        " and 'suggestions'.\n"
        f"Metrics: {metrics_str}"
    )

    llm_raw = _query_llm(prompt)
    try:
        parsed = json.loads(llm_raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("LLM response must be a JSON object")

    reason = parsed.get("reason", "")
    suggestions = parsed.get("suggestions", "")

    prev_best = state.best_score
    tol = 0.01
    improved = prev_best is None or new_score > prev_best + tol
    if improved:
        state.best_score = new_score
        state.no_improve_rounds = 0
    else:
        state.no_improve_rounds += 1

    state.iterate = not (
        state.no_improve_rounds >= 5
        or state.iteration >= state.max_iter
    )

    log_msg = f"ModelEvaluation decision: iterate={state.iterate} - {reason}".strip()
    state.append_log(log_msg)
    if suggestions:
        state.append_log(f"ModelEvaluation suggestions: {suggestions}")

    state.iteration_history.append(
        {
            "iteration": state.iteration,
            "metrics": metrics_str,
            "best_score": state.best_score,
            "no_improve_rounds": state.no_improve_rounds,
            "iterate": state.iterate,
        }
    )

    return state
