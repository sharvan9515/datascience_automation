"""Evaluate a simple model and decide if another iteration is needed."""

from __future__ import annotations

import json
import os
from automation.pipeline_state import PipelineState
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.linear_model import LogisticRegression, LinearRegression


def _query_llm(prompt: str) -> str | None:
    """Return raw LLM response or ``None`` if the call fails."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
    except Exception:
        return None

    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message["content"].strip()
    except Exception:
        return None


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

    # Ask the LLM for refinement suggestions
    prompt = (
        "Given these evaluation metrics, suggest ways to improve the model. "
        "Mention new features, transformations, or model adjustments. "
        "Should we iterate further? Respond in JSON with keys 'iterate', 'reason',"
        " and 'suggestions'.\n"
        f"Metrics: {metrics_str}"
    )

    llm_raw = _query_llm(prompt)
    iterate = None
    reason = ""
    suggestions = ""
    if llm_raw:
        try:
            parsed = json.loads(llm_raw)
            iterate = str(parsed.get("iterate", "")).lower().startswith("y")
            reason = parsed.get("reason", "")
            suggestions = parsed.get("suggestions", "")
        except Exception:
            iterate = None

    if iterate is None:
        # basic heuristic fallback
        if state.task_type == "classification":
            iterate = acc < 0.9
            reason = (
                "heuristic: accuracy < 0.9" if iterate else "heuristic: accuracy sufficient"
            )
        else:
            iterate = r2 < 0.8
            reason = (
                "heuristic: r2 < 0.8" if iterate else "heuristic: r2 sufficient"
            )

    state.iterate = bool(iterate)
    log_msg = f"ModelEvaluation decision: iterate={state.iterate} - {reason}".strip()
    state.append_log(log_msg)
    if suggestions:
        state.append_log(f"ModelEvaluation suggestions: {suggestions}")

    return state
