"""Incrementally evaluate new features and keep only useful ones."""

from __future__ import annotations

import os
from automation.pipeline_state import PipelineState
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
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


def _evaluate(df, target: str, task_type: str) -> dict[str, float]:
    """Train a simple model and return relevant metrics."""

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    if task_type == "classification":
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    return {"r2": r2, "rmse": rmse}


def run(state: PipelineState) -> PipelineState:
    """Evaluate new features incrementally and keep only beneficial ones."""

    df = state.df.copy()
    new_feats = [f for f in state.features if f in df.columns]

    # Baseline without proposed features
    baseline_df = df.drop(columns=new_feats, errors="ignore")
    baseline_metrics = _evaluate(baseline_df, state.target, state.task_type)

    if state.task_type == "classification":
        state.append_log(
            (
                "FeatureSelection: baseline accuracy="
                f"{baseline_metrics['accuracy']:.4f}, f1={baseline_metrics['f1']:.4f}"
            )
        )
    else:
        state.append_log(
            (
                "FeatureSelection: baseline r2="
                f"{baseline_metrics['r2']:.4f}, rmse={baseline_metrics['rmse']:.4f}"
            )
        )

    current_df = baseline_df
    current_metrics = baseline_metrics
    kept_features: list[str] = []

    for feat in new_feats:
        trial_df = current_df.join(df[[feat]])
        trial_metrics = _evaluate(trial_df, state.target, state.task_type)

        if state.task_type == "classification":
            delta_acc = trial_metrics["accuracy"] - current_metrics["accuracy"]
            delta_f1 = trial_metrics["f1"] - current_metrics["f1"]
            prompt = (
                f"Baseline accuracy {current_metrics['accuracy']:.4f}, f1 {current_metrics['f1']:.4f}. "
                f"After adding feature '{feat}', accuracy {trial_metrics['accuracy']:.4f}, f1 {trial_metrics['f1']:.4f}. "
                "Should we keep this feature? Reply yes or no."
            )
        else:
            delta_r2 = trial_metrics["r2"] - current_metrics["r2"]
            delta_rmse = trial_metrics["rmse"] - current_metrics["rmse"]
            prompt = (
                f"Baseline r2 {current_metrics['r2']:.4f}, rmse {current_metrics['rmse']:.4f}. "
                f"After adding feature '{feat}', r2 {trial_metrics['r2']:.4f}, rmse {trial_metrics['rmse']:.4f}. "
                "Should we keep this feature? Reply yes or no."
            )

        llm_decision = _query_llm(prompt)
        keep = None
        if llm_decision:
            keep = llm_decision.strip().lower().startswith("y")

        if keep is None:
            # simple heuristic fallback
            if state.task_type == "classification":
                keep = delta_f1 > 0 or delta_acc > 0
            else:
                keep = delta_r2 > 0 or delta_rmse < 0

        if keep:
            kept_features.append(feat)
            current_df = trial_df
            current_metrics = trial_metrics
            if state.task_type == "classification":
                state.append_log(
                    f"FeatureSelection: kept {feat} (+{delta_f1:.4f} f1, +{delta_acc:.4f} acc)"
                )
            else:
                state.append_log(
                    f"FeatureSelection: kept {feat} (+{delta_r2:.4f} r2, {delta_rmse:.4f} rmse change)"
                )
        else:
            if state.task_type == "classification":
                state.append_log(
                    f"FeatureSelection: dropped {feat} ({delta_f1:.4f} f1, {delta_acc:.4f} acc)"
                )
            else:
                state.append_log(
                    f"FeatureSelection: dropped {feat} ({delta_r2:.4f} r2, {delta_rmse:.4f} rmse change)"
                )

    # Update dataframe and feature list
    state.df = current_df
    state.features = kept_features

    return state
