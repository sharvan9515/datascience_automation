"""Evaluate a simple model and decide if another iteration is needed."""

from __future__ import annotations

import json
from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent
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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier

__all__ = ["compute_score", "run"]


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt, expect_json=True)


def compute_score(
    df: pd.DataFrame, target: str, task_type: str, time_col: str | None = None
) -> float:
    """Return a simple train/test score for the given dataset."""

    X = df.drop(columns=[target])
    y = df[target]
    X = X.copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes
    X = X.fillna(0)
    if time_col:
        df_sorted = df.sort_values(time_col)
        split_idx = int(len(df_sorted) * 0.8)
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average="weighted")

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)


class ModelEvaluationAgent(BaseAgent):
    """Evaluate model performance and track best scores."""

    def run(self, state: PipelineState) -> PipelineState:
        """Train a quick model, analyze metrics, and update state with hardcoded logic only."""
        state.append_log("Evaluator supervisor: starting")

        df = state.df
        X = df.drop(columns=[state.target]).copy()
        y = df[state.target]
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category").cat.codes
        X = X.fillna(0)
        if state.timeseries_mode and state.time_col:
            df_sorted = df.sort_values(state.time_col)
            split_idx = int(len(df_sorted) * 0.8)
            train_df = df_sorted.iloc[:split_idx]
            test_df = df_sorted.iloc[split_idx:]
            X_train = train_df.drop(columns=[state.target])
            y_train = train_df[state.target]
            X_test = test_df.drop(columns=[state.target])
            y_test = test_df[state.target]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        if state.task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
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

        prev_best = state.best_score
        tol = 0.01
        improved = prev_best is None or new_score > prev_best + tol
        if improved:
            state.best_score = new_score
            state.no_improve_rounds = 0
        else:
            state.no_improve_rounds += 1

        state.iterate = state.no_improve_rounds < state.patience

        log_msg = f"ModelEvaluation decision: iterate={state.iterate} (hardcoded logic)"
        state.append_log(log_msg)

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


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    """Backwards compatible function API."""
    return ModelEvaluationAgent().run(state)
