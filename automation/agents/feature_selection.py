"""Incrementally evaluate new features and keep only useful ones."""

from __future__ import annotations

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from . import model_evaluation
from .base import BaseAgent
from .preprocessing import ensure_numeric_features
from concurrent.futures import ThreadPoolExecutor
import sklearn.model_selection


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt)




class Agent(BaseAgent):
    """Feature selection agent."""

    def run(self, state: PipelineState) -> PipelineState:
        """Evaluate new features incrementally and keep only beneficial ones."""
        state.append_log("Feature engineering supervisor: selection start")

        df = state.df.copy()
        stage_name = "feature_selection"
        new_feats = [f for f in state.features if f in df.columns]

        # Ensure task_type is always a string
        task_type = state.task_type if state.task_type is not None else 'classification'

        from sklearn.model_selection import cross_val_score
        # Baseline without proposed features
        baseline_df = df.drop(columns=new_feats, errors="ignore")
        baseline_df = ensure_numeric_features(baseline_df, state.target, state)
        baseline_X = baseline_df.drop(columns=[state.target], errors="ignore")
        baseline_y = baseline_df[state.target]
        baseline_X = baseline_X.fillna(0)
        # Use cross-validation for baseline score
        if task_type == "classification":
            from sklearn.ensemble import RandomForestClassifier
            baseline_model = RandomForestClassifier(n_estimators=10, random_state=42)
            baseline_score = cross_val_score(baseline_model, baseline_X, baseline_y, cv=3, scoring="accuracy").mean()
        else:
            from sklearn.ensemble import RandomForestRegressor
            baseline_model = RandomForestRegressor(n_estimators=10, random_state=42)
            baseline_score = cross_val_score(baseline_model, baseline_X, baseline_y, cv=3, scoring="r2").mean()
        state.append_log(
            f"FeatureSelection: baseline CV score={baseline_score:.4f}"
        )

        current_df = baseline_df
        current_score = baseline_score
        # Evaluate features in sets for synergy
        kept_features: list[str] = []
        feature_sets = []
        for i, feat in enumerate(new_feats):
            # Try adding each feature to the current set
            trial_set = kept_features + [feat]
            trial_df = df[trial_set + [state.target]].copy()
            trial_df = ensure_numeric_features(trial_df, state.target, state)
            trial_X = trial_df.drop(columns=[state.target], errors="ignore")
            trial_y = trial_df[state.target]
            trial_X = trial_X.fillna(0)
            if task_type == "classification":
                model = RandomForestClassifier(n_estimators=10, random_state=42)
                trial_score = cross_val_score(model, trial_X, trial_y, cv=3, scoring="accuracy").mean()
            else:
                model = RandomForestRegressor(n_estimators=10, random_state=42)
                trial_score = cross_val_score(model, trial_X, trial_y, cv=3, scoring="r2").mean()
            delta = trial_score - current_score
            # Accept the set if the score is neutral or slightly negative (within delta)
            delta_accept = -0.05
            if delta >= delta_accept:
                kept_features.append(feat)
                current_df = trial_df
                current_score = trial_score
                state.append_log(f"FeatureSelection: kept set {kept_features} ({delta:+.4f} score)")
            else:
                state.append_log(f"FeatureSelection: dropped {feat} ({delta:+.4f} score)")
        # At the end, keep the set if it improves or is neutral
        state.features = kept_features
        state.df = current_df

        # Update must_keep in state to only those features that improved score
        if hasattr(state, 'must_keep'):
            state.must_keep = [state.target] + kept_features
        else:
            setattr(state, 'must_keep', [state.target] + kept_features)

        snippet = f"df = df[['{state.target}'] + {kept_features!r}]"
        state.append_pending_code(stage_name, snippet)

        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state)
