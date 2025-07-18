"""Incrementally evaluate new features and keep only useful ones."""

from __future__ import annotations

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from . import model_evaluation
from .base import BaseAgent
from concurrent.futures import ThreadPoolExecutor


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

        # Baseline without proposed features
        baseline_df = df.drop(columns=new_feats, errors="ignore")
        baseline_score = model_evaluation.compute_score(
            baseline_df, state.target, task_type
        )
        state.append_log(
            f"FeatureSelection: baseline score={baseline_score:.4f}"
        )

        current_df = baseline_df
        current_score = baseline_score
        kept_features: list[str] = []

        results = []
        def evaluate_feature(feat):
            trial_df = current_df.join(df[[feat]])
            trial_score = model_evaluation.compute_score(
                trial_df, state.target, task_type
            )
            return feat, trial_df, trial_score

        # Parallelize feature evaluation
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(evaluate_feature, new_feats))

        for feat, trial_df, trial_score in results:
            delta = trial_score - current_score
            prompt = (
                f"Baseline score {current_score:.4f}. "
                f"After adding feature '{feat}', score {trial_score:.4f}. "
                "Should we keep this feature? Reply yes or no."
            )
            llm_decision = _query_llm(prompt)
            if not llm_decision:
                raise RuntimeError("LLM failed to return feature selection decision")
            keep = llm_decision.strip().lower().startswith("y")

            if keep:
                kept_features.append(feat)
                current_df = trial_df
                current_score = trial_score
                state.append_log(
                    f"FeatureSelection: kept {feat} ({delta:+.4f} score)"
                )
            else:
                state.append_log(
                    f"FeatureSelection: dropped {feat} ({delta:+.4f} score)"
                )

        snippet = f"df = df[['{state.target}'] + {kept_features!r}]"
        state.append_pending_code(stage_name, snippet)

        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state)
