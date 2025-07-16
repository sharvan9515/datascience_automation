"""Incrementally evaluate new features and keep only useful ones."""

from __future__ import annotations

import os
from automation.pipeline_state import PipelineState
from . import model_evaluation


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




def run(state: PipelineState) -> PipelineState:
    """Evaluate new features incrementally and keep only beneficial ones."""

    df = state.df.copy()
    stage_name = "feature_selection"
    new_feats = [f for f in state.features if f in df.columns]

    # Baseline without proposed features
    baseline_df = df.drop(columns=new_feats, errors="ignore")
    baseline_score = model_evaluation.compute_score(
        baseline_df, state.target, state.task_type
    )
    state.append_log(
        f"FeatureSelection: baseline score={baseline_score:.4f}"
    )

    current_df = baseline_df
    current_score = baseline_score
    kept_features: list[str] = []

    for feat in new_feats:
        trial_df = current_df.join(df[[feat]])
        trial_score = model_evaluation.compute_score(
            trial_df, state.target, state.task_type
        )
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
