from __future__ import annotations

"""LLM-driven orchestration of data science agents."""

import json
from typing import Dict

import pandas as pd
from sklearn.decomposition import PCA

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .task_identification import Agent as TaskIdentificationAgent
from .preprocessing import Agent as PreprocessingAgent
from .correlation_eda import Agent as CorrelationEDAAgent
from .feature_ideation import Agent as FeatureIdeationAgent
from .feature_implementation import Agent as FeatureImplementationAgent
from .feature_selection import Agent as FeatureSelectionAgent
from .feature_reduction import Agent as FeatureReductionAgent
from .model_training import Agent as ModelTrainingAgent
from .model_evaluation import Agent as ModelEvaluationAgent
from .hyperparameter_search import Agent as HyperparameterSearchAgent
from . import model_evaluation
from .. import code_assembler


def _compute_score(df: pd.DataFrame, target: str, task_type: str) -> float:
    """Wrapper around :func:`model_evaluation.compute_score`."""

    return model_evaluation.compute_score(df, target, task_type)


STEP_AGENTS = {
    "preprocessing": PreprocessingAgent(),
    "correlation_eda": CorrelationEDAAgent(),
    "feature_ideation": FeatureIdeationAgent(),
    "feature_implementation": FeatureImplementationAgent(),
    "feature_selection": FeatureSelectionAgent(),
    "feature_reduction": FeatureReductionAgent(),
}


def _query_llm(prompt: str) -> str:
    """Query the LLM with a few-shot example for deterministic JSON output."""

    example_user = (
        "Rows: 10\nSchema: {'a': 'int64', 'b': 'float64'}\nRecent logs: []"
    )
    example_assistant = json.dumps(
        {
            "preprocessing": {"run": "yes", "reason": "clean"},
            "correlation_eda": {"run": "no", "reason": "few features"},
            "feature_ideation": {"run": "yes", "reason": "improve"},
            "feature_implementation": {"run": "yes", "reason": "apply"},
            "feature_selection": {"run": "no", "reason": "none yet"},
            "feature_reduction": {"run": "no", "reason": "small"},
        }
    )

    return query_llm(
        prompt, few_shot=[(example_user, example_assistant)], expect_json=True
    )


def _decide_steps(state: PipelineState) -> Dict[str, Dict[str, object]]:
    """Ask the LLM which steps to run and return decisions."""

    df = state.df
    schema = {c: str(df[c].dtype) for c in df.columns}
    prompt = (
        "You orchestrate an automated ML pipeline. "
        "Decide whether to run these steps next: preprocessing, correlation_eda, "
        "feature_ideation, feature_implementation, feature_selection, "
        "feature_reduction. "
        "Return JSON where each key maps to an object with 'run' (yes/no) and "
        "'reason'.\n"
        f"Rows: {len(df)}\nSchema: {json.dumps(schema)}\n"
        f"Recent logs: {state.log[-3:]}"
    )

    llm_raw = _query_llm(prompt)
    try:
        parsed = json.loads(llm_raw)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError("LLM response must be a JSON object")

    decisions: Dict[str, Dict[str, object]] = {}
    for step in STEP_AGENTS:
        entry = parsed.get(step)
        if not isinstance(entry, dict) or "run" not in entry:
            raise RuntimeError(f"LLM response missing decision for step '{step}'")
        run_flag = str(entry.get("run", "")).lower().startswith("y")
        reason = entry.get("reason", "")
        decisions[step] = {"run": run_flag, "reason": reason}

    if decisions.get("feature_ideation", {}).get("run"):
        decisions.setdefault("feature_implementation", {})["run"] = True
    return decisions


def _run_decided_steps(state: PipelineState) -> PipelineState:
    """Run a round of agents based on LLM decisions."""

    if state.current_score is None:
        state.current_score = _compute_score(state.df, state.target, state.task_type)

    prev_best_score = state.best_score if state.best_score is not None else state.current_score

    decisions = _decide_steps(state)

    for stage, agent in STEP_AGENTS.items():
        decision = decisions.get(stage, {"run": True, "reason": ""})
        run_step = bool(decision.get("run"))
        reason = str(decision.get("reason", ""))
        state.append_log(f"Orchestrator decision: {stage}={run_step} - {reason}")
        if not run_step:
            continue

        state = agent.run(state)

        for snippet in list(state.pending_code.get(stage, [])):
            trial_df = state.df.copy()
            env = {"pd": pd, "PCA": PCA}
            local_vars = {"df": trial_df, "target": state.target}
            try:
                exec(snippet, env, local_vars)
            except Exception as exc:  # noqa: BLE001
                state.append_log(f"{stage} snippet failed: {exc}")
                state.snippet_history.append(
                    {
                        "iteration": state.iteration,
                        "stage": stage,
                        "snippet": snippet,
                        "accepted": False,
                        "score": None,
                    }
                )
                continue

            trial_df = local_vars.get("df", trial_df)
            try:
                trial_score = _compute_score(
                    trial_df, state.target, state.task_type
                )
            except Exception as exc:  # noqa: BLE001
                state.append_log(f"{stage}: scoring failed: {exc}")
                state.snippet_history.append(
                    {
                        "iteration": state.iteration,
                        "stage": stage,
                        "snippet": snippet,
                        "accepted": False,
                        "score": None,
                    }
                )
                continue

            if trial_score > (state.current_score or 0.0):
                delta = trial_score - (state.current_score or 0.0)
                state.append_log(f"{stage}: accepted snippet (+{delta:.4f} score)")
                state.df = trial_df
                state.current_score = trial_score
                state.append_code(stage, snippet)
                state.snippet_history.append(
                    {
                        "iteration": state.iteration,
                        "stage": stage,
                        "snippet": snippet,
                        "accepted": True,
                        "score": trial_score,
                    }
                )
            else:
                state.append_log(
                    f"{stage}: rejected snippet ({trial_score:.4f} <= {state.current_score:.4f})"
                )
                state.snippet_history.append(
                    {
                        "iteration": state.iteration,
                        "stage": stage,
                        "snippet": snippet,
                        "accepted": False,
                        "score": trial_score,
                    }
                )

        state.pending_code[stage] = []

    state = ModelTrainingAgent().run(state)
    state = ModelEvaluationAgent().run(state)
    if state.no_improve_rounds >= max(1, state.patience // 2):
        state.append_log("Orchestrator: triggering hyperparameter search")
        state = HyperparameterSearchAgent().run(state)

    state.current_score = _compute_score(state.df, state.target, state.task_type)

    if state.best_score is not None and state.best_score > prev_best_score:
        state.best_df = state.df.copy()
        state.best_code_blocks = {k: v.copy() for k, v in state.code_blocks.items()}
        state.best_features = list(state.features)
    elif state.best_score == prev_best_score and state.best_df is not None:
        state.append_log("No improvement this iteration; reverting to last best state.")
        state.df = state.best_df.copy()
        state.code_blocks = {k: v.copy() for k, v in state.best_code_blocks.items()}
        state.features = list(state.best_features)
        state.current_score = prev_best_score

    return state


def run(state: PipelineState, max_iter: int = 10, patience: int = 5) -> PipelineState:
    """Run the pipeline with LLM-guided orchestration and iteration.

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations to perform.
    patience : int
        Stop after this many iterations without improvement.
    """

    state.append_log("Orchestrator supervisor: booting pipeline")

    state.best_score = None
    state.no_improve_rounds = 0
    state.max_iter = max_iter
    state.patience = patience
    state.best_df = state.df.copy()
    state.best_code_blocks = {k: v.copy() for k, v in state.code_blocks.items()}
    state.best_features = list(state.features)

    # Initialize code tracking structures for each step
    for step in STEP_AGENTS:
        state.pending_code.setdefault(step, [])
        state.code_blocks.setdefault(step, [])

    state = TaskIdentificationAgent().run(state)
    state.iteration = 0
    state.iterate = True

    while state.iterate and state.iteration < max_iter:
        state.append_log(f"Orchestrator: starting iteration {state.iteration}")
        state = _run_decided_steps(state)
        state.iteration += 1

    state = code_assembler.run(state)
    return state
