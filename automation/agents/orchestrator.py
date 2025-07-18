from __future__ import annotations

"""LLM-driven orchestration of data science agents."""

import json
from typing import Dict, cast

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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
from .ensemble_agent import EnsembleAgent
from . import model_evaluation
from .. import code_assembler
from .baseline_agent import BaselineAgent


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


def _evaluate_preprocessing(snippet, df, target, task_type):
    trial_df = df.copy()
    try:
        exec(snippet, {"pd": pd}, {"df": trial_df, "target": target})
    except Exception as e:
        return False, f"Preprocessing failed: {e}", None
    # Data integrity check
    if trial_df.isnull().sum().sum() > 0:
        return False, "Preprocessing introduced new missing values", None
    # Model performance check
    X = trial_df.drop(columns=[target])
    y = trial_df[target]
    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=10, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=10, random_state=42)
    try:
        model.fit(X, y)
        score = model.score(X, y)
    except Exception as e:
        return False, f"Model training failed after preprocessing: {e}", None
    return True, f"Preprocessing accepted, model score: {score}", trial_df


def _run_decided_steps(state: PipelineState) -> PipelineState:
    """Run a round of agents based on LLM decisions."""

    # Use a default if state.task_type is None
    task_type = state.task_type if state.task_type is not None else 'classification'

    if state.current_score is None:
        state.current_score = _compute_score(state.df, state.target, task_type)

    prev_best_score = state.best_score if state.best_score is not None else state.current_score

    decisions = _decide_steps(state)

    # Always run and evaluate preprocessing code first if requested
    if decisions.get("preprocessing", {}).get("run", True):
        state = STEP_AGENTS["preprocessing"].run(state)
        for snippet in list(state.pending_code.get("preprocessing", [])):
            ok, msg, new_df = _evaluate_preprocessing(snippet, state.df, state.target, task_type)
            if ok:
                if new_df is not None and isinstance(new_df, pd.DataFrame):
                    state.df = new_df
                state.append_code("preprocessing", snippet)
                state.append_log(f"preprocessing: accepted snippet ({msg})")
            else:
                state.append_log(f"preprocessing: rejected snippet ({msg})")
            state.pending_code["preprocessing"] = []

    # Now run the rest of the agents as before, skipping preprocessing
    for stage, agent in STEP_AGENTS.items():
        if stage == "preprocessing":
            continue
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
                retry_code = None
                if stage == "feature_implementation":
                    feat_desc = "; ".join(
                        f"{name} = {state.feature_formulas.get(name, '')}"
                        for name in state.features
                    )
                    prompt = (
                        "The previous code for implementing features failed with "
                        f"error: {exc}. The intended features are: {feat_desc}. "
                        "Provide corrected pandas code in JSON with key 'code'."
                    )
                    try:
                        llm_raw = _query_llm(prompt)
                        parsed = json.loads(llm_raw)
                        retry_code = parsed.get("code")
                    except Exception as e:  # noqa: BLE001
                        state.append_log(f"FeatureImplementation retry LLM failed: {e}")

                if retry_code:
                    try:
                        exec(retry_code, env, local_vars)
                        state.append_log("FeatureImplementation retry succeeded")
                        snippet = retry_code
                    except Exception as exc2:  # noqa: BLE001
                        state.append_log(f"FeatureImplementation retry failed: {exc2}")
                        state.snippet_history.append(
                            {
                                "iteration": state.iteration,
                                "stage": stage,
                                "snippet": retry_code,
                                "accepted": False,
                                "score": None,
                            }
                        )
                        continue
                else:
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
                    trial_df, state.target, task_type
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
                if stage == "feature_implementation":
                    # Log which features were added
                    new_cols = set(trial_df.columns) - set(state.df.columns) if hasattr(trial_df, 'columns') and hasattr(state.df, 'columns') else set()
                    if new_cols:
                        state.append_log(f"FeatureImplementation: added features {sorted(new_cols)}")
                if trial_df is not None and isinstance(trial_df, pd.DataFrame):
                    state.df = cast(pd.DataFrame, trial_df)
                else:
                    state.append_log(f"Warning: trial_df is not a DataFrame, skipping state.df update for {stage}.")
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

    state.current_score = _compute_score(state.df, state.target, task_type)

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


def is_model_ready(df, target):
    # Returns True if all features (except target) are numeric and no missing values
    X = df.drop(columns=[target])
    return X.select_dtypes(exclude=[float, int]).empty and not X.isnull().any().any()

def try_model_training(df, target, task_type):
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    X = df.drop(columns=[target])
    y = df[target]
    try:
        if task_type == 'classification':
            model = RandomForestClassifier(n_estimators=10, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        return True
    except Exception:
        return False


def run(state: PipelineState, max_iter: int = 10, patience: int = 5) -> PipelineState:
    """Run the pipeline with LLM-guided orchestration and iteration."""
    state.append_log("Orchestrator supervisor: booting pipeline")
    state.best_score = None
    state.no_improve_rounds = 0
    state.max_iter = max_iter
    state.patience = patience
    state.best_df = state.df.copy()
    state.best_code_blocks = {k: v.copy() for k, v in state.code_blocks.items()}
    state.best_features = list(state.features)
    for step in STEP_AGENTS:
        state.pending_code.setdefault(step, [])
        state.code_blocks.setdefault(step, [])
    state = TaskIdentificationAgent().run(state)
    state.iteration = 0
    state.iterate = True
    agentic_attempts = 0
    max_agentic_attempts = 3
    while state.iterate and state.iteration < max_iter:
        state.append_log(f"Orchestrator: starting iteration {state.iteration}")
        state = _run_decided_steps(state)
        agentic_attempts += 1
        # After model training and evaluation, try ensembling
        prev_score = state.best_score
        state = EnsembleAgent().run(state)
        # Optionally, check if ensemble improved the score and update state accordingly
        if state.best_score is not None and prev_score is not None and state.best_score > prev_score:
            state.append_log("EnsembleAgent: ensemble improved the score.")
        # Check if data is model-ready and model training works
        ready = is_model_ready(state.df, state.target)
        task_type = state.task_type if state.task_type is not None else 'classification'
        trained = try_model_training(state.df, state.target, task_type)
        if ready and trained:
            state.append_log("Data is model-ready and model training succeeded.")
            state.iteration += 1
            continue
        if agentic_attempts >= max_agentic_attempts:
            state.append_log("Agentic attempts exhausted. Invoking BaselineAgent as fallback.")
            state = BaselineAgent().run(state)
            # After BaselineAgent, check again
            ready = is_model_ready(state.df, state.target)
            trained = try_model_training(state.df, state.target, task_type)
            if ready and trained:
                state.append_log("BaselineAgent produced model-ready data. Continuing agentic improvements.")
                state.iteration += 1
                continue
            else:
                state.append_log("BaselineAgent fallback failed. Stopping pipeline.")
                break
        state.iteration += 1
    state = code_assembler.run(state)
    return state
