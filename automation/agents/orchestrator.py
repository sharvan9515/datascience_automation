from __future__ import annotations

"""LLM-driven orchestration of data science agents."""

import json
from typing import Dict, cast

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from automation.utils.sandbox import safe_exec
from automation.validators import CodeQualityValidator
from .task_identification import TaskIdentificationAgent
from .preprocessing import PreprocessingAgent
from .correlation_eda import CorrelationEDAAgent
from .feature_ideation import FeatureIdeationAgent
from .feature_implementation import FeatureImplementationAgent
from .feature_selection import FeatureSelectionAgent
from .feature_reduction import FeatureReductionAgent
from .model_training import ModelTrainingAgent
from .model_evaluation import ModelEvaluationAgent
from .hyperparameter_search import HyperparameterSearchAgent
from .ensemble_agent import EnsembleAgent
from ..intelligent_model_selector import IntelligentModelSelector
from . import model_evaluation
from .. import code_assembler
from .baseline_agent import BaselineAgent


class Orchestrator:
    """Coordinate all agents to build a full ML pipeline."""

    STEP_AGENTS = {
        "preprocessing": PreprocessingAgent(),
        "correlation_eda": CorrelationEDAAgent(),
        "feature_ideation": FeatureIdeationAgent(),
        "feature_implementation": FeatureImplementationAgent(),
        "feature_selection": FeatureSelectionAgent(),
        "feature_reduction": FeatureReductionAgent(),
    }


    def _compute_score(self, df: pd.DataFrame, target: str, task_type: str) -> float:
        """Wrapper around :func:`model_evaluation.compute_score`."""

        return model_evaluation.compute_score(df, target, task_type)


    def _query_llm(self, prompt: str) -> str:
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


    def _decide_steps(self, state: PipelineState) -> Dict[str, Dict[str, object]]:
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

        try:
            llm_raw = self._query_llm(prompt)
        except RuntimeError as exc:
            state.append_log(f"Orchestrator: LLM query failed: {exc}")
            return {}
        try:
            parsed = json.loads(llm_raw)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

        if not isinstance(parsed, dict):
            raise RuntimeError("LLM response must be a JSON object")

        decisions: Dict[str, Dict[str, object]] = {}
        for step in self.STEP_AGENTS:
            entry = parsed.get(step)
            if not isinstance(entry, dict) or "run" not in entry:
                raise RuntimeError(f"LLM response missing decision for step '{step}'")
            run_flag = str(entry.get("run", "")).lower().startswith("y")
            reason = entry.get("reason", "")
            decisions[step] = {"run": run_flag, "reason": reason}

        if decisions.get("feature_ideation", {}).get("run"):
            decisions.setdefault("feature_implementation", {})["run"] = True
        return decisions


    def _evaluate_preprocessing(self, snippet, df, target, task_type):
        ok, msg, new_df, _ = CodeQualityValidator.validate_preprocessing_code(
            snippet, df, target, task_type
        )
        return ok, msg, new_df


    def _run_decided_steps(self, state: PipelineState) -> PipelineState:
        """Run a round of agents based on LLM decisions."""

        task_type = state.task_type if state.task_type is not None else "classification"

        if state.current_score is None:
            state.current_score = self._compute_score(state.df, state.target, task_type)

        prev_best_score = state.best_score if state.best_score is not None else state.current_score

        decisions = self._decide_steps(state)

        if decisions.get("preprocessing", {}).get("run", True):
            state = self.STEP_AGENTS["preprocessing"].run(state)
            for snippet in list(state.pending_code.get("preprocessing", [])):
                ok, msg, new_df = self._evaluate_preprocessing(snippet, state.df, state.target, task_type)
                if ok:
                    if isinstance(new_df, pd.DataFrame):
                        state.df = new_df
                    state.append_code("preprocessing", snippet)
                    state.append_log(f"preprocessing: accepted snippet ({msg})")
                else:
                    state.append_log(f"preprocessing: rejected snippet ({msg})")
            state.pending_code["preprocessing"] = []

        for stage, agent in self.STEP_AGENTS.items():
            if stage == "preprocessing":
                continue
            decision = decisions.get(stage, {"run": True, "reason": ""})
            run_step = bool(decision.get("run"))
            reason = str(decision.get("reason", ""))
            state.append_log(f"Orchestrator decision: {stage}={run_step} - {reason}")
            if not run_step:
                continue

            for snippet in list(state.pending_code.get(stage, [])):
                ok, msg, trial_df, trial_score = CodeQualityValidator.validate_feature_code(
                    snippet, state.df, state.target, task_type
                )
                if not ok:
                    state.append_log(f"{stage}: validation failed - {msg}")
                    continue

                delta = 0.05
                if trial_score >= (state.current_score or 0.0) - delta:
                    if isinstance(trial_df, pd.DataFrame):
                        state.df = trial_df
                    state.current_score = trial_score
                    state.append_code(stage, snippet)
                    state.snippet_history.append({"iteration": state.iteration, "stage": stage, "snippet": snippet, "accepted": True, "score": trial_score})
                else:
                    state.append_log(f"{stage}: rejected snippet ({trial_score:.4f} < {state.current_score:.4f} - delta)")
                    state.snippet_history.append({"iteration": state.iteration, "stage": stage, "snippet": snippet, "accepted": False, "score": trial_score})
            state.pending_code[stage] = []

        state = ModelTrainingAgent().run(state)
        for snippet in list(state.pending_code.get("model_training", [])):
            ok, msg, _, _ = CodeQualityValidator.validate_generic_code(
                snippet, state.df, state.target, task_type
            )
            if ok:
                state.append_code("model_training", snippet)
                state.append_log(f"model_training: accepted snippet ({msg})")
            else:
                state.append_log(f"model_training: rejected snippet ({msg})")
        state.pending_code["model_training"] = []

        state = ModelEvaluationAgent().run(state)
        run_hyper = state.no_improve_rounds >= max(1, state.patience // 2)
        if run_hyper:
            if any("randomforest" in alg.lower() for alg in state.recommended_algorithms):
                state.append_log("Orchestrator: triggering hyperparameter search")
                state = HyperparameterSearchAgent().run(state)
            else:
                state.append_log("Orchestrator: skipping hyperparameter search (algorithms not recommended)")

        state.current_score = self._compute_score(state.df, state.target, task_type)

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
    """Return ``True`` if all features (except the target) are numeric and have no
    missing values."""

    import numpy as np

    X = df.drop(columns=[target])
    # ``exclude=[np.number]`` covers all numeric dtypes (int8, int16, float32, etc.)
    non_numeric = X.select_dtypes(exclude=[np.number])
    return non_numeric.empty and not X.isnull().any().any()

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


    def run_pipeline(self, state: PipelineState, patience: int = 20, score_threshold: float = 0.80) -> PipelineState:
        """Run the pipeline with LLM-guided orchestration and iteration."""
        state.append_log("Orchestrator supervisor: booting pipeline")
        state.best_score = None
        state.no_improve_rounds = 0
        state.max_iter = 0  # deprecated
        state.patience = patience
        state.best_df = state.df.copy()
        state.best_code_blocks = {k: v.copy() for k, v in state.code_blocks.items()}
        state.best_features = list(state.features)
        for step in self.STEP_AGENTS:
            state.pending_code.setdefault(step, [])
            state.code_blocks.setdefault(step, [])
        state = TaskIdentificationAgent().run(state)
        if state.profile is None:
            from automation.dataset_profiler import EnhancedDatasetProfiler
            state.append_log("Orchestrator: computing dataset profile")
            state.profile = EnhancedDatasetProfiler.generate_comprehensive_profile(state.df, state.target)
        state.recommended_algorithms = IntelligentModelSelector.select_optimal_algorithms(
            state.profile, state.task_type or "classification"
        )
        state.iteration = 0
        state.iterate = True
        agentic_attempts = 0
        max_agentic_attempts = 3
        while state.iterate:
            state.append_log(f"Orchestrator: starting iteration {state.iteration}")
            snapshot_version = state.create_snapshot()
            try:
                state = self._run_decided_steps(state)
                agentic_attempts += 1
                prev_score = state.best_score
                if len(state.recommended_algorithms) > 1:
                    state = EnsembleAgent().run(state)
                    if (
                        state.best_score is not None
                        and prev_score is not None
                        and state.best_score > prev_score
                    ):
                        state.append_log("EnsembleAgent: ensemble improved the score.")
            except Exception as exc:  # noqa: BLE001
                state.rollback_to(snapshot_version)
                state.append_log(
                    f"Orchestrator: rolled back to version {snapshot_version} due to error: {exc}"
                )
                break
            if state.no_improve_rounds >= patience:
                if any("randomforest" in alg.lower() for alg in state.recommended_algorithms):
                    state.append_log(
                        f"No improvement for {patience} consecutive iterations. Triggering hyperparameter search."
                    )
                    state = HyperparameterSearchAgent().run(state)
                else:
                    state.append_log(
                        f"No improvement for {patience} rounds but hyperparameter search skipped (algorithms not recommended)."
                    )
                break
            ready = is_model_ready(state.df, state.target)
            task_type = state.task_type if state.task_type is not None else "classification"
            trained = try_model_training(state.df, state.target, task_type)
            if ready and trained:
                state.append_log("Data is model-ready and model training succeeded.")
                state.iteration += 1
                continue
            if agentic_attempts >= max_agentic_attempts:
                state.append_log("Agentic attempts exhausted. Invoking BaselineAgent as fallback.")
                state = BaselineAgent().run(state)
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
        if state.no_improve_rounds >= patience:
            if any("randomforest" in alg.lower() for alg in state.recommended_algorithms):
                state.append_log("Triggering hyperparameter search after iterations.")
                state = HyperparameterSearchAgent().run(state)
            else:
                state.append_log(
                    "Hyperparameter search skipped after iterations (algorithms not recommended)."
                )
        import pprint
        print("[DEBUG] state.pending_code['feature_implementation']:")
        pprint.pprint(state.pending_code.get('feature_implementation', []))
        print("[DEBUG] state.code_blocks['feature_implementation']:")
        pprint.pprint(state.code_blocks.get('feature_implementation', []))
        print("[DEBUG] state.pending_code['feature_selection']:")
        pprint.pprint(state.pending_code.get('feature_selection', []))
        print("[DEBUG] state.code_blocks['feature_selection']:")
        pprint.pprint(state.code_blocks.get('feature_selection', []))
        state = code_assembler.run(state)
        return state


def run(state: PipelineState, patience: int = 20, score_threshold: float = 0.80) -> PipelineState:
    """Backwards compatible function API."""
    return Orchestrator().run_pipeline(state, patience=patience, score_threshold=score_threshold)
