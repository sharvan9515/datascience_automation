from __future__ import annotations

"""LLM-driven orchestration of data science agents."""

import json
import os
from typing import Dict

from automation.pipeline_state import PipelineState
from . import (
    task_identification,
    preprocessing,
    correlation_eda,
    feature_ideation,
    feature_implementation,
    feature_selection,
    feature_reduction,
    model_training,
    model_evaluation,
)
from .. import code_assembler


STEP_AGENTS = {
    "preprocessing": preprocessing,
    "correlation_eda": correlation_eda,
    "feature_ideation": feature_ideation,
    "feature_implementation": feature_implementation,
    "feature_selection": feature_selection,
    "feature_reduction": feature_reduction,
}


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


def _decide_steps(state: PipelineState) -> Dict[str, Dict[str, object]]:
    """Ask the LLM which steps to run and return decisions."""

    df = state.df
    schema = {c: str(df[c].dtype) for c in df.columns}
    prompt = (
        "You orchestrate an automated ML pipeline. "
        "Decide whether to run these steps next: preprocessing, correlation_eda, "
        "feature_ideation, feature_selection, feature_reduction. "
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
    return decisions


def _run_decided_steps(state: PipelineState) -> PipelineState:
    """Run a round of agents based on LLM decisions."""

    decisions = _decide_steps(state)
    for step, agent in STEP_AGENTS.items():
        decision = decisions.get(step, {"run": True, "reason": ""})
        run_step = bool(decision.get("run"))
        reason = str(decision.get("reason", ""))
        state.append_log(f"Orchestrator decision: {step}={run_step} - {reason}")
        if run_step:
            state = agent.run(state)
    state = model_training.run(state)
    state = model_evaluation.run(state)
    return state


def run(state: PipelineState, max_iter: int = 3) -> PipelineState:
    """Run the pipeline with LLM-guided orchestration and iteration."""

    state.best_score = None
    state.no_improve_rounds = 0
    state.max_iter = max_iter

    state = task_identification.run(state)
    state.iteration = 0
    state = _run_decided_steps(state)

    while state.iterate and state.iteration < max_iter:
        state.iteration += 1
        state.append_log(f"Orchestrator: starting iteration {state.iteration}")
        state = _run_decided_steps(state)

    state = code_assembler.run(state)
    return state
