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


STEP_AGENTS = {
    "preprocessing": preprocessing,
    "correlation_eda": correlation_eda,
    "feature_ideation": feature_ideation,
    "feature_implementation": feature_implementation,
    "feature_selection": feature_selection,
    "feature_reduction": feature_reduction,
}


def _query_llm(prompt: str) -> str | None:
    """Return raw LLM response or ``None`` if the call fails."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        import openai
    except Exception:
        return None

    client = openai.OpenAI(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return None


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
    decisions: Dict[str, Dict[str, object]] = {}
    if llm_raw:
        try:
            parsed = json.loads(llm_raw)
            if isinstance(parsed, dict):
                for step in STEP_AGENTS:
                    entry = parsed.get(step, {})
                    if isinstance(entry, dict):
                        run_flag = str(entry.get("run", "")).lower().startswith("y")
                        reason = entry.get("reason", "")
                    else:
                        run_flag = str(entry).lower().startswith("y")
                        reason = ""
                    decisions[step] = {"run": run_flag, "reason": reason}
        except Exception:
            decisions = {}

    # fallback: run everything if parsing failed
    for step in STEP_AGENTS:
        decisions.setdefault(step, {"run": True, "reason": "fallback: default yes"})
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

    state = task_identification.run(state)
    iteration = 0
    state = _run_decided_steps(state)

    while state.iterate and iteration < max_iter:
        iteration += 1
        state.append_log(f"Orchestrator: starting iteration {iteration}")
        state = _run_decided_steps(state)
    return state
