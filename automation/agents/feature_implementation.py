"""Generate feature implementation code via LLM and queue it for later execution."""

from __future__ import annotations

import json

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent



def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt, expect_json=True)


class Agent(BaseAgent):
    """Feature implementation agent."""

    def run(self, state: PipelineState) -> PipelineState:
        """Generate pandas code for each feature and queue it for validation."""
        state.append_log("Feature engineering supervisor: implementation start")

        stage_name = "feature_implementation"

        if not state.features:
            # Nothing to implement
            return state

        schema = {col: str(state.df[col].dtype) for col in state.df.columns}

        base_prompt = (
            "You are a pandas expert. Given a DataFrame `df` with columns "
            f"{json.dumps(schema)}, implement the following new features: "
            f"{state.features}. Return JSON with 'code' (Python code modifying df in"
            " place) and 'logs' (one message per feature describing the action)."
        )

        llm_resp = _query_llm(base_prompt)
        try:
            parsed: dict[str, object] = json.loads(llm_resp)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

        if "code" not in parsed:
            raise RuntimeError("LLM response missing 'code'")

        code = parsed.get("code", "")
        logs = parsed.get("logs", [])
        for msg in logs or []:
            state.append_log(f"FeatureImplementation: {msg}")

        state.append_pending_code(stage_name, code)
        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state)
