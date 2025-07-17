import json
import pandas as pd
from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt, expect_json=True)


class Agent(BaseAgent):
    """Preprocessing agent."""

    def run(self, state: PipelineState) -> PipelineState:
        """Query the LLM for preprocessing code and store it for later execution."""
        state.append_log("Preprocessing supervisor: starting")

        df = state.df.copy()
        stage_name = "preprocessing"

        schema = {col: str(df[col].dtype) for col in df.columns}
        missing = df.isnull().sum().to_dict()

        base_prompt = (
            "You are a data preprocessing assistant. "
            "Given a pandas DataFrame `df` with schema "
            f"{json.dumps(schema)} and missing counts {json.dumps(missing)}, "
            "suggest preprocessing steps for machine learning. "
            "Return JSON with keys 'logs' (list of messages describing each step) "
            "and 'code' (Python pandas code that modifies df in place)."
        )

        llm_resp = _query_llm(base_prompt)
        try:
            parsed = json.loads(llm_resp)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

        if 'code' not in parsed:
            raise RuntimeError("LLM response missing 'code' field")

        code = parsed.get('code', '')
        logs = parsed.get('logs', [])

        for msg in logs:
            state.append_log(f"Preprocessing: {msg}")
        if rationale := parsed.get('rationale'):
            state.append_log(f"Preprocessing rationale: {rationale}")
        state.append_pending_code(stage_name, code)
        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state)
