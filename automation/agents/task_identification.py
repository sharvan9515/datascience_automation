import json
from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent
from automation.utils import safe_json_parse


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt, expect_json=True)


class TaskIdentificationAgent(BaseAgent):
    """Determine the ML task type and note any immediate data issues."""

    def run(self, state: PipelineState) -> PipelineState:
        df = state.df

        schema = {col: str(df[col].dtype) for col in df.columns}
        stats = df.describe(include="all").to_dict()

        prompt = (
            "Given the following dataset schema and summary statistics, "
            "determine whether predicting the target column should be "
            "treated as classification or regression. "
            "Also mention any immediate data quality issues. "
            "Respond in JSON with keys 'task_type' and optional 'issues'.\n"
            f"Target: {state.target}\n"
            f"Schema: {json.dumps(schema)}\n"
            f"Stats: {json.dumps(stats, default=str)}"
        )

        try:
            llm_raw = _query_llm(prompt)
        except RuntimeError as exc:
            state.append_log(f"TaskIdentification: LLM query failed: {exc}")
            return state
        try:
            parsed = safe_json_parse(llm_raw)
        except Exception as exc:
            raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

        if not isinstance(parsed, dict) or "task_type" not in parsed:
            raise RuntimeError("LLM response missing required 'task_type'")

        state.task_type = parsed["task_type"].lower()
        if issues := parsed.get("issues"):
            state.append_log(f"TaskIdentification issues: {issues}")
        state.append_log(
            f"TaskIdentification: determined task_type={state.task_type} via LLM"
        )
        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    """Backwards compatible function API."""
    return TaskIdentificationAgent().run(state)
