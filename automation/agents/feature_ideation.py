import json

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt, expect_json=True)


def run(state: PipelineState) -> PipelineState:
    """Ask the LLM for new feature ideas and store them in the state."""
    state.append_log("Feature engineering supervisor: ideation start")

    feature_cols = [c for c in state.df.columns if c != state.target]

    prompt = (
        "You are a feature engineering assistant. "
        f"The current task type is {state.task_type}. "
        f"Existing features are: {feature_cols}. "
        "Propose up to 3 new features that could improve model performance. "
        "Return JSON list where each item has keys 'name', 'formula', and 'rationale'."
    )

    llm_raw = _query_llm(prompt)
    try:
        proposals = json.loads(llm_raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse LLM response: {exc}") from exc

    if isinstance(proposals, dict):
        proposals = proposals.get("features", [])

    if not isinstance(proposals, list) or not proposals:
        raise RuntimeError("LLM did not return any feature proposals")

    for prop in proposals:
        name = prop.get("name")
        formula = prop.get("formula")
        rationale = prop.get("rationale")
        if not name or not formula:
            raise RuntimeError("LLM proposal missing 'name' or 'formula'")
        state.features.append(name)
        state.append_log(
            f"FeatureIdeation: propose {name} = {formula} because {rationale}"
        )

    return state
