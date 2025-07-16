import json
import os

from automation.pipeline_state import PipelineState


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
    """Ask the LLM for new feature ideas and store them in the state."""
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
