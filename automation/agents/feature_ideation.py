import json
import os

from automation.pipeline_state import PipelineState


def _query_llm(prompt: str) -> str | None:
    """Return raw LLM response or None if call fails."""
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
    proposals = None
    if llm_raw:
        try:
            proposals = json.loads(llm_raw)
        except json.JSONDecodeError:
            proposals = None

    if isinstance(proposals, list):
        for prop in proposals:
            name = prop.get("name")
            formula = prop.get("formula")
            rationale = prop.get("rationale")
            if name and formula:
                state.features.append(name)
                state.append_log(
                    f"FeatureIdeation: propose {name} = {formula} because {rationale}"
                )
        if proposals:
            return state

    # Fallback: simple ratio of first two numeric columns
    numeric_cols = [c for c in state.df.columns if state.df[c].dtype != "O" and c != state.target]
    if len(numeric_cols) >= 2:
        feat_name = f"{numeric_cols[0]}_over_{numeric_cols[1]}"
        formula = f"{numeric_cols[0]} / ({numeric_cols[1]} + 1e-6)"
        state.append_log(f"FeatureIdeation fallback: {feat_name} = {formula}")
        state.features.append(feat_name)
    return state
