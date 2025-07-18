import json

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt, expect_json=True)


class Agent(BaseAgent):
    """Feature ideation agent."""

    def run(self, state: PipelineState) -> PipelineState:
        """Ask the LLM for new feature ideas and store them in the state."""
        state.append_log("Feature engineering supervisor: ideation start")

        feature_cols = [c for c in state.df.columns if c != state.target]
        existing = sorted(set(feature_cols) | state.known_features)

        # Dynamically build few-shot example from state
        if state.feature_ideas:
            example_existing = [f["name"] for f in state.feature_ideas if "name" in f]
            example_prompt = (
                "You are a creative feature engineering assistant. "
                f"Existing features: {example_existing}. "
                "Propose up to 3 new features that could improve model performance. "
                "Return JSON list where each item has keys 'name', 'formula', and 'rationale'."
            )
            example_response = json.dumps(state.feature_ideas, indent=2)
        else:
            example_existing = [c for c in state.df.columns if c != state.target]
            example_prompt = (
                "You are a creative feature engineering assistant. "
                f"Existing features: {example_existing}. "
                "Propose up to 3 new features that could improve model performance. "
                "Return JSON list where each item has keys 'name', 'formula', and 'rationale'."
            )
            example_response = json.dumps([
                {"name": "Feature1", "formula": "...", "rationale": "..."}
            ], indent=2)
        # Main prompt for the current dataset
        prompt = (
            "You are a creative feature engineering assistant. "
            f"The current task type is {state.task_type}. "
            f"Existing features are: {existing}. "
            "Propose up to 3 new features that could improve model performance. "
            "Be creative: consider feature interactions (e.g., Age * Pclass), polynomial features, group-based statistics (e.g., mean survival rate by Title), rare category handling, and feature combinations. "
            "Return your answer as a JSON list where each item has keys 'name', 'formula', and 'rationale'."
        )
        combined_prompt = (
            example_prompt + '\n' + example_response + '\n' + prompt
        )
        llm_raw = _query_llm(combined_prompt)
        try:
            proposals = json.loads(llm_raw)
        except Exception as exc:
            state.append_log(f"FeatureIdeation: LLM response JSON parse failed: {exc}. Raw response: {llm_raw}")
            # Retry with a simpler prompt
            simple_prompt = (
                f"Given a pandas DataFrame with columns {existing}, propose up to 3 new features as a JSON list with keys 'name', 'formula', and 'rationale'."
            )
            llm_raw_simple = _query_llm(simple_prompt)
            try:
                proposals = json.loads(llm_raw_simple)
            except Exception as exc2:
                state.append_log(f"FeatureIdeation: LLM simple prompt JSON parse failed: {exc2}. Raw response: {llm_raw_simple}")
                raise RuntimeError(f"LLM did not return any feature proposals. Last response: {llm_raw_simple}")
        if isinstance(proposals, dict):
            # If it's a dict with feature keys, wrap in a list
            if all(k in proposals for k in ("name", "formula", "rationale")):
                proposals = [proposals]
            else:
                proposals = proposals.get("features", [])
        if not isinstance(proposals, list) or not proposals:
            raise RuntimeError(f"LLM did not return any feature proposals. Last response: {llm_raw_simple if 'llm_raw_simple' in locals() else llm_raw}")

        if not isinstance(proposals, list) or not proposals:
            raise RuntimeError("LLM did not return any feature proposals")

        for prop in proposals:
            name = prop.get("name")
            formula = prop.get("formula")
            rationale = prop.get("rationale")
            if not name or not formula:
                raise RuntimeError("LLM proposal missing 'name' or 'formula'")
            if name in state.known_features:
                state.append_log(f"FeatureIdeation: skip duplicate feature '{name}'")
                continue
            state.features.append(name)
            state.known_features.add(name)
            state.feature_formulas[name] = formula
            state.feature_ideas.append({"name": name, "formula": formula, "rationale": rationale})
            state.append_log(
                f"FeatureIdeation: propose {name} = {formula} because {rationale}"
            )

        return state


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    return Agent().run(state)
