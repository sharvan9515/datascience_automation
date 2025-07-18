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

        # Dynamically build few-shot example from state or synthesize one
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
            # Synthesize a realistic example based on current columns
            synth_features = [c for c in state.df.columns if c != state.target][:2]
            synth_example = [
                {
                    "name": f"{synth_features[0]}_squared" if len(synth_features) > 0 else "Feature1",
                    "formula": f"{synth_features[0]} ** 2" if len(synth_features) > 0 else "...",
                    "rationale": f"Square of {synth_features[0]} may capture nonlinearity" if len(synth_features) > 0 else "..."
                },
                {
                    "name": f"{synth_features[0]}_x_{synth_features[1]}" if len(synth_features) > 1 else "Feature2",
                    "formula": f"{synth_features[0]} * {synth_features[1]}" if len(synth_features) > 1 else "...",
                    "rationale": f"Interaction between {synth_features[0]} and {synth_features[1]}" if len(synth_features) > 1 else "..."
                }
            ]
            example_existing = synth_features
            example_prompt = (
                "You are a creative feature engineering assistant. "
                f"Existing features: {example_existing}. "
                "Propose up to 3 new features that could improve model performance. "
                "Return JSON list where each item has keys 'name', 'formula', and 'rationale'."
            )
            example_response = json.dumps(synth_example, indent=2)
        # Main prompt for the current dataset
        prompt = (
            "You are a creative feature engineering assistant. "
            f"The current task type is {state.task_type}. "
            f"Existing features are: {existing}. "
            "Propose up to 3 new features that could improve model performance. "
            "Be creative: consider feature interactions (e.g., A * B), polynomial features (e.g., A ** 2), group-based statistics (e.g., mean(target) by C), rare category handling (e.g., is_rare_category), and combinations of categorical and numeric features. "
            "Avoid duplicating existing features. "
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
