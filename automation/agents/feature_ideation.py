import json

from automation.pipeline_state import PipelineState
from ..prompt_utils import query_llm
from .base import BaseAgent


def _query_llm(prompt: str) -> str:
    """Wrapper around :func:`query_llm` with no examples."""

    return query_llm(prompt, expect_json=True)


class Agent(BaseAgent):
    """Feature ideation agent."""

    def _get_domain_knowledge(self, state: PipelineState) -> str:
        """Return a domain knowledge string for the current dataset, or empty string if unknown."""
        # Try to infer dataset name from file path if available
        dataset_name = getattr(state, 'dataset_name', None)
        if not dataset_name and hasattr(state.df, 'attrs'):
            dataset_name = state.df.attrs.get('dataset_name')
        # Fallback: try to infer from columns
        columns = set(state.df.columns)
        # Titanic example
        if {'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Name', 'Ticket'}.issubset(columns):
            dataset_name = 'titanic'
        if dataset_name and 'titanic' in dataset_name.lower():
            return (
                "Domain knowledge: This is the Titanic survival dataset. "
                "Useful features often include: extracting titles from the Name column (e.g., Mr, Mrs, Miss), family size (SibSp + Parch + 1), deck from Cabin, ticket groupings, fare per person, and binary flags for being alone. "
                "Text extraction from Name, rare category flags for Embarked, and interaction terms (e.g., Age*Pclass) are also effective. "
                "Missing Age can be imputed using group means. "
            )
        # No fallback for unknown datasets
        return "Domain knowledge: This is a tabular dataset. "
        prompt = "Domain knowledge: This is a tabular dataset. "
        if num_cols:
            prompt += f"Numeric columns: {num_cols}. You can propose features such as log transforms, polynomial features, ratios, differences, sums, means, z-scores, and interactions between numeric columns etc and many more "
        if cat_cols:
            prompt += f"Categorical columns: {cat_cols}. You can propose features such as one-hot encoding, frequency encoding, extracting substrings, rare category flags, or group-based statistics (e.g., mean target by category). and many more "
        if not num_cols and not cat_cols:
            prompt += "No numeric or categorical columns detected. You can propose features such as row counts, missing value indicators, or other creative transformations based on available data types. "
        prompt += "You may also propose features that combine different column types, such as group means, or binary flags for special values. Always be creative and consider feature interactions, polynomial features, and domain-agnostic best practices."
        return prompt

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
        # Add feature tracker summary to the prompt if available
        tracker_summary = ""
        summary = getattr(state, 'feature_tracking_summary', None)
        if summary:
            if summary.get('successful_patterns'):
                tracker_summary += f"\nSuccessful feature patterns so far (by type):\n{summary['successful_patterns']}"
            if summary.get('rejection_summary'):
                tracker_summary += f"\nRejected feature patterns so far (by type):\n{summary['rejection_summary']}"
        tracker_summary += ("\nIf a feature type (e.g., interaction, polynomial, ratio, text extraction, group stat, rare category) has repeatedly failed, avoid proposing it again. "
                            "If a feature type has been successful, try proposing more features of that type or combinations. "
                            "Be dataset-agnostic: do not use dataset-specific names, but use general feature engineering best practices.")
        # Add few-shot example to the prompt
        few_shot_example = (
            "Example feature proposals: [\n"
            "  {\"name\": \"Fare_per_Person\", \"formula\": \"Fare / (Family_Size + 1)\", \"rationale\": \"Normalizes fare by family size to capture per-person cost\"},\n"
            "  {\"name\": \"IsAlone\", \"formula\": \"1 if Family_Size == 0 else 0\", \"rationale\": \"Binary flag for passengers traveling alone\"},\n"
            "  {\"name\": \"Title\", \"formula\": \"Name.str.extract(' ([A-Za-z]+)\\.')\", \"rationale\": \"Extracts honorifics from names for social status\"}\n"
            "]\n"
        )
        # --- Inject domain knowledge dynamically ---
        domain_knowledge = self._get_domain_knowledge(state)
        # Main prompt for the current dataset
        prompt = (
            f"{domain_knowledge}\n"
            "You are a creative and domain-aware feature engineering assistant. "
            f"The current task type is {state.task_type}. "
            f"Existing features are: {existing}. "
            f"{tracker_summary} "
            f"{few_shot_example} "
            "Propose up to 3 new features that could improve model performance. "
            "Be creative: consider feature interactions (e.g., A * B), polynomial features (e.g., A ** 2), ratios (e.g., Fare / Age), group-based statistics (e.g., mean(target) by C), rare category handling (e.g., is_rare_category), and combinations of categorical and numeric features. "
            "For classification, consider extracting information from text columns, creating binary flags, or encoding high-cardinality categoricals. "
            "For regression, consider log transforms, scaling, or outlier handling. "
            "Return JSON list where each item has keys 'name', 'formula', and 'rationale'."
        )
        combined_prompt = (
            example_prompt + '\n' + example_response + '\n' + prompt
        )
        try:
            llm_raw = _query_llm(combined_prompt)
        except RuntimeError as exc:
            state.append_log(f"FeatureIdeation: LLM query failed: {exc}")
            return state
        try:
            proposals = json.loads(llm_raw)
        except Exception as exc:
            state.append_log(f"FeatureIdeation: LLM response JSON parse failed: {exc}. Raw response: {llm_raw}")
            # Retry with a simpler prompt
            simple_prompt = (
                f"Given a pandas DataFrame with columns {existing}, propose up to 3 new features as a JSON list with keys 'name', 'formula', and 'rationale'."
            )
            try:
                llm_raw_simple = _query_llm(simple_prompt)
            except RuntimeError as exc2:
                state.append_log(f"FeatureIdeation: retry query failed: {exc2}")
                raise RuntimeError(
                    "LLM did not return any feature proposals after retry"
                ) from exc2
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
