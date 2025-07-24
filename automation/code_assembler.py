from __future__ import annotations

import json
import os
import re
import ast
from automation.pipeline_state import PipelineState


class CodeAssembler:
    """Assemble validated code snippets into a runnable script."""

    ORDER = [
        "preprocessing",
        "feature_implementation",
        "feature_selection",
        "feature_reduction",
        "model_training",
        "hyperparameter_search",
    ]

    IMPORT_MAP = {
        'train_test_split': 'from sklearn.model_selection import train_test_split',
        'GridSearchCV': 'from sklearn.model_selection import GridSearchCV',
        'PCA': 'from sklearn.decomposition import PCA',
        'RandomForestClassifier': 'from sklearn.ensemble import RandomForestClassifier',
        'RandomForestRegressor': 'from sklearn.ensemble import RandomForestRegressor',
        'XGBClassifier': 'from xgboost import XGBClassifier',
        'XGBRegressor': 'from xgboost import XGBRegressor',
        'pd': 'import pandas as pd',
        'os': 'import os',
        're': 'import re',
        'np': 'import numpy as np',
    }
    ALWAYS_IMPORTS = ['import os', 'import pandas as pd']

    @staticmethod
    def _dedup_imports(lines: list[str]) -> list[str]:
        """Return import lines with duplicates removed more thoroughly."""
        norm_map: dict[str, set[str]] = {}
        simple_imports: set[str] = set()
        ordered: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("from ") and " import " in stripped:
                match = re.match(r"from\s+(\S+)\s+import\s+(.+)", stripped)
                if match:
                    module = match.group(1)
                    names = {n.strip() for n in match.group(2).split(",")}
                    if module not in norm_map:
                        norm_map[module] = set()
                        ordered.append(module)
                    norm_map[module].update(names)
                else:
                    if stripped not in simple_imports:
                        simple_imports.add(stripped)
                        ordered.append(stripped)
            elif stripped.startswith("import "):
                canonical = re.sub(r"\s+", " ", stripped)
                if canonical not in simple_imports:
                    simple_imports.add(canonical)
                    ordered.append(canonical)
            else:
                if stripped not in simple_imports:
                    simple_imports.add(stripped)
                    ordered.append(stripped)

        result: list[str] = []
        for item in ordered:
            if item in norm_map:
                names = ", ".join(sorted(norm_map[item]))
                result.append(f"from {item} import {names}")
            else:
                result.append(item)
        return result

    def run(self, state: PipelineState) -> PipelineState:
        """Assemble the final pipeline code and persist pipeline artifacts."""
        state.append_log("Code assembler supervisor: assembling pipeline")
        os.makedirs("output", exist_ok=True)
        os.makedirs("artifacts", exist_ok=True)

        # 1. Collect all code snippets by stage
        code_blocks = {stage: state.code_blocks.get(stage, []) for stage in self.ORDER}
        included_snippets = []
        engineered_features = set()
        for stage in ["feature_implementation", "feature_selection"]:
            for snippet in code_blocks.get(stage, []):
                snippet = snippet.strip()
                if (snippet.startswith('{') and snippet.endswith('}')) or (
                    snippet.startswith('import re') and '{' in snippet and '}' in snippet
                ):
                    try:
                        dict_str = snippet.split('import re')[-1].strip() if 'import re' in snippet else snippet
                        code_dict = ast.literal_eval(dict_str)
                        for val in code_dict.values():
                            for line in str(val).splitlines():
                                line = line.strip()
                                if line:
                                    included_snippets.append(line)
                                    m = re.match(r"df\['([^']+)'\]", line)
                                    if m:
                                        engineered_features.add(m.group(1))
                    except Exception as e:
                        included_snippets.append(f"# [Assembler Warning] Failed to parse dict snippet: {e}")
                        included_snippets.append(snippet)
                else:
                    for line in snippet.splitlines():
                        line = line.strip()
                        if line:
                            included_snippets.append(line)
                            m = re.match(r"df\['([^']+)'\]", line)
                            if m:
                                engineered_features.add(m.group(1))

        # 2. Collect all modeling code
        modeling_code = []
        for stage in ["feature_reduction", "model_training", "hyperparameter_search"]:
            for snippet in code_blocks.get(stage, []):
                for line in snippet.splitlines():
                    line = line.strip()
                    if line:
                        modeling_code.append(line)

        # 3. Collect preprocessing code
        preprocessing_code = []
        for snippet in code_blocks.get("preprocessing", []):
            for line in snippet.splitlines():
                line = line.strip()
                if line:
                    preprocessing_code.append(line)

        preprocessing_logs = [
            msg.split("Preprocessing:", 1)[1].strip()
            for msg in state.log
            if msg.startswith("Preprocessing:")
        ]
        feature_logs = [
            msg.split("FeatureImplementation:", 1)[1].strip()
            for msg in state.log
            if msg.startswith("FeatureImplementation:")
        ]

        # 4. Build the full script
        used_imports = set(self.ALWAYS_IMPORTS)
        all_code = '\n'.join(preprocessing_code + included_snippets + modeling_code)
        for keyword, import_stmt in self.IMPORT_MAP.items():
            if re.search(rf'\b{keyword}\b', all_code):
                used_imports.add(import_stmt)
        if "cross_val_score" not in all_code:
            used_imports.add('from sklearn.model_selection import cross_val_score')
        import_lines = self._dedup_imports(sorted(used_imports))

        need_helper = any(
            "ensure_numeric_features(" in line
            for line in preprocessing_code + included_snippets + modeling_code
        )

        def dedup_lines(lines: list[str]) -> list[str]:
            seen = set()
            result = []
            for line in lines:
                canonical = line.strip()
                if canonical.startswith("import ") or canonical.startswith("from "):
                    canonical = re.sub(r"\s+", " ", canonical)
                    if canonical in seen:
                        continue
                    seen.add(canonical)
                result.append(line)
            return result

        lines: list[str] = []
        lines.extend(import_lines)
        lines.append("")
        if need_helper:
            lines.extend([
                "def ensure_numeric_features(df, target):",
                "    for col in df.columns:",
                "        if col == target:",
                "            continue",
                "        if not pd.api.types.is_numeric_dtype(df[col]):",
                "            df[col] = df[col].astype('category').cat.codes",
                "        if df[col].isnull().any():",
                "            fill_value = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]",
                "            df[col] = df[col].fillna(fill_value)",
                "    return df",
                "",
            ])
        lines.append("# Set your dataset path and target column here")
        lines.append("csv_path = 'your_data.csv'  # TODO: set your CSV file path")
        lines.append("target = 'your_target_column'  # TODO: set your target column name")
        lines.append("df = pd.read_csv(csv_path)")
        lines.append("")
        if engineered_features:
            lines.append(f"# Engineered features included in this pipeline: {sorted(engineered_features)}")
        for log in feature_logs:
            lines.append(f"# Feature step: {log}")
        lines.append("")

        if preprocessing_code:
            lines.append("# --- Preprocessing ---")
            for log in preprocessing_logs:
                lines.append(f"# Step: {log}")
            lines.append("try:")
            lines.extend([f"    {l}" for l in preprocessing_code])
            lines.append("except Exception as e:")
            lines.append("    print(f'Preprocessing failed: {e}')")
            lines.append("")

        if included_snippets:
            lines.append("# --- Feature Engineering ---")
            lines.append("try:")
            lines.extend([f"    {l}" for l in included_snippets])
            lines.append("except Exception as e:")
            lines.append("    print(f'Feature engineering failed: {e}')")
            lines.append("")

        if modeling_code:
            lines.append("# --- Modeling ---")
            lines.extend(modeling_code)
            lines.append("")
        if "cross_val_score" not in all_code:
            lines.append("cv_score = cross_val_score(model, X, y, cv=3).mean()")
            lines.append("print('Cross-validation score:', cv_score)")
            lines.append("")

        deduped_lines = dedup_lines(lines)
        print("[DEBUG] Final deduplicated lines to be written to finalcode.py:")
        for line in deduped_lines:
            print(line)
        with open("finalcode.py", "w") as f:
            f.write("\n".join(deduped_lines))

        with open("output/logs.json", "w") as f:
            json.dump(state.log, f, indent=2)
        with open("output/history.json", "w") as f:
            json.dump(state.iteration_history, f, indent=2)
        report_lines = [
            f"Iterations run: {state.iteration}",
            f"Best score: {state.best_score}",
            f"Features: {state.features}",
        ]
        with open("output/report.txt", "w") as f:
            f.write("\n".join(report_lines))

        state.append_log("Final pipeline code assembled successfully")
        return state


def run(state: PipelineState) -> PipelineState:
    """Backwards compatible function API."""
    return CodeAssembler().run(state)
