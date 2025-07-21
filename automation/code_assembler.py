from __future__ import annotations

import json
import os
import re
import ast
from automation.pipeline_state import PipelineState

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
    'LogisticRegression': 'from sklearn.linear_model import LogisticRegression',
    'LinearRegression': 'from sklearn.linear_model import LinearRegression',
    'RandomForestClassifier': 'from sklearn.ensemble import RandomForestClassifier',
    'RandomForestRegressor': 'from sklearn.ensemble import RandomForestRegressor',
    'SVC': 'from sklearn.svm import SVC',
    'SVR': 'from sklearn.svm import SVR',
    'pd': 'import pandas as pd',
    'os': 'import os',
    're': 'import re',
    'np': 'import numpy as np',
}
ALWAYS_IMPORTS = ['import os', 'import pandas as pd']

def run(state: PipelineState) -> PipelineState:
    """Assemble the final pipeline code and persist pipeline artifacts."""
    state.append_log("Code assembler supervisor: assembling pipeline")
    os.makedirs("output", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # 1. Collect all code snippets by stage
    code_blocks = {stage: state.code_blocks.get(stage, []) for stage in ORDER}
    included_snippets = []
    engineered_features = set()
    for stage in ["feature_implementation", "feature_selection"]:
        for snippet in code_blocks.get(stage, []):
            # Flatten dict-formatted code
            snippet = snippet.strip()
            if (snippet.startswith('{') and snippet.endswith('}')) or (snippet.startswith('import re') and '{' in snippet and '}' in snippet):
                try:
                    dict_str = snippet
                    if 'import re' in snippet:
                        dict_str = snippet.split('import re')[-1].strip()
                    code_dict = ast.literal_eval(dict_str)
                    for v in code_dict.values():
                        for l in str(v).splitlines():
                            if l.strip():
                                included_snippets.append(l)
                                m = re.match(r"df\['([^']+)'\]", l)
                                if m:
                                    engineered_features.add(m.group(1))
                except Exception as e:
                    included_snippets.append(f"# [Assembler Warning] Failed to parse dict snippet: {e}")
                    included_snippets.append(snippet)
            else:
                for l in snippet.splitlines():
                    if l.strip():
                        included_snippets.append(l)
                        m = re.match(r"df\['([^']+)'\]", l)
                        if m:
                            engineered_features.add(m.group(1))

    # 2. Collect all modeling code (after feature engineering)
    modeling_code = []
    for stage in ["feature_reduction", "model_training", "hyperparameter_search"]:
        for snippet in code_blocks.get(stage, []):
            for l in snippet.splitlines():
                if l.strip():
                    modeling_code.append(l)

    # 3. Collect preprocessing code (before feature engineering)
    preprocessing_code = []
    for snippet in code_blocks.get("preprocessing", []):
        for l in snippet.splitlines():
            if l.strip():
                preprocessing_code.append(l)

    # 4. Build the full script
    # 4.1. Imports
    used_imports = set(ALWAYS_IMPORTS)
    all_code = '\n'.join(preprocessing_code + included_snippets + modeling_code)
    for keyword, import_stmt in IMPORT_MAP.items():
        if re.search(rf'\b{keyword}\b', all_code):
            used_imports.add(import_stmt)
    import_lines = sorted(used_imports)

    # Deduplicate lines while preserving order
    def dedup_lines(lines):
        seen = set()
        result = []
        for line in lines:
            if line not in seen:
                seen.add(line)
                result.append(line)
        return result

    lines = []
    lines.extend(import_lines)
    lines.append("")
    lines.append("# Set your dataset path and target column here")
    lines.append("csv_path = 'your_data.csv'  # TODO: set your CSV file path")
    lines.append("target = 'your_target_column'  # TODO: set your target column name")
    lines.append("df = pd.read_csv(csv_path)")
    lines.append("")
    if engineered_features:
        lines.append(f"# Engineered features included in this pipeline: {sorted(engineered_features)}")
    lines.append("")
    # 4.2. Preprocessing
    if preprocessing_code:
        lines.append("# --- Preprocessing ---")
        lines.extend(preprocessing_code)
        lines.append("")
    # 4.3. Feature Engineering
    if included_snippets:
        lines.append("# --- Feature Engineering ---")
        lines.extend(included_snippets)
        lines.append("")
    # 4.4. Modeling
    if modeling_code:
        lines.append("# --- Modeling ---")
        lines.extend(modeling_code)
        lines.append("")

    # Deduplicate all lines before writing
    deduped_lines = dedup_lines(lines)

    # 5. Write to finalcode.py
    print("[DEBUG] Final deduplicated lines to be written to finalcode.py:")
    for line in deduped_lines:
        print(line)
    with open("finalcode.py", "w") as f:
        f.write("\n".join(deduped_lines))

    # 6. Write logs and reports
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
