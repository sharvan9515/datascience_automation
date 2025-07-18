from __future__ import annotations

import json
import os
import re

from automation.pipeline_state import PipelineState

# Execution order for collected code blocks. Only stages that generate
# executable snippets are included here.
ORDER = [
    "preprocessing",
    "feature_implementation",
    "feature_selection",
    "feature_reduction",
    "model_training",
    "hyperparameter_search",
]

__all__ = ["run"]

# Map of keywords to their import statements
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
}

# Always needed for the pipeline
ALWAYS_IMPORTS = ['import os', 'import pandas as pd']

def run(state: PipelineState) -> PipelineState:
    """Assemble only the dynamically generated and validated code blocks and persist pipeline artifacts."""
    state.append_log("Code assembler supervisor: assembling pipeline")

    os.makedirs("output", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    # Collect all code lines from accepted code blocks
    code_lines = []
    for stage in ORDER:
        for snippet in state.code_blocks.get(stage, []):
            for line in snippet.splitlines():
                if 'joblib.dump' in line:
                    continue  # Remove model saving lines
                code_lines.append(line)

    # Find which imports are needed
    used_imports = set(ALWAYS_IMPORTS)
    code_str = '\n'.join(code_lines)
    for keyword, import_stmt in IMPORT_MAP.items():
        # Use regex to match whole words only
        if re.search(rf'\b{keyword}\b', code_str):
            used_imports.add(import_stmt)

    # Remove duplicates and sort
    import_lines = sorted(used_imports)

    # Main function definition (no argparse, notebook/colab style)
    lines: list[str] = []
    lines.extend(import_lines)
    lines.append("")
    lines.append("# Set your dataset path and target column here")
    lines.append("csv_path = 'your_data.csv'  # TODO: set your CSV file path")
    lines.append("target = 'your_target_column'  # TODO: set your target column name")
    lines.append("df = pd.read_csv(csv_path)")
    lines.append("")
    for line in code_lines:
        lines.append(line)

    with open("finalcode.py", "w") as f:
        f.write("\n".join(lines))

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
