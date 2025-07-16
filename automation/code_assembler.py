from __future__ import annotations

import json
import os

from automation.pipeline_state import PipelineState

# Execution order for collected code blocks. Only stages that generate
# executable snippets are included here.
ORDER = [
    "preprocessing",
    "feature_implementation",
    "feature_selection",
    "feature_reduction",
    "model_training",
]

__all__ = ["run"]


def run(state: PipelineState) -> PipelineState:
    """Assemble the generated code blocks and persist pipeline artifacts."""
    os.makedirs("output", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    lines: list[str] = [
        "import pandas as pd",
        "from sklearn.model_selection import train_test_split",
        "from sklearn.decomposition import PCA",
        "from sklearn.linear_model import LogisticRegression, LinearRegression",
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
        "from sklearn.svm import SVC, SVR",
        "import joblib",
    ]

    for stage in ORDER:
        for snippet in state.code_blocks.get(stage, []):
            lines.append(snippet)

    with open("pipeline.py", "w") as f:
        f.write("\n\n".join(lines))

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
