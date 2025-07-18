from __future__ import annotations

import json
import os
import argparse

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


def run(state: PipelineState) -> PipelineState:
    """Assemble the generated code blocks and persist pipeline artifacts."""
    state.append_log("Code assembler supervisor: assembling pipeline")

    os.makedirs("output", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    lines: list[str] = [
        "import argparse",
        "import os",
        "import pandas as pd",
        "from sklearn.model_selection import train_test_split, GridSearchCV",
        "from sklearn.decomposition import PCA",
        "from sklearn.linear_model import LogisticRegression, LinearRegression",
        "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor",
        "from sklearn.svm import SVC, SVR",
        "import joblib",
        "",
        "def main(args: list[str] | None = None) -> None:",
        "    parser = argparse.ArgumentParser(description='Run assembled pipeline')",
        "    parser.add_argument('csv', help='Path to CSV file')",
        "    parser.add_argument('target', help='Target column name')",
        "    parser.add_argument('--max-iter', type=int, default=10, help='Maximum iterations')",
        "    parser.add_argument('--patience', type=int, default=5, help='Rounds without improvement')",
        "    parsed = parser.parse_args(args)",
        "    df = pd.read_csv(parsed.csv)",
        "    target = parsed.target",
    ]

    for stage in ORDER:
        for snippet in state.code_blocks.get(stage, []):
            for line in snippet.splitlines():
                lines.append(f'    {line}')

    lines.extend([
        "",
        "if __name__ == '__main__':",
        "    main()",
    ])

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
