"""Command-line entry point for the agentic data science pipeline."""

import argparse
import pandas as pd
import os

from automation.pipeline_state import PipelineState
from automation.agents import orchestrator


def run_pipeline(csv_path: str, target: str, max_iter: int = 10, patience: int = 5) -> PipelineState:
    """Load data, initialize :class:`PipelineState`, and run the orchestrator."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    df = pd.read_csv(csv_path)
    state = PipelineState(df=df, target=target)
    return orchestrator.run(state, max_iter=max_iter, patience=patience)


def compile_log(state: PipelineState) -> str:
    """Return the pipeline log as a single formatted string."""
    return "\n".join(state.log)


def print_final_log(state: PipelineState) -> None:
    """Print the collected pipeline log entries."""
    print(compile_log(state))


def main(args: list[str] | None = None) -> None:
    """Parse CLI arguments, run the pipeline, and print the log."""

    parser = argparse.ArgumentParser(
        description="Run data science automation pipeline"
    )
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("target", help="Target column name")
    parser.add_argument("--max-iter", type=int, default=10, help="Maximum iterations")
    parser.add_argument("--patience", type=int, default=5, help="Rounds without improvement before stopping")
    parsed = parser.parse_args(args)
    final_state = run_pipeline(parsed.csv, parsed.target, max_iter=parsed.max_iter, patience=parsed.patience)
    print_final_log(final_state)


if __name__ == "__main__":
    main()
