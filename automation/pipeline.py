"""Command-line entry point for the agentic data science pipeline."""

import argparse
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

from automation.dataset_profiler import EnhancedDatasetProfiler

from automation.pipeline_state import PipelineState
from automation.agents import orchestrator


def run_pipeline(csv_path: str, target: str, patience: int = 20, score_threshold: float = 0.80) -> PipelineState:
    """Load data, initialize :class:`PipelineState`, and run the orchestrator."""

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY environment variable is required")

    df = pd.read_csv(csv_path)
    state = PipelineState(df=df, target=target)
    state.profile = EnhancedDatasetProfiler.generate_comprehensive_profile(df, target)
    return orchestrator.run(state, patience=patience, score_threshold=score_threshold)


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
    parser.add_argument("--patience", type=int, default=20, help="Rounds without improvement before stopping")
    parser.add_argument("--score-threshold", type=float, default=0.80, help="Score threshold to trigger hyperparameter search")
    parsed = parser.parse_args(args)
    final_state = run_pipeline(
        parsed.csv,
        parsed.target,
        patience=parsed.patience,
        score_threshold=parsed.score_threshold,
    )
    print_final_log(final_state)
    final_state.write_log("output/pipeline.log")
    print("Logs written to output/pipeline.log")


if __name__ == "__main__":
    main()
