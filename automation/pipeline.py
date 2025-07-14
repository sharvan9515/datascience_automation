import pandas as pd
from automation.pipeline_state import PipelineState
from automation.agents import orchestrator


def run_pipeline(csv_path: str, target: str):
    df = pd.read_csv(csv_path)
    state = PipelineState(df=df, target=target)
    state = orchestrator.run(state)
    return state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run data science automation pipeline")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("target", help="Target column name")
    args = parser.parse_args()
    final_state = run_pipeline(args.csv, args.target)
    for entry in final_state.log:
        print(entry)
