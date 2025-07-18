import sys
import pandas as pd
from automation.pipeline_state import PipelineState
from automation.agents.orchestrator import run

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_agentic_pipeline.py <csv_path> <target_column>")
        sys.exit(1)
    csv_path = sys.argv[1]
    target = sys.argv[2]
    df = pd.read_csv(csv_path)
    state = PipelineState(df=df, target=target)
    state = run(state)
    print("Agentic pipeline completed. Check logs and output for results.") 