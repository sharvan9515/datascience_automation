import sys
import pandas as pd
from automation.pipeline_state import PipelineState
from automation.agents.orchestrator import run
from automation import code_assembler

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_agentic_pipeline.py <csv_path> <target_column>")
        sys.exit(1)
    csv_path = sys.argv[1]
    target = sys.argv[2]
    df = pd.read_csv(csv_path)
    state = PipelineState(df=df, target=target)
    final_state = state
    try:
        final_state = run(state, patience=20)
    finally:
        # Always assemble code, even on error
        code_assembler.run(final_state)
        final_state.write_log("output/pipeline.log")
    print("Agentic pipeline completed. Check logs and output for results.")
    # Print iteration history for accuracy/score progression
    print("\nIteration History (score progression):")
    for entry in final_state.iteration_history:
        print(
            f"Iteration {entry['iteration']}: score={entry.get('best_score', 'N/A')}, metrics={entry.get('metrics', '')}"
        )
    print("Logs written to output/pipeline.log")