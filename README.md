# Data Science Automation

This project implements a simplified version of the agent-based pipeline described in
`AGENTS.md`.  Each agent is a small Python module that operates on a shared
`PipelineState` object and the orchestrator chains them together.  The system can
consult an OpenAI model when an API key is supplied, but it also works purely with
the built‑in heuristics.

## Quickstart

1. **Install the requirements**

   ```bash
   pip install -r requirements.txt
   ```

2. **Create or supply a CSV dataset.**  Any tabular data with a target column will
   work.  The snippet below exports the classic Iris dataset:

   ```bash
   python - <<'EOF'
   import pandas as pd
   from sklearn.datasets import load_iris

   iris = load_iris(as_frame=True)
   iris.frame.to_csv('iris.csv', index=False)
   EOF
   ```

3. **Run the pipeline** on the CSV file, specifying the name of the target column:

   ```bash
   python -m automation.pipeline iris.csv target
   ```

   If you set the environment variable `OPENAI_API_KEY`, the agents will query the
   API for more detailed suggestions.  Without it, they fall back to deterministic
   behaviour.

The script prints log entries from every agent step.  Example output:

```
TaskIdentification: LLM unclear, used heuristic task_type=classification
...
ModelEvaluation decision: iterate=False - heuristic: accuracy sufficient
```

See [AGENTS.md](AGENTS.md) for a description of each agent in the pipeline.
