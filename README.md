# Data Science Automation

This project implements a simplified version of the agent-based pipeline
described in `AGENTS.md`. Each step of the data science workflow is handled by a
dedicated agent which operates on a shared `PipelineState` object. The
orchestrator glues these agents together and optionally consults an OpenAI model
for richer suggestions. If no API key is provided, the agents fall back to the
built‑in heuristics.

## Quickstart

1. **Install the requirements**

   Ensure you are using Python 3.12 or later. The project relies on
   `openai>=1.0.0` and other modern versions of the dependencies listed in
   `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

2. **Create or supply a CSV dataset.**  Any tabular data with a target column will
   work.  The snippet below exports the classic Iris dataset as an example. Make
   sure the dataset includes the column you plan to predict:

   ```bash
   python - <<'EOF'
   import pandas as pd
   from sklearn.datasets import load_iris

   iris = load_iris(as_frame=True)
   iris.frame.to_csv('iris.csv', index=False)
   EOF
   ```

   For your own data, replace `iris.csv` with the path to your dataset.

3. **Run the pipeline** on the CSV file, passing the path and the target column
   name. Optional flags `--max-iter` and `--patience` control the number of
   iterations and how many non-improving rounds to tolerate:

   ```bash
   python -m automation.pipeline iris.csv target --max-iter 15 --patience 3
   ```

   The script produces a detailed log of each agent step. The pipeline checks
   that the environment variable `OPENAI_API_KEY` is set before starting.
   When provided the orchestrator consults the OpenAI API for smarter decisions;
   otherwise it raises an error. After completion you will find:

   - `finalcode.py` – assembled code for the final pipeline
   - `output/` – directory containing logs, iteration history and a short report

   The generated `finalcode.py` can be run standalone on the CSV using:

   ```bash
   python finalcode.py iris.csv target
   ```

The script prints log entries from every agent step.  Example output:

```
TaskIdentification: LLM resonpse
ModelEvaluation decision: iterate=False -LLM decision 
```

See [AGENTS.md](AGENTS.md) for a description of each agent in the pipeline.
