# Data Science Automation

This project implements a simplified version of the agent-based pipeline described in `AGENTS.md`. Each agent is a small Python module that modifies a shared `PipelineState` object.

## Usage

Install dependencies and run the pipeline on a CSV file:

```bash
pip install -r requirements.txt
python -m automation.pipeline data.csv target_column
```

The script prints log entries for each agent step.
