# Agentic Data Science Automation

This project provides an agentic, LLM-driven data science pipeline that automates feature engineering, model selection, and evaluation. Now with a user-friendly Streamlit frontend!

## Features
- Upload your own CSV dataset
- Select the target column
- Set patience (number of rounds without improvement before stopping; default 20)
- Run the pipeline with a single click
- View the final model score (accuracy or F1)
- Download or copy the final assembled code for reproducibility

## Quickstart

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

Create a `.env` file in the project root containing your OpenAI key:

```bash
echo "OPENAI_API_KEY=your-key-here" > .env
```

### 2. Run the Streamlit Frontend
```bash
streamlit run streamlit_app.py
```

### 3. Use the Web UI
- Open [http://localhost:8501](http://localhost:8501) in your browser.
- **Upload** your CSV file.
- **Select** the target column from the dropdown.
- **Set** the patience value (number of rounds with no improvement before stopping, default 20).
- **Click** "Run Pipeline".
- **View** the final model score (accuracy or F1) and the full assembled code.

### 4. Output
- The app will show only the final model score and the generated Python code.
- You can copy the code and run it in Colab or any Python environment for reproducibility.

## Notes
- The backend pipeline is fully automated and agentic, using LLMs for feature ideation, implementation, and model selection.
- The frontend is designed for ease of use and hides all intermediate logs and errors, focusing on results.
- The `max-iter` logic is removed; only patience is used for early stopping.

## Enhanced Profiling, Adaptive Model Selection, and Validation

This repository now includes a more robust preprocessing and modeling pipeline:

- **Enhanced Profiling** – `EnhancedDatasetProfiler` generates a detailed profile
  of your dataset including missing-value patterns, outliers, correlations and
  domain insights. These metrics guide subsequent agents in the pipeline.
- **Adaptive Model Selection** – `IntelligentModelSelector` ranks potential
  algorithms based on the dataset profile and task type. The pipeline trains the
  recommended models first and can automatically switch if performance stalls.
- **Validation Pipeline** – All code produced by the agents is checked with
  `DataValidator` and `CodeQualityValidator` to ensure transformations are safe
  and do not degrade model performance.

### Minimal Example

You can run the pipeline programmatically without the Streamlit UI:

```python
from automation.pipeline import run_pipeline, compile_log

final_state = run_pipeline("data.csv", "target_column")
print(compile_log(final_state))
```

This loads your CSV, performs profiling and adaptive model selection, validates
each step, and prints the aggregated log when finished.

## Advanced Usage
- You can still run the pipeline from the command line:
  ```bash
  python run_agentic_pipeline.py <csv_path> <target_column> --patience 20
  ```
- All outputs and logs are saved in the `output/` directory.

## License
MIT
