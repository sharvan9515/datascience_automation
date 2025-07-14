# agents.md

## Overview: Agentic Data Science Pipeline

This document outlines the modular agents used in an LLM-driven, LangGraph-based data science automation pipeline. Each agent corresponds to a key step in the ML lifecycle (e.g., preprocessing, feature engineering, modeling), and is orchestrated dynamically via LLM-based planning logic. The system iteratively refines data transformations and model components using real-time feedback from modeling metrics and LLM reasoning.

## 🎛 Agent Architecture

All agents operate on shared pipeline state and communicate decisions via state updates and structured logs. Agents are categorized into functional groups:

---

## 🔍 1. Task Identification Agent

**Role:** Infers the ML task (classification, regression, etc.) and initial data quality insights.

**LLM Prompt Example:**

```text
We have a dataset with columns: Age (numeric), BloodPressure (numeric), Outcome (values: Positive/Negative). Determine the ML task type and highlight immediate data issues.
```

**Output:**

* `task_type`: classification | regression | clustering
* `issues_detected`: missing values, skewed distribution, etc.
* Updates: pipeline state and logs

---

## 🧹 2. Preprocessing Agent

**Role:** Decides how to handle missing values, encode categorical variables, scale/normalize numeric columns, etc.

**LLM Prompt Strategy:**
Feed column stats and issue descriptions; ask LLM for preprocessing plan + code snippet using `pandas`/`sklearn`.

**Tool Integration:** Executes generated code, validates, and logs all actions.

---

## 📊 3. Correlation & EDA Agent (Optional)

**Role:** Computes and summarizes correlations and outliers to inform feature selection or dimensionality reduction.

**Tool:** Python correlation matrix, univariate plots, mutual info scores.

**LLM Use:** Interprets results, suggests redundant features or highly predictive columns.

---

## 🧠 4. Feature Ideation Agent

**Role:** Suggests new features based on domain knowledge and dataset semantics (e.g., ratios, deltas, logarithmic transforms).

**Prompt Template:**

```text
Given the dataset and task: classification (predict loan approval). Propose 3 new features that might improve accuracy. Justify each feature.
```

**Output:** Structured list of `feature_name`, `formula`, `rationale`.

---

## 🛠 5. Feature Implementation Agent

**Role:** Converts proposed features into executable Python code.

**Prompt Template:**

```text
Implement the following features in a pandas DataFrame `df`... [list]
```

**Execution:**

* Code is executed safely with retries.
* If errors occur, LLM is re-prompted with error trace.
* All transformations are logged with reasoning.

---

## 📉 6. Feature Selection & Evaluation Agent

**Role:** Measures utility of new features via model testing. Drops features that degrade or don’t improve performance.

**Workflow:**

1. Train baseline model.
2. Add feature(s) incrementally.
3. Evaluate impact (accuracy, F1, etc).
4. Use LLM to decide which features to retain/drop based on performance gains.

**Logs:** Record accuracy deltas and LLM explanations.

---

## 🔻 7. Feature Reduction Agent

**Role:** Optionally reduces dimensionality (PCA, embeddings) if feature count is high or correlated.

**Decision Logic:** Based on feature count/sample size and multicollinearity.

**Prompt Template:**

```text
We have 40 features, 500 samples, many highly correlated. Should we apply PCA? If so, how many components?
```

**Execution:** Sklearn PCA or similar, preserving explained variance >90%.

---

## 🧪 8. Model Training Agent

**Role:** Selects and trains an ML model (Random Forest, XGBoost, etc.) on the current features.

**Prompt Template:**

```text
Recommend a model for classification on 1000 samples, 8 features. Provide code to train and evaluate.
```

**Code Output:** Trains using sklearn, prints accuracy, confusion matrix, classification report.

---

## 📈 9. Model Evaluation & Refinement Agent

**Role:** Analyzes model outputs and suggests improvements (feature edits, model switch, tuning).

**LLM Use:** Diagnoses poor precision/recall, suggests new features or alternate models.

**State Update:** Set `state.iterate = True` if improvements proposed.

---

## 🤖 10. Orchestration & Planner Agent

**Role:** Dynamically decides which agents to activate based on dataset and intermediate results.

**LLM Prompt Strategy:** Feed metadata, ask for binary decisions on each step:

```text
Should we run: preprocessing? feature generation? selection? reduction?
```

**Control Flow:** LangGraph conditional branching. Iterations loop if `state.iterate=True`.

---

## 🔍 Logging Protocol

* Every agent appends to `state.log`
* Log entries are human-readable and include reasoning:

```text
"FeatureSelection: Kept 'Debt_to_Income_Ratio' (+3% accuracy), dropped 'Risk_Factor' (no gain)."
```

* Final report compiled from logs

---

## 📌 Sample Prompt to Bootstrap Codex System

Use this prompt to start building the full pipeline:

```text
You are building an agentic data science system using LangGraph. Create an end-to-end pipeline where:
- An LLM-based Orchestrator decides which agents to activate.
- Agents cover preprocessing, feature ideation, implementation, selection, dimensionality reduction, model training, and evaluation.
- Features are only retained if they improve model performance.
- The system logs every action and rationale.
- Use Python + LangGraph-style functions. Support retries on code execution errors.
Generate the full pipeline starting with a basic CSV input and iteratively improve it.
```

---

## 📎 Future Extensions

* Human approval nodes
* Advanced AutoML integration
* Experiment tracking & rollback
* Model card auto-documentation agent

---

## ✅ Summary

Each agent contributes a specific skill to the pipeline. LLMs provide intelligence, tool nodes enforce correctness, and LangGraph manages dynamic execution. This architecture supports full-cycle, explainable, and adaptive data science pipelines driven by collaborative agents.
