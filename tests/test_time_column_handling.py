import pandas as pd
from automation.agents import model_evaluation


def test_compute_score_with_date_column():
    df = pd.DataFrame({
        "Date": ["1981-01-01", "1981-01-02", "1981-01-03", "1981-01-04"],
        "Temp": [1.0, 2.0, 3.0, 4.0],
    })
    score = model_evaluation.compute_score(df, target="Temp", task_type="regression", time_col="Date")
    assert isinstance(score, float)
