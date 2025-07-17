from __future__ import annotations

"""Grid search hyperparameter tuning agent."""

import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from automation.pipeline_state import PipelineState

__all__ = ["run"]


def run(state: PipelineState) -> PipelineState:
    """Run grid search to tune a RandomForest model."""
    df = state.df
    X = df.drop(columns=[state.target]).copy()
    y = df[state.target]
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes
    X = X.fillna(0)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if state.task_type == "classification":
        model = RandomForestClassifier(random_state=42)
        param_grid = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
        scoring = "f1_weighted"
    else:
        model = RandomForestRegressor(random_state=42)
        param_grid = {"n_estimators": [100, 200], "max_depth": [None, 5, 10]}
        scoring = "r2"

    search = GridSearchCV(model, param_grid=param_grid, cv=3, scoring=scoring)
    search.fit(X_train, y_train)

    state.best_params = search.best_params_
    state.append_log(
        f"HyperparameterSearch: best_params={search.best_params_} score={search.best_score_:.4f}"
    )

    joblib.dump(search.best_estimator_, "artifacts/model.pkl")
    code_snippet = (
        f"param_grid = {param_grid}\n"
        f"search = GridSearchCV({model.__class__.__name__}(random_state=42), param_grid, cv=3, scoring='{scoring}')\n"
        "search.fit(X_train, y_train)\n"
        "best_model = search.best_estimator_\n"
        "joblib.dump(best_model, 'artifacts/model.pkl')\n"
        "best_params = search.best_params_"
    )
    state.append_code("hyperparameter_search", code_snippet)
    return state
