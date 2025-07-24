from __future__ import annotations

"""Grid search hyperparameter tuning agent."""

import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from automation.pipeline_state import PipelineState
from .base import BaseAgent

__all__ = ["run"]


class HyperparameterSearchAgent(BaseAgent):
    """Perform a simple grid search when performance stalls."""

    def run(self, state: PipelineState) -> PipelineState:
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
        if state.best_score is None or search.best_score_ > state.best_score:
            state.best_score = search.best_score_
            state.current_score = search.best_score_

        joblib.dump(search.best_estimator_, "artifacts/model.pkl")
        # Track the best estimator for ensembling
        y_pred = search.best_estimator_.predict(X_test)
        state.add_trained_model(
            model=search.best_estimator_,
            name=search.best_estimator_.__class__.__name__,
            model_type=state.task_type,
            predictions=y_pred,
            score=search.best_score_
        )
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


# Backwards compatible function API
def run(state: PipelineState) -> PipelineState:
    """Backwards compatible function API."""
    return HyperparameterSearchAgent().run(state)
