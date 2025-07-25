from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from automation.utils.sandbox import safe_exec
from automation.validators import DataValidator

__all__ = [
    "execute_preprocessing_code",
    "execute_feature_engineering_code",
    "test_model_performance",
]


def execute_preprocessing_code(df: pd.DataFrame, code: str, target: str) -> pd.DataFrame:
    """Execute preprocessing code safely and validate the result."""
    try:
        local_vars = {"df": df.copy(), "target": target}
        local_vars = safe_exec(
            code,
            extra_globals={"pd": pd},
            local_vars=local_vars,
            allowed_modules={"pandas"},
        )
        new_df = local_vars.get("df", df.copy())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Execution failed: {exc}") from exc

    ok, reason = DataValidator.validate_transformation(df, new_df, target)
    if not ok:
        raise RuntimeError(f"Validation failed: {reason}")

    for col in new_df.columns:
        if col == target:
            continue
        if not pd.api.types.is_numeric_dtype(new_df[col]):
            raise RuntimeError(f"Column '{col}' is not numeric after preprocessing")
        if new_df[col].isnull().any():
            raise RuntimeError(f"Missing values remain in column '{col}'")

    return new_df


def execute_feature_engineering_code(df: pd.DataFrame, code: str, target: str) -> pd.DataFrame:
    """Execute feature engineering code and verify new columns."""
    try:
        local_vars = {"df": df.copy(), "target": target}
        local_vars = safe_exec(
            code,
            extra_globals={"pd": pd},
            local_vars=local_vars,
            allowed_modules={"pandas", "numpy", "sklearn", "xgboost"},
        )
        new_df = local_vars.get("df", df.copy())
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Execution failed: {exc}") from exc

    ok, reason = DataValidator.validate_transformation(df, new_df, target)
    if not ok:
        raise RuntimeError(f"Validation failed: {reason}")

    new_cols = [c for c in new_df.columns if c not in df.columns]
    for col in new_cols:
        if not pd.api.types.is_numeric_dtype(new_df[col]):
            raise RuntimeError(f"Engineered column '{col}' is not numeric")
        if new_df[col].isnull().any():
            raise RuntimeError(f"Engineered column '{col}' contains missing values")

    return new_df


def test_model_performance(df: pd.DataFrame, target: str, task_type: str) -> float:
    """Train a small RandomForest and return its validation score."""
    X = df.drop(columns=[target]).copy()
    y = df[target]
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes
    X = X.fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)
