import numpy as np
import pandas as pd
import re

IMPORT_FIXES = {
    'train_test_split': 'from sklearn.model_selection import train_test_split',
    'GridSearchCV': 'from sklearn.model_selection import GridSearchCV',
}


def inject_missing_imports(code: str) -> str:
    """Prepend common sklearn imports if used but missing."""
    imports = []
    for keyword, stmt in IMPORT_FIXES.items():
        if keyword in code and stmt not in code:
            imports.append(stmt)
    if imports:
        code = "\n".join(imports) + "\n" + code
    return code


class DataValidator:
    """Utility class for validating dataframe transformations."""

    @staticmethod
    def validate_transformation(
        df_before: pd.DataFrame, df_after: pd.DataFrame, target: str
    ) -> tuple[bool, str]:
        """Validate that a transformation produced sensible output.

        Checks for new NaN/inf values, column mismatches and large shifts
        in target statistics. Returns ``(True, message)`` if validation
        passes, otherwise ``(False, reason)``.
        """
        # Target column must exist after transformation
        if target not in df_after.columns:
            return False, f"Target column '{target}' missing after transformation"

        # Check for new NaN values
        before_nans = df_before.isna().sum().sum()
        after_nans = df_after.isna().sum().sum()
        if after_nans > before_nans:
            return False, "NaN values introduced"

        # Check for inf values
        if np.isinf(df_after.select_dtypes(include=[np.number])).any().any():
            return False, "Infinite values present"

        # Check for column mismatch
        cols_before = set(df_before.columns)
        cols_after = set(df_after.columns)
        if cols_before != cols_after:
            missing = cols_before - cols_after
            extra = cols_after - cols_before
            return False, f"Column mismatch. Missing: {missing}, Extra: {extra}"

        # Check for large shift in target mean
        before_mean = df_before[target].mean()
        after_mean = df_after[target].mean()
        if np.isfinite(before_mean) and np.isfinite(after_mean):
            denom = abs(before_mean) if before_mean != 0 else 1.0
            if abs(after_mean - before_mean) / denom > 0.1:
                return False, "Large shift in target mean (>10%)"
        return True, "Validation passed"
class CodeQualityValidator:
    """Validate code snippets for safety and effectiveness."""

    @staticmethod
    def _validate(
        code: str,
        df: pd.DataFrame,
        target: str,
        task_type: str,
        allowed_modules: set[str],
    ) -> tuple[bool, str, pd.DataFrame | None, float | None]:
        import ast
        from automation.utils.sandbox import safe_exec
        from automation.agents import model_evaluation

        code = inject_missing_imports(code)
        try:
            ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}", None, None

        trial_df = df.copy()
        try:
            local_vars = {"df": trial_df, "target": target}
            safe_exec(
                code,
                extra_globals={"pd": pd},
                local_vars=local_vars,
                allowed_modules=allowed_modules,
            )
            trial_df = local_vars.get("df", trial_df)
        except Exception as e:  # noqa: BLE001
            return False, f"Execution failed: {e}", None, None

        ok, reason = DataValidator.validate_transformation(df, trial_df, target)
        if not ok:
            return False, f"Data validation failed: {reason}", None, None

        try:
            baseline = model_evaluation.compute_score(df, target, task_type)
            trial_score = model_evaluation.compute_score(trial_df, target, task_type)
        except Exception as e:
            return False, f"Scoring failed: {e}", None, None

        if trial_score < baseline - 0.05:
            return False, f"Performance drop: {trial_score:.4f} < {baseline:.4f} - 0.05", None, None

        return True, "Validation passed", trial_df, trial_score

    @staticmethod
    def validate_preprocessing_code(
        code: str,
        df: pd.DataFrame,
        target: str,
        task_type: str,
    ) -> tuple[bool, str, pd.DataFrame | None, float | None]:
        allowed = {"pandas"}
        return CodeQualityValidator._validate(code, df, target, task_type, allowed)

    @staticmethod
    def validate_feature_code(
        code: str,
        df: pd.DataFrame,
        target: str,
        task_type: str,
    ) -> tuple[bool, str, pd.DataFrame | None, float | None]:
        allowed = {"pandas", "numpy", "sklearn", "xgboost"}
        return CodeQualityValidator._validate(code, df, target, task_type, allowed)

    @staticmethod
    def validate_generic_code(
        code: str,
        df: pd.DataFrame,
        target: str,
        task_type: str,
    ) -> tuple[bool, str, pd.DataFrame | None, float | None]:
        allowed = {"pandas", "numpy", "sklearn", "xgboost"}
        return CodeQualityValidator._validate(code, df, target, task_type, allowed)
