import numpy as np
import pandas as pd


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
