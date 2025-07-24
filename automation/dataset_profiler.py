from __future__ import annotations

import pandas as pd
from typing import Dict, Any

__all__ = ["EnhancedDatasetProfiler"]


class EnhancedDatasetProfiler:
    """Utility class for generating detailed dataset profiles."""

    @classmethod
    def generate_comprehensive_profile(
        cls, df: pd.DataFrame, target: str
    ) -> Dict[str, Any]:
        """Return a rich profiling summary for the given dataframe."""
        profile: Dict[str, Any] = {}
        profile["statistical_summary"] = df.describe(include="all").to_dict()
        profile["missing_patterns"] = cls._analyze_missing_patterns(df)
        profile["outlier_detection"] = cls._detect_outliers(df)
        profile["skewness"] = (
            df.select_dtypes(include="number").skew(numeric_only=True).to_dict()
        )
        profile["correlations"] = cls._compute_correlations(df, target)
        profile["domain_insights"] = cls._infer_domain_insights(df)
        profile["complexity_metrics"] = cls._calculate_complexity_metrics(df, target)
        return profile

    @staticmethod
    def _analyze_missing_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Return missing value stats per column and row-wise patterns."""
        col_summary = {
            col: {
                "missing_count": int(df[col].isna().sum()),
                "missing_percent": float(df[col].isna().mean()),
            }
            for col in df.columns
        }
        row_pattern = df.isna().sum(axis=1).value_counts().sort_index().to_dict()
        return {"column_summary": col_summary, "row_patterns": row_pattern}

    @staticmethod
    def _detect_outliers(df: pd.DataFrame) -> Dict[str, int]:
        """Detect simple numeric outliers via z-score > 3."""
        outliers: Dict[str, int] = {}
        numeric_df = df.select_dtypes(include="number")
        for col in numeric_df.columns:
            series = numeric_df[col]
            z = (series - series.mean()) / (series.std() + 1e-6)
            outliers[col] = int((z.abs() > 3).sum())
        return outliers

    @staticmethod
    def _compute_correlations(df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Return correlation matrix including encoded target if needed."""
        numeric_df = df.select_dtypes(include="number").copy()
        if target in df.columns and target not in numeric_df.columns:
            numeric_df[target] = df[target].astype("category").cat.codes
        corr = numeric_df.corr().to_dict()
        return corr

    @classmethod
    def _infer_domain_insights(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """Return column semantics, temporal patterns and categorical cardinality."""
        semantics = cls._infer_column_semantics(df)
        temporal = cls._extract_temporal_patterns(df)
        cardinality = cls._categorical_cardinality(df)
        return {
            "column_semantics": semantics,
            "temporal_patterns": temporal,
            "categorical_cardinality": cardinality,
        }

    @staticmethod
    def _infer_column_semantics(df: pd.DataFrame) -> Dict[str, str]:
        """Infer simple semantics for each column."""
        semantics: Dict[str, str] = {}
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_bool_dtype(series):
                semantics[col] = "boolean"
            elif pd.api.types.is_datetime64_any_dtype(series):
                semantics[col] = "datetime"
            elif pd.api.types.is_numeric_dtype(series):
                semantics[col] = "numeric"
            elif series.nunique(dropna=False) == len(series):
                semantics[col] = "identifier"
            else:
                semantics[col] = "categorical"
        return semantics

    @staticmethod
    def _extract_temporal_patterns(df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze datetime columns for range and dominant interval."""
        temporal: Dict[str, Any] = {}
        for col in df.select_dtypes(include="datetime"):
            series = df[col].dropna().sort_values()
            if series.empty:
                continue
            diffs = series.diff().dropna()
            freq = None
            if not diffs.empty:
                freq = diffs.mode().iloc[0]
                if hasattr(freq, "to_pytimedelta"):
                    freq = freq.to_pytimedelta()
            temporal[col] = {
                "min": str(series.min()),
                "max": str(series.max()),
                "most_common_interval": str(freq) if freq is not None else None,
            }
        return temporal

    @staticmethod
    def _categorical_cardinality(df: pd.DataFrame) -> Dict[str, int]:
        """Return unique value counts for categorical columns."""
        cat_cols = df.select_dtypes(include=["object", "category"])
        return {col: int(cat_cols[col].nunique()) for col in cat_cols.columns}

    @staticmethod
    def _calculate_complexity_metrics(df: pd.DataFrame, target: str) -> Dict[str, Any]:
        """Calculate dataset complexity metrics."""
        n_features = df.shape[1] - (1 if target in df.columns else 0)
        n_rows = len(df)
        feature_target_ratio = n_features / n_rows if n_rows else 0.0

        class_imbalance = None
        if target in df.columns and df[target].dtype != float and df[target].dtype != int:
            counts = df[target].value_counts(normalize=True)
            class_imbalance = counts.to_dict()

        noise_level = float(df.duplicated().mean())

        return {
            "feature_target_ratio": feature_target_ratio,
            "class_imbalance": class_imbalance,
            "noise_level": noise_level,
        }
