from __future__ import annotations

import numpy as np
import pandas as pd


class TemporalFeatureEngineer:
    """Create common time series features."""

    @staticmethod
    def create_timeseries_features(df: pd.DataFrame, target: str, time_col: str) -> pd.DataFrame:
        df = df.sort_values(time_col).copy()
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        df[f"{target}_lag_1"] = df[target].shift(1)
        df[f"{target}_lag_7"] = df[target].shift(7)
        df[f"{target}_lag_30"] = df[target].shift(30)

        df[f"{target}_rolling_mean_7"] = df[target].shift(1).rolling(window=7).mean()
        df[f"{target}_rolling_std_7"] = df[target].shift(1).rolling(window=7).std()

        df["hour"] = df[time_col].dt.hour
        df["day_of_week"] = df[time_col].dt.dayofweek
        df["month"] = df[time_col].dt.month
        df["quarter"] = df[time_col].dt.quarter

        df["sin_day_of_year"] = np.sin(2 * np.pi * df[time_col].dt.dayof_year / 365.25)
        df["cos_day_of_year"] = np.cos(2 * np.pi * df[time_col].dt.day_of_year / 365.25)

        df["time_since_start"] = (df[time_col] - df[time_col].min()).dt.total_seconds()
        df["linear_trend"] = np.arange(len(df))

        df["change_1d"] = df[target].diff()
        df["change_7d"] = df[target].diff(7)
        df["rate_of_change"] = df[target].pct_change()
        df["acceleration"] = df["rate_of_change"].diff()

        return df
