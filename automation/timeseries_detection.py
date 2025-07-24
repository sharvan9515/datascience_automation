import pandas as pd

class TimeseriesDetectionUtility:
    """Detect if a dataset is timeseries and identify the time column."""

    @staticmethod
    def detect_timeseries(df: pd.DataFrame, target: str) -> tuple[bool, str | None]:
        # 1. Look for datetime columns
        datetime_cols = [
            col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
        ]

        # 2. If none, look for common time column names and try to convert
        if not datetime_cols:
            for col in df.columns:
                if any(name in col.lower() for name in ["date", "time", "timestamp"]):
                    try:
                        pd.to_datetime(df[col])
                        datetime_cols.append(col)
                        break
                    except Exception:
                        continue

        if not datetime_cols:
            return False, None

        time_col = datetime_cols[0]

        # 3. Target should be numeric and have variation over time
        if not pd.api.types.is_numeric_dtype(df[target]):
            return False, None

        if df.sort_values(time_col)[target].nunique() <= 1:
            return False, None

        return True, time_col
