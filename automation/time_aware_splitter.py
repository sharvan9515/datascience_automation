import pandas as pd

class TimeAwareSplitter:
    """Perform chronological train/test splits for timeseries data."""

    @staticmethod
    def chronological_split(
        df: pd.DataFrame, time_col: str, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return train/test DataFrames split chronologically by ``time_col``."""
        df_sorted = df.sort_values(time_col)
        n_test = int(len(df_sorted) * test_size)
        train = df_sorted.iloc[:-n_test]
        test = df_sorted.iloc[-n_test:]
        return train, test
