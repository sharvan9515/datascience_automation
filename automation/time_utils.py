from __future__ import annotations

import pandas as pd
from typing import List, Tuple


class TimeAwareSplitter:
    """Utilities for chronological train/test splits."""

    @staticmethod
    def chronological_split(df: pd.DataFrame, time_col: str, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_sorted = df.sort_values(time_col)
        split_idx = int(len(df_sorted) * (1 - test_size))
        train = df_sorted.iloc[:split_idx]
        test = df_sorted.iloc[split_idx:]
        return train, test


class TimeAwareValidation:
    """Simple walk-forward cross-validation."""

    @staticmethod
    def time_series_cv(df: pd.DataFrame, time_col: str, n_splits: int = 5) -> List[Tuple[pd.Index, pd.Index]]:
        df_sorted = df.sort_values(time_col)
        fold_size = len(df_sorted) // (n_splits + 1)
        splits: List[Tuple[pd.Index, pd.Index]] = []
        for i in range(1, n_splits + 1):
            train_end = i * fold_size
            test_end = train_end + fold_size
            train_idx = df_sorted.index[:train_end]
            test_idx = df_sorted.index[train_end:test_end]
            splits.append((train_idx, test_idx))
        return splits
