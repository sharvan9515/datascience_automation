from __future__ import annotations

import pandas as pd

from automation.pipeline_state import PipelineState
from .base import BaseAgent


class TimeseriesDetectionAgent(BaseAgent):
    """Detect whether the dataset has temporal structure."""

    def _detect_time_column(self, df: pd.DataFrame) -> str | None:
        # Prefer existing datetime columns
        time_cols = list(df.select_dtypes(include="datetime").columns)
        if time_cols:
            return time_cols[0]
        # Fallback: attempt to parse columns with 'date' or 'time' in the name
        for col in df.columns:
            if any(token in col.lower() for token in ["date", "time"]):
                try:
                    pd.to_datetime(df[col])
                    return col
                except Exception:  # noqa: BLE001
                    continue
        return None

    def detect_timeseries(self, df: pd.DataFrame, target: str) -> tuple[bool, str | None]:
        time_col = self._detect_time_column(df)
        if time_col is None:
            return False, None
        # Basic check: ensure the time column is sortable and has multiple values
        try:
            pd.to_datetime(df[time_col])
        except Exception:  # noqa: BLE001
            return False, None
        return True, time_col

    def run(self, state: PipelineState) -> PipelineState:
        detected, time_col = self.detect_timeseries(state.df, state.target)
        state.timeseries_mode = detected
        state.time_col = time_col
        if detected:
            state.append_log(f"TimeseriesDetection: detected time column '{time_col}'")
        else:
            state.append_log("TimeseriesDetection: no time column detected")
        return state


def run(state: PipelineState) -> PipelineState:
    return TimeseriesDetectionAgent().run(state)
