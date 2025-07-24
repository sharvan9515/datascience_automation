from __future__ import annotations

import pandas as pd

from automation.pipeline_state import PipelineState
from .base import BaseAgent
from automation.timeseries_detection import TimeseriesDetectionUtility


class TimeseriesDetectionAgent(BaseAgent):
    """Detect whether the dataset has temporal structure."""

    @staticmethod
    def detect_timeseries(df: pd.DataFrame, target: str) -> tuple[bool, str | None]:
        """Delegate detection to :class:`TimeseriesDetectionUtility`."""

        return TimeseriesDetectionUtility.detect_timeseries(df, target)

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
