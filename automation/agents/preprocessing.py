import pandas as pd
from automation.pipeline_state import PipelineState


def run(state: PipelineState) -> PipelineState:
    df = state.df.copy()
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'O':
                fill_value = df[col].mode().iloc[0]
            else:
                fill_value = df[col].mean()
            df[col] = df[col].fillna(fill_value)
            state.append_log(f"Preprocessing: filled missing values in {col} with {fill_value}")
    state.df = df
    return state
