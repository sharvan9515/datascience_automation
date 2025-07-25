import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from main_pipeline import DataSciencePipeline
from automation.pipeline_state import PipelineState


@pytest.mark.asyncio
async def test_async_pipeline_run(monkeypatch):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "target": [0, 1, 0]})
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df.to_csv(tmp.name, index=False)
        csv_path = tmp.name

    def fake_run(state: PipelineState):
        state.best_score = 0.42
        return state

    def fake_assemble(state: PipelineState):
        return state

    monkeypatch.setattr("automation.agents.orchestrator.run", fake_run)
    monkeypatch.setattr("automation.code_assembler.run", fake_assemble)

    pipeline = DataSciencePipeline(csv_path, "target")
    final = await pipeline.run()

    os.remove(csv_path)
    assert final.score == 0.42
