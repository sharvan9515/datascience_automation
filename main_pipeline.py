"""Asynchronous wrapper around the orchestrator-driven pipeline."""

# TODO: Add resume functionality to reload a saved PipelineState and
# continue execution without starting from scratch.

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import pandas as pd

from automation.pipeline_state import PipelineState
from automation.agents import orchestrator
from automation import code_assembler
from automation.models import FinalPipeline


class DataSciencePipeline:
    """Load data and orchestrate the agentic pipeline."""

    def __init__(self, csv_path: str, target_column: str) -> None:
        self.csv_path = csv_path
        self.target_column = target_column
        self.state: PipelineState | None = None

    async def load_csv(self) -> None:
        """Load the CSV file asynchronously into :class:`PipelineState`."""
        df = await asyncio.to_thread(pd.read_csv, self.csv_path)
        self.state = PipelineState(df=df, target=self.target_column)

    async def run(self) -> FinalPipeline:
        """Execute the orchestration loop and assemble final code."""
        if self.state is None:
            await self.load_csv()

        assert self.state is not None
        final_state = await asyncio.to_thread(orchestrator.run, self.state)
        await asyncio.to_thread(code_assembler.run, final_state)

        model_file = Path("artifacts/model.pkl")
        return FinalPipeline(
            code_blocks=final_state.code_blocks,
            model_path=str(model_file) if model_file.exists() else None,
            score=final_state.best_score,
            logs=final_state.log,
        )


async def _async_main(csv_path: str, target: str) -> None:
    pipeline = DataSciencePipeline(csv_path, target)
    final_pipeline = await pipeline.run()
    if final_pipeline.code_blocks:
        total = sum(len(v) for v in final_pipeline.code_blocks.values())
        print(f"Assembled pipeline with {total} code blocks")
    print("Final score:", final_pipeline.score)
    print("Code written to finalcode.py")


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run async data science pipeline")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("target", help="Target column name")
    parsed = parser.parse_args(args)
    asyncio.run(_async_main(parsed.csv, parsed.target))


if __name__ == "__main__":
    main()
