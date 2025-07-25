import asyncio
import json
import os
import tempfile

import pandas as pd
import streamlit as st

from main_pipeline import DataSciencePipeline


async def run_async_pipeline(csv_path: str, target: str):
    """Execute DataSciencePipeline asynchronously and return results."""
    pipeline = DataSciencePipeline(csv_path, target)
    return await pipeline.run()


st.title("Agentic Data Science Pipeline")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded data:")
    st.dataframe(df.head())

    target = st.selectbox("Select target column", df.columns)

    if st.button("Run Pipeline"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        try:
            with st.spinner("Running pipeline..."):
                final = asyncio.run(run_async_pipeline(csv_path, target))
            if final.score is not None:
                st.metric("Final Model Score", final.score)
            if os.path.exists("finalcode.py"):
                with open("finalcode.py") as f:
                    code_content = f.read()
                st.subheader("Final Assembled Code")
                st.code(code_content, language="python")
                st.download_button(
                    "Download finalcode.py", code_content, file_name="finalcode.py"
                )
            if os.path.exists("output/logs.json"):
                with open("output/logs.json") as f:
                    logs = json.load(f)
                st.subheader("Last 5 Log Entries")
                st.code("\n".join(logs[-5:]))
        finally:
            os.remove(csv_path)
