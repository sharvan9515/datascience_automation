import streamlit as st
import pandas as pd
import tempfile
import os
import subprocess
import re
import json

st.title('Agentic Data Science Pipeline')

# 1. File uploader
uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('Preview of uploaded data:')
    st.dataframe(df.head())
    # 2. Target column selector
    target = st.selectbox('Select target column', df.columns)
    # 3. Patience input
    patience = st.number_input('Set patience (early stopping rounds)', min_value=1, max_value=100, value=20)
    # 4. Run button
    if st.button('Run Pipeline'):
        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            df.to_csv(tmp.name, index=False)
            csv_path = tmp.name
        # Build command (remove max-iter logic)
        cmd = [
            'python', 'run_agentic_pipeline.py',
            csv_path, target,
            '--patience', str(patience)
        ]
        with st.spinner('Running pipeline...'):
            result = subprocess.run(cmd, capture_output=True, text=True)
        # Extract and display final model score
        score = None
        match_acc = re.search(r'Accuracy:\s*([0-9.]+)', result.stdout)
        match_f1 = re.search(r'F1 score:\s*([0-9.]+)', result.stdout)
        if match_acc:
            score = float(match_acc.group(1))
            st.metric('Final Model Accuracy', score)
        elif match_f1:
            score = float(match_f1.group(1))
            st.metric('Final Model F1 Score', score)
        # Only show final assembled code, not pipeline output or errors
        if os.path.exists('finalcode.py'):
            with open('finalcode.py') as f:
                code_content = f.read()
            st.subheader('Final Assembled Code')
            st.code(code_content, language='python')
            st.download_button('Download finalcode.py', code_content, file_name='finalcode.py')

        if os.path.exists('output/logs.json'):
            with open('output/logs.json') as f:
                logs = json.load(f)
            st.subheader('Last 5 Log Entries')
            st.code('\n'.join(logs[-5:]))
