import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from automation.time_aware_splitter import TimeAwareSplitter
from sklearn.metrics import f1_score, r2_score, accuracy_score
import argparse


def agentic_preprocessing(df, target):
    """
    This function should be replaced by an agent/LLM call that:
    - Analyzes df's schema and sample
    - Decides which columns to encode, drop, or impute
    - Returns the processed DataFrame and logs all actions
    """
    logs = []
    for col in df.columns:
        if col == target:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
                logs.append(f"Filled missing values in numeric column '{col}' with mean.")
        else:
            df[col] = df[col].astype('category').cat.codes
            logs.append(f"Encoded categorical column '{col}' as category codes.")
    return df, logs


def run_basic_template(df, target, task_type, time_col: str | None = None):
    df, logs = agentic_preprocessing(df, target)
    for log in logs:
        print('[Preprocessing]', log)
    X = df.drop(columns=[target])
    y = df[target]
    if time_col:
        train_df, test_df = TimeAwareSplitter.chronological_split(df, time_col, test_size=0.2)
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target]
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if task_type == 'classification':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds, average='weighted')
        acc = accuracy_score(y_test, preds)
        print('F1 score:', f1)
        print('Accuracy:', acc)
        score = f1
    else:
        model = XGBRegressor()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        print('R2 score:', r2)
        score = r2
    return df, model, score

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agentic Data Science Baseline')
    parser.add_argument('csv_path', type=str, help='Path to CSV file')
    parser.add_argument('target', type=str, help='Target column name')
    parser.add_argument('task_type', type=str, choices=['classification', 'regression'], help='Task type')
    parser.add_argument('--time-col', type=str, default=None, help='Optional time column for chronological split')
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    run_basic_template(df, args.target, args.task_type, time_col=args.time_col)