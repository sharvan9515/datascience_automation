import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, r2_score, mean_squared_error, mean_absolute_error

# Set these variables
csv_path = 'your_dataset.csv'  # <-- Set your CSV file path here
target = 'your_target_column'  # <-- Set your target column here

# Load data
df = pd.read_csv(csv_path)

# Infer task type
target_is_numeric = pd.api.types.is_numeric_dtype(df[target])
if target_is_numeric and df[target].nunique() > 20:
    task_type = 'regression'
else:
    task_type = 'classification'

# Preprocess features
def preprocess(df, target):
    df = df.copy()
    # Encode non-numeric features
    for col in df.columns:
        if col == target:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Drop high-cardinality or free-text columns
            if df[col].nunique(dropna=False) > min(50, len(df) * 0.3):
                df = df.drop(columns=[col])
            else:
                df[col] = df[col].astype('category').cat.codes
        if df[col].isnull().any():
            fill_value = df[col].mean() if pd.api.types.is_numeric_dtype(df[col]) else df[col].mode()[0]
            df[col] = df[col].fillna(fill_value)
    # Encode target if needed
    if not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = df[target].astype('category').cat.codes
    if df[target].isnull().any():
        fill_value = df[target].mean() if pd.api.types.is_numeric_dtype(df[target]) else df[target].mode()[0]
        df[target] = df[target].fillna(fill_value)
    return df

df = preprocess(df, target)

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if task_type == 'classification':
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, preds))
    print('Classification Report:')
    print(classification_report(y_test, preds))
else:
    model = XGBRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    print(f'R2 Score: {r2:.4f}')
    print(f'MSE: {mse:.4f}')
    print(f'MAE: {mae:.4f}') 