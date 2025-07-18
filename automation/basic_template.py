import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score

# Set your dataset path and target column here
csv_path = 'your_data.csv'  # TODO: set your CSV file path
# target = 'your_target_column'  # TODO: set your target column name
# task_type = 'classification'  # or 'regression'

df = pd.read_csv(csv_path)

# --- BASIC PREPROCESSING ---
# Drop columns that are clearly non-numeric and not useful for modeling
for col in ['Name', 'Ticket', 'Cabin', 'PassengerId']:
    if col in df.columns:
        df = df.drop(columns=[col])

# Fill missing values for numeric columns
for col in df.select_dtypes(include=[np.number]).columns:
    df[col] = df[col].fillna(df[col].mean())

# Encode categorical columns
for col in df.select_dtypes(include=['object', 'category']).columns:
    if col != 'target':  # Don't encode the target
        df[col] = df[col].astype('category').cat.codes

# --- SPLIT DATA ---
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL TRAINING & EVALUATION ---
if task_type == 'classification':
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('F1 score:', f1_score(y_test, preds, average='weighted'))
else:
    model = XGBRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('R2 score:', r2_score(y_test, preds)) 