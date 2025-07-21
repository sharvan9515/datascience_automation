from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import re

# Set your dataset path and target column here
csv_path = 'your_data.csv'  # TODO: set your CSV file path
target = 'your_target_column'  # TODO: set your target column name
df = pd.read_csv(csv_path)
# Engineered features included in this pipeline: ['Age_x_Fare', 'Family_Size_x_Fare']
# --- Preprocessing ---
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Sex'] = df['Sex'].astype('category').cat.codes
df = pd.get_dummies(df, columns=['Embarked'])
# --- Feature Engineering ---
df['Family_Size_x_Fare'] = (df['SibSp'] + df['Parch'] + 1) * df['Fare']
df['Age_x_Fare'] = df['Age'] * df['Fare']
# --- Modeling ---
X = df.drop(columns=['Survived'])
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(**{'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1})
model.fit(X_train, y_train)
joblib.dump(model, 'artifacts/model.pkl')
param_grid = {'n_estimators': [100, 200], 'max_depth': [None, 5, 10]}
search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1_weighted')
search.fit(X_train, y_train)
best_model = search.best_estimator_
joblib.dump(best_model, 'artifacts/model.pkl')
best_params = search.best_params_