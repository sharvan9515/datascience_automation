from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import os
import pandas as pd

# Set your dataset path and target column here
csv_path = 'your_data.csv'  # TODO: set your CSV file path
target = 'your_target_column'  # TODO: set your target column name
df = pd.read_csv(csv_path)
# Engineered features included in this pipeline: ['Age_Pclass', 'Family_Size']
# --- Preprocessing ---
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['PassengerId'] = df['PassengerId'].fillna(df['PassengerId'].mean())
df['Survived'] = df['Survived'].fillna(df['Survived'].mean())
df['Pclass'] = df['Pclass'].fillna(df['Pclass'].mean())
df['Name'] = df['Name'].fillna(df['Name'].mean())
df['Sex'] = df['Sex'].fillna(df['Sex'].mean())
df['SibSp'] = df['SibSp'].fillna(df['SibSp'].mean())
df['Parch'] = df['Parch'].fillna(df['Parch'].mean())
df['Ticket'] = df['Ticket'].fillna(df['Ticket'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
df['Cabin'] = df['Cabin'].fillna(df['Cabin'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mean())
# --- Feature Engineering ---
df['Family_Size'] = df['SibSp'] + df['Parch'] + 1
df['Age_Pclass'] = df['Age'] * df['Pclass']
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