import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import joblib

def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Run assembled pipeline')
    parser.add_argument('csv', help='Path to CSV file')
    parser.add_argument('target', help='Target column name')
    parser.add_argument('--max-iter', type=int, default=10, help='Maximum iterations')
    parser.add_argument('--patience', type=int, default=5, help='Rounds without improvement')
    parsed = parser.parse_args(args)
    df = pd.read_csv(parsed.csv)
    target = parsed.target
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(**{'n_estimators': 100, 'max_depth': 5})
    model.fit(X_train, y_train)
    joblib.dump(model, 'artifacts/model.pkl')

if __name__ == '__main__':
    main()