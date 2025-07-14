from automation.pipeline_state import PipelineState
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression


def run(state: PipelineState) -> PipelineState:
    df = state.df
    X = df.drop(columns=[state.target])
    y = df[state.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if state.task_type == 'classification':
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        report = classification_report(y_test, preds)
        state.append_log(f"ModelEvaluation:\n{report}")
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        state.append_log(f"ModelEvaluation: mse={mse:.4f}")
    return state
