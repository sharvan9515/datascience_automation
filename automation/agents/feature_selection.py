from automation.pipeline_state import PipelineState
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LogisticRegression, LinearRegression


def run(state: PipelineState) -> PipelineState:
    df = state.df
    X = df.drop(columns=[state.target])
    y = df[state.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if state.task_type == 'classification':
        model = LogisticRegression(max_iter=200)
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        score = r2_score(y_test, model.predict(X_test))
    state.append_log(f"FeatureSelection: baseline score {score:.4f}")
    return state
