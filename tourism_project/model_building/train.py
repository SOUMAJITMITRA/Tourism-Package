import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

print("Train and test splits loaded.")

categorical_cols = Xtrain.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

model = RandomForestClassifier(random_state=42)
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", model)])

param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5],
}

mlflow.set_experiment("wellness_tourism_prediction")

with mlflow.start_run():
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    mlflow.log_params(best_params)

    ypred = best_model.predict(Xtest)
    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred)
    roc_auc = roc_auc_score(ytest, best_model.predict_proba(Xtest)[:, 1])

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    deployment_dir = "tourism_project/deployment"
    os.makedirs(deployment_dir, exist_ok=True)
    joblib.dump(best_model, os.path.join(deployment_dir, "best_model.pkl"))

    mlflow.sklearn.log_model(best_model, "model")

    print("Best model saved to tourism_project/deployment/best_model.pkl")
