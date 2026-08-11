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

# Paths for train/test splits
Xtrain_path = "Xtrain.csv"
Xtest_path = "Xtest.csv"
ytrain_path = "ytrain.csv"
ytest_path = "ytest.csv"

# Load data
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

print("✅ Train and test splits loaded.")

# Identify categorical columns
categorical_cols = Xtrain.select_dtypes(include=["object"]).columns

# Preprocessor: OneHotEncode categorical columns, pass through numeric ones
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

# Define pipeline: preprocessing + classifier
model = RandomForestClassifier(random_state=42)
pipeline = Pipeline(steps=[("preprocessor", preprocessor),
                           ("classifier", model)])

# Hyperparameter grid (note the "classifier__" prefix)
param_grid = {
    "classifier__n_estimators": [50, 100, 200],
    "classifier__max_depth": [None, 5, 10],
    "classifier__min_samples_split": [2, 5],
}

# MLflow experiment setup
mlflow.set_experiment("wellness_tourism_prediction")

with mlflow.start_run():
    # Hyperparameter tuning
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Log parameters
    mlflow.log_params(best_params)

    # Evaluate
    ypred = best_model.predict(Xtest)
    acc = accuracy_score(ytest, ypred)
    f1 = f1_score(ytest, ypred)
    roc_auc = roc_auc_score(ytest, best_model.predict_proba(Xtest)[:, 1])

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    print(f"📊 Accuracy: {acc:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Save best model locally
    deployment_dir = "tourism_project/deployment"
    os.makedirs(deployment_dir, exist_ok=True)
    model_path = os.path.join(deployment_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)

    mlflow.sklearn.log_model(best_model, "model")

    print(f"💾 Best model saved to {model_path}")
