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

# Load the training and testing datasets created in prep.py
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()  # squeeze converts to Series
ytest = pd.read_csv("ytest.csv").squeeze()

print("Train and test splits loaded.")

# Identify categorical columns (object type) for preprocessing
categorical_cols = Xtrain.select_dtypes(include=["object"]).columns

# Preprocessor: OneHotEncode categorical columns, pass through numeric columns unchanged
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

# Define the model
# class_weight="balanced" ensures minority classes are weighted more
