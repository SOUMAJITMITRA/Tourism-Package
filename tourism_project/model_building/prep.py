# for data manipulation
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define constants for dataset path
DATASET_PATH = "data/tourism.csv"

# Load dataset
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully.")

# Drop unnecessary columns
DROP_COLUMNS = ["CustomerID"]
df_clean = df.drop(columns=DROP_COLUMNS, errors="ignore")
print(f"🧹 Cleaned dataset shape: {df_clean.shape}")

# Define target variable
target = "ProdTaken"

# Split predictors (X) and target (y)
X = df_clean.drop(columns=[target])
y = df_clean[target]

# Split dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save splits locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("💾 Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv locally.")
