import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully.")

#DROP_COLUMNS = ["CustomerID"]

DROP_COLUMNS = ["CustomerID", "Unnamed: 0"]
df_clean = df.drop(columns=DROP_COLUMNS, errors="ignore")
print(f"🧹 Cleaned dataset shape: {df_clean.shape}")

target = "ProdTaken"

X = df_clean.drop(columns=[target])
y = df_clean[target]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("💾 Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv locally.")
