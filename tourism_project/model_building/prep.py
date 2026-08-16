import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

DATASET_PATH = "tourism_project/data/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Drop unnecessary columns
DROP_COLUMNS = ["CustomerID", "Unnamed: 0"]
df_clean = df.drop(columns=DROP_COLUMNS, errors="ignore")
print(f"Cleaned dataset shape: {df_clean.shape}")

# Define target variable
target = "ProdTaken"

X = df_clean.drop(columns=[target])
y = df_clean[target]

# Split dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Original class distribution in training set:")
print(ytrain.value_counts())

# Handle class imbalance with oversampling
ros = RandomOverSampler(random_state=42)
Xtrain_resampled, ytrain_resampled = ros.fit_resample(Xtrain, ytrain)

print("Resampled class distribution in training set:")
print(ytrain_resampled.value_counts())

# Save splits locally
Xtrain_resampled.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain_resampled.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Saved balanced Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv locally.")
