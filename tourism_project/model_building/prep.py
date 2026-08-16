import pandas as pd
from sklearn.model_selection import train_test_split

# Path to the dataset file
DATASET_PATH = "tourism_project/data/tourism.csv"

# Load the dataset into a pandas DataFrame
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Columns to drop from the dataset (unique identifiers or unnecessary fields)
DROP_COLUMNS = ["CustomerID", "Unnamed: 0"]

# Remove the specified columns from the dataset
df_clean = df.drop(columns=DROP_COLUMNS)
print(f"Cleaned dataset shape: {df_clean.shape}")

# Define the target column (label) for prediction
target = "ProdTaken"

# Separate features (X) and target (y)
X = df_clean.drop(columns=[target])
y = df_clean[target]

# Split the dataset into training and testing sets
# - test_size=0.2 means 20% of the data will be used for testing
# - random_state=42 ensures reproducibility of the split
# - stratify=y ensures class distribution is preserved in both train and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Save the train and test splits as CSV files for later use
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Confirmation message after saving the files
print("Saved Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv locally.")
