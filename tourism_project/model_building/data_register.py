import pandas as pd
import sys

# Path to the dataset file
DATA_PATH = "tourism_project/data/tourism.csv"

# Define the expected schema (columns) for the dataset
EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "Occupation", "Gender", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "OwnCar",
    "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
    "PitchSatisfactionScore", "ProductPitched", "NumberOfFollowups",
    "DurationOfPitch"
]

def main():
    try:
        # Attempt to load the dataset
        df = pd.read_csv(DATA_PATH)
        print("Dataset loaded successfully.")
    except FileNotFoundError:
        # If the dataset is not found, print an error and exit
        print(f"Dataset not found at {DATA_PATH}")
        sys.exit(1)

    # Identify missing columns (expected but not present in the dataset)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    # Identify extra columns (present in dataset but not expected)
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    # Report missing columns if any
    if missing_cols:
        print("Missing columns:", missing_cols)
    else:
        print("All expected columns are present.")

    # Report extra columns if any
    if extra_cols:
        print("Extra columns found:", extra_cols)

    # Print dataset summary
    print("\n--- Dataset Summary ---")
    # Show number of rows and columns
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    # Show data types of each column
    print("\nColumn Data Types:")
    print(df.dtypes)
    # Show count of missing values per column
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    # Show first 5 sample records
    print("\nSample Records:")
    print(df.head(5))

# Entry point of the script
if __name__ == "__main__":
    main()
