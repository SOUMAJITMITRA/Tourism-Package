import pandas as pd
import sys

DATA_PATH = "data/tourism.csv"

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
        df = pd.read_csv(DATA_PATH)
        print("✅ Dataset loaded successfully.")
    except FileNotFoundError:
        print(f"❌ Dataset not found at {DATA_PATH}")
        sys.exit(1)

    # Check for missing/extra columns
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]

    if missing_cols:
        print("⚠️ Missing columns:", missing_cols)
    else:
        print("✅ All expected columns are present.")

    if extra_cols:
        print("ℹ️ Extra columns found:", extra_cols)

    # Print summary
    print("\n--- Dataset Summary ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumn Data Types:")
    print(df.dtypes)
    print("\nMissing Values per Column:")
    print(df.isnull().sum())
    print("\nSample Records:")
    print(df.head(5))

if __name__ == "__main__":
    main()
