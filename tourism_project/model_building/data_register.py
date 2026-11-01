import os
import pandas as pd

RAW_PATH = "tourism_project/data/tourism.csv"

# Load the raw dataset
df = pd.read_csv(RAW_PATH)

# Validate critical expected columns
expected_columns = ['ProdTaken', 'Age', 'TypeofContact', 'Gender', 'MaritalStatus', 'MonthlyIncome']
missing_cols = [col for col in expected_columns if col not in df.columns]

if missing_cols:
    print(f"[Validation Failed] Missing critical columns: {missing_cols}")
else:
    print("[Validation Passed] All critical columns exist in the dataset.")

# Print short summary
print("\n--- Dataset Summary ---")
print(f"Total Records: {len(df)}")
print(f"Total Features: {len(df.columns)}")
print("\nClass Distribution for 'ProdTaken':")
print(df['ProdTaken'].value_counts(normalize=True))
