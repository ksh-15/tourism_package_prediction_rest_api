import os
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("tourism_project/data/tourism.csv")

# Data Validation

# Data Cleaning and Engineering
# Fix Gender inconsistencies ('Fe male' -> 'Female')
if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe male', 'Female')
        print("Standardized 'Gender' column values.")

# Fix Marital Status inconsistencies ('Unmarried' -> 'Single')
if 'MaritalStatus' in df.columns:
   df['MaritalStatus'] = df['MaritalStatus'].replace('Unmarried', 'Single')
   print("Standardized 'MaritalStatus' column values.")

# Remove unnecessary columns
cols_to_drop = [col for col in ['Unnamed: 0', 'CustomerID'] if col in df.columns]
if cols_to_drop:
   df = df.drop(columns=cols_to_drop)
   print(f"Dropped unnecessary columns: {cols_to_drop}")

# 2. Split the data into training and testing sets
target_col = 'ProdTaken'
stratify_target = df[target_col] if target_col in df.columns else None

train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target
    )

# Save locally as CSV files in the model_building directory
os.makedirs("tourism_project/model_building", exist_ok=True)
train_path = "tourism_project/model_building/train.csv"
test_path = "tourism_project/model_building/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)
print(f"Successfully saved processed datasets: {train_path} & {test_path}")

