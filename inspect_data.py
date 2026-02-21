import pandas as pd
import numpy as np

DATA_DIR = r"c:\Users\aadit\Documents\vs code\kaggle-royale\dataset"

train = pd.read_csv(f"{DATA_DIR}/matches_train.csv", nrows=5)
test  = pd.read_csv(f"{DATA_DIR}/matches_test.csv",  nrows=5)
pb    = pd.read_csv(f"{DATA_DIR}/players_behavior.csv", nrows=5)
ss    = pd.read_csv(f"{DATA_DIR}/sample_submission.csv", nrows=5)

print("=== TRAIN COLUMNS ===")
print(list(train.columns))
print(train.dtypes.to_string())
print()
print("=== TRAIN SAMPLE ===")
print(train.head(3).to_string())

print()
print("=== TEST COLUMNS ===")
print(list(test.columns))
print(test.dtypes.to_string())
print()
print("=== TEST SAMPLE ===")
print(test.head(3).to_string())

print()
print("=== PLAYERS_BEHAVIOR COLUMNS ===")
print(list(pb.columns))
print(pb.dtypes.to_string())
print()
print("=== PB SAMPLE ===")
print(pb.head(3).to_string())

print()
print("=== SAMPLE_SUBMISSION ===")
print(list(ss.columns))
print(ss.head(3).to_string())

# Full row counts
train_full = pd.read_csv(f"{DATA_DIR}/matches_train.csv")
test_full  = pd.read_csv(f"{DATA_DIR}/matches_test.csv")
pb_full    = pd.read_csv(f"{DATA_DIR}/players_behavior.csv")
print()
print(f"train shape: {train_full.shape}")
print(f"test  shape: {test_full.shape}")
print(f"pb    shape: {pb_full.shape}")
print()
print("=== TRAIN MISSING ===")
print(train_full.isnull().sum().to_string())
print()
print("=== TEST MISSING ===")
print(test_full.isnull().sum().to_string())
print()
print("=== PB MISSING ===")
print(pb_full.isnull().sum().to_string())
print()
print("=== TRAIN TARGET DIST ===")
if "player_wins" in train_full.columns:
    print(train_full["player_wins"].value_counts())
