from pathlib import Path
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# DIR
DATA_DIR = Path(__file__).parent.parent / "data"
RESULT_DIR = Path(__file__).parent.parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 1. LOAD DATA
df = pd.read_csv(DATA_DIR / "data.csv", index_col='ecg_id')

print("Data loaded successfully.")
print(f"Data shape: {df.shape}")
print(f"# Patients: {df['patient_id'].nunique()}")

print(df.head())
print(df.info())

# 2. SHOW SAMPLE DATA 
record = df.iloc[0]
record_path = DATA_DIR / record['filename_lr']

signal, metadata = wfdb.rdsamp(str(record_path))

print(f"Signal shape: {signal.shape}")
print(f"Frequency: {metadata['fs']} Hz")
print(f"Number of leads: {metadata['sig_name']}")


