from pathlib import Path
import ast

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb

# DIR
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "ptb"
RESULT_DIR = Path(__file__).parent.parent / "results"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 1. LOAD DATA
df = pd.read_csv(DATA_DIR / "ptbxl_database.csv", index_col='ecg_id')

df['scp_codes'] = df['scp_codes'].apply(
    ast.literal_eval
)

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


assert signal.shape == (1000,12)
assert metadata['fs'] == 100
assert np.isfinite(signal).all()

# 3. PLOT 
lead_index = metadata['sig_name'].index('II')  # Lead II
time = np.arange(len(signal)) / metadata['fs']

plt.figure(figsize=(12, 4))

plt.plot(time, signal[:, lead_index])
plt.title(f"ECG Signal - Lead II (Patient ID: {record['patient_id']})")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (mV)")
plt.grid()

plt.tight_layout()
plt.savefig(RESULT_DIR / "sample_ecg_lead_II.png")
plt.show()