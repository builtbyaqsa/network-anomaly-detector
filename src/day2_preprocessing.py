import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Load Data
DATA_PATH = os.path.join("data", "dataset.csv")

columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

print("--> Loading dataset for preprocessing...")
df = pd.read_csv(DATA_PATH, names=columns)

# Drop difficulty level column (not needed for detection)
if 'difficulty_level' in df.columns:
    df.drop(columns=['difficulty_level'], inplace=True)

# 2. Binary Target Variable Creation
# Normal = 0, Attack = 1
df['target'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
df.drop(columns=['label'], inplace=True)

# 3. Categorical Encoding (Text -> Numbers)
categorical_cols = ['protocol_type', 'service', 'flag']
print(f"--> Encoding categorical columns: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Save Processed Dataset
OUTPUT_PATH = os.path.join("data", "processed_dataset.csv")
df.to_csv(OUTPUT_PATH, index=False)

print("\n[SUCCESS] Preprocessing Complete!")
print(f"Processed file saved at: {OUTPUT_PATH}")
print(f"Dataset Shape: {df.shape} (Rows, Columns)")
print("\nFirst 5 Rows Preview:")
print(df[['protocol_type', 'service', 'flag', 'target']].head())