import os
import pandas as pd

# Data path set karein
DATA_PATH = os.path.join("data", "dataset.csv")

# Column names
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

if os.path.exists(DATA_PATH):
    print("--> Loading Dataset...")
    df = pd.read_csv(DATA_PATH, names=columns)
    print("\n[SUCCESS] Dataset Loaded Successfully!")
    print(f"Total Connections Logged: {df.shape[0]}")
    
    # Attack vs Normal Traffic
    df['is_attack'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
    
    print("\n--- Traffic Summary ---")
    normal_count = (df['is_attack'] == 0).sum()
    attack_count = (df['is_attack'] == 1).sum()
    
    print(f"Normal Traffic: {normal_count} ({normal_count/len(df)*100:.2f}%)")
    print(f"Malicious Traffic: {attack_count} ({attack_count/len(df)*100:.2f}%)")
else:
    print(f"[ERROR] '{DATA_PATH}' file nahi mili! Pehle dataset download karke data folder me rakhein.")