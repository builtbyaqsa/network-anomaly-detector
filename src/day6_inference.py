import os
import joblib
import pandas as pd
import numpy as np

# 1. Load Trained Artifacts
MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

print("--> Loading model and scaler for inference...")
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_network_traffic(sample_data):
    """
    Accepts raw feature input, applies scaling, and predicts security status.
    """
    # Convert input list or dictionary to DataFrame
    df = pd.DataFrame([sample_data])
    
    # Scale features using saved scaler
    scaled_data = scaler.transform(df)
    
    # Generate prediction and probability
    prediction = model.predict(scaled_data)[0]
    probabilities = model.predict_proba(scaled_data)[0]
    
    label = "ATTACK" if prediction == 1 else "NORMAL"
    confidence = probabilities[prediction] * 100
    
    return label, confidence

if __name__ == "__main__":
    # Test with sample normal packet data (41 features matching dataset schema)
    # Using sample values from preprocessed NSL-KDD
    sample_normal_packet = {
        'duration': 0, 'protocol_type': 1, 'service': 20, 'flag': 9, 'src_bytes': 181,
        'dst_bytes': 5450, 'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
        'num_failed_logins': 0, 'logged_in': 1, 'num_compromised': 0, 'root_shell': 0,
        'su_attempted': 0, 'num_root': 0, 'num_file_creations': 0, 'num_shells': 0,
        'num_access_files': 0, 'num_outbound_cmds': 0, 'is_host_login': 0,
        'is_guest_login': 0, 'count': 8, 'srv_count': 8, 'serror_rate': 0.0,
        'srv_serror_rate': 0.0, 'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
        'same_srv_rate': 1.0, 'diff_srv_rate': 0.0, 'srv_diff_host_rate': 0.0,
        'dst_host_count': 9, 'dst_host_srv_count': 9, 'dst_host_same_srv_rate': 1.0,
        'dst_host_diff_srv_rate': 0.0, 'dst_host_same_src_port_rate': 0.11,
        'dst_host_srv_diff_host_rate': 0.0, 'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0, 'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }

    print("\n" + "="*50)
    print("        INFERENCE TEST: REALTIME PACKET EVALUATION")
    print("="*50)
    
    result, confidence = predict_network_traffic(sample_normal_packet)
    print(f"Prediction Result : {result}")
    print(f"Confidence Score  : {confidence:.2f}%")
    print("="*50)