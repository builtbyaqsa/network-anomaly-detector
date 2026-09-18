import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Processed Dataset and Saved Artifacts
DATA_PATH = os.path.join("data", "processed_dataset.csv")
MODEL_PATH = os.path.join("models", "random_forest_model.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

print("--> Loading dataset and trained model artifacts...")
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

X = df.drop(columns=['target'])
y = df['target']

# 2. Train-Test Split (Same seed used in training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Transform Test Data
X_test_scaled = scaler.transform(X_test)

# 4. Generate Predictions
print("--> Evaluating Random Forest Classifier...")
y_pred = model.predict(X_test_scaled)

# 5. Calculate Security Metrics
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Normal (0)', 'Attack (1)'])

print("\n" + "="*50)
print("             SECURITY MODEL EVALUATION REPORT")
print("="*50)
print("\n--> Confusion Matrix:")
print(f"True Negatives  (Normal correctly classified) : {cm[0][0]}")
print(f"False Positives (Normal misclassified as Attack): {cm[0][1]}")
print(f"False Negatives (Attack missed by system)      : {cm[1][0]}")
print(f"True Positives  (Attack correctly detected)   : {cm[1][1]}")

print("\n--> Detailed Classification Report:")
print(report)
print("="*50)