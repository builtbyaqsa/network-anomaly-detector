import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load Processed Dataset
DATA_PATH = os.path.join("data", "processed_dataset.csv")

print("--> Loading preprocessed data for model training...")
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['target'])
y = df['target']

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save Scaler for Future Inference
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, os.path.join("models", "scaler.pkl"))

# 4. Train Decision Tree Classifier
print("--> Training Decision Tree Classifier...")
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)
dt_preds = dt_model.predict(X_test_scaled)
dt_acc = accuracy_score(y_test, dt_preds)

# 5. Train Random Forest Classifier
print("--> Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)
rf_preds = rf_model.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_preds)

# Save Best Model (Random Forest)
joblib.dump(rf_model, os.path.join("models", "random_forest_model.pkl"))

print("\n[SUCCESS] Day 4 Model Training Complete!")
print(f"Decision Tree Accuracy : {dt_acc * 100:.2f}%")
print(f"Random Forest Accuracy : {rf_acc * 100:.2f}%")
print("--> Saved best performing model to 'models/random_forest_model.pkl'")