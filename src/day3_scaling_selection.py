import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# 1. Load Processed Dataset
DATA_PATH = os.path.join("data", "processed_dataset.csv")

print("--> Loading processed dataset for scaling and selection...")
df = pd.read_csv(DATA_PATH)

# Separate Features (X) and Target (y)
X = df.drop(columns=['target'])
y = df['target']

# 2. Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Feature Selection using Random Forest Importance
print("--> Calculating feature importance...")
rf = RandomForestClassifier(n_estimators=50, random_state=42)
rf.fit(X_train_scaled, y_train)

# Get top 15 most important features
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n[SUCCESS] Top 10 Most Important Network Features:")
print(feature_importance_df.head(10).to_string(index=False))

# Save top selected feature names for model training
top_features = feature_importance_df.head(15)['Feature'].tolist()
print(f"\n--> Selected Top 15 Features for Model Training:\n{top_features}")