import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# ============================
# 📌 Step 1: Load & Clean Data
# ============================

print("🚀 Loading dataset locally...")
df = pd.read_csv("/Users/satwikmylavarapu/Downloads/creditcard.csv")

# ✅ Ensure 'Class' column is strictly 0 or 1
df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
df = df[df["Class"].isin([0, 1])]

# ✅ Drop duplicate rows
df = df.drop_duplicates()

# ✅ Remove non-numeric columns (if dataset has any)
df = df.select_dtypes(include=[np.number])

# ✅ Check for missing values
if df.isnull().sum().sum() > 0:
    df = df.dropna()  # Remove rows with NaN values

print("✅ Data Loaded & Cleaned!")
print(df["Class"].value_counts())  # Check fraud vs. non-fraud balance
print(df.info())  # Verify dataset types

# ============================
# 📌 Step 2: Train-Test Split
# ============================

print("🚀 Splitting dataset into Train/Test...")

X = df.drop(columns=["Class"])  # Features
y = df["Class"]  # Target

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"✅ Train Set: {X_train.shape}, Test Set: {X_test.shape}")

# ============================
# 📌 Step 3: Feature Scaling
# ============================

print("🚀 Applying Feature Scaling...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Feature Scaling Applied!")

# ============================
# 📌 Step 4: Convert Data to XGBoost DMatrix
# ============================

print("🚀 Converting Data to XGBoost Format...")

# Convert to DMatrix format (XGBoost-specific format)
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

print("✅ Data Converted for XGBoost!")

# ============================
# 📌 Step 5: Train XGBoost Model
# ============================

params = {
    "max_depth": 5,
    "eta": 0.2,
    "objective": "binary:logistic",
    "eval_metric": "auc"
}

print("🚀 Training XGBoost Model Locally...")

model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "Test")])

print("✅ Model Training Completed!")

# ============================
# 📌 Step 6: Make Predictions
# ============================

print("🚀 Making Predictions...")

# Predict fraud probabilities
y_pred_prob = model.predict(dtest)

# Convert probabilities to 0 or 1 (Threshold = 0.5)
y_pred_binary = [1 if p > 0.5 else 0 for p in y_pred_prob]

print("✅ Predictions Completed!")

# ============================
# 📌 Step 7: Evaluate Model Performance
# ============================

print("🚀 Evaluating Model Performance...")

accuracy = accuracy_score(y_test, y_pred_binary)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"✅ Model Accuracy: {accuracy:.4f}")
print(f"✅ Model AUC Score: {roc_auc:.4f}")

print("\n🔍 Classification Report:\n", classification_report(y_test, y_pred_binary))

# ============================
# 📌 Step 8: Save Model for Future Use
# ============================

print("🚀 Saving Trained Model...")

model.save_model("xgboost_fraud_model.json")

print("✅ Model Saved as xgboost_fraud_model.json!")
