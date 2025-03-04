# 🚀 Credit Card Fraud Detection using XGBoost

This project focuses on detecting fraudulent credit card transactions using the **XGBoost** machine learning algorithm. The dataset used is a reduced version of the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains anonymized credit card transactions labeled as fraudulent or legitimate.

---

## 📌 Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Setup](#setup)
4. [Local Model Training](#local-model-training)
5. [Training on SageMaker](#training-on-sagemaker)
6. [Deploying the Model with Streamlit](#deploying-the-model-with-streamlit)
7. [Troubleshooting](#troubleshooting)
8. [License](#license)

---

## 🔍 Project Overview
The goal of this project is to build a **machine learning model** that accurately classifies credit card transactions as either **fraudulent (1)** or **legitimate (0)**.  

The model is trained using **XGBoost**, a powerful gradient boosting algorithm, and deployed in **Amazon SageMaker** for scalable and efficient training. Additionally, a **Streamlit web app** is created to interact with the deployed model for real-time predictions.

---

## 📊 Dataset
The dataset (`creditcard_reduced.csv`) contains the following features:
- **Time**: Number of seconds elapsed between the transaction and the first transaction.
- **V1-V28**: Anonymized features obtained through PCA (Principal Component Analysis).
- **Amount**: Transaction amount.
- **Class**: Target variable (`0` for legitimate, `1` for fraudulent).

### 🏷 Dataset Statistics
- **Total transactions**: 284,807  
- **Fraudulent transactions**: 492 (0.172%)  
- **Legitimate transactions**: 284,315 (99.828%)  

---

## ⚙️ Setup
To run this project, install the required dependencies:

### 📌 Prerequisites
- Python 3.7+
- Required libraries: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `boto3`, `sagemaker`, `streamlit`
- AWS account with SageMaker and S3 access

### 📥 Install Dependencies
```bash
pip install pandas numpy xgboost scikit-learn boto3 sagemaker streamlit
```
### 📥 Download the Dataset
1. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
2. Save it as `creditcard_reduced.csv` in the project directory.

---

## 🏗 Local Model Training

### 1️⃣ **Data Preprocessing**
Clean and preprocess the dataset before training:
```python
import pandas as pd

# Load the dataset
df = pd.read_csv("creditcard_reduced.csv")

# Ensure the 'Class' column is properly formatted
df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
df = df[df["Class"].isin([0, 1])]

# Save the cleaned dataset
df.to_csv("creditcard_reduced_cleaned.csv", index=False)
```

### 2️⃣ **Train the XGBoost Model Locally**
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("creditcard_reduced_cleaned.csv")

# Split into features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
```

---

## 🚀 Training on SageMaker

### 1️⃣ **Upload Dataset to S3**
```python
import boto3

s3 = boto3.client("s3")
bucket_name = "fraud-detection-data-project"
s3.upload_file("creditcard_reduced_cleaned.csv", bucket_name, "train/train.csv")
```

### 2️⃣ **Train Model in SageMaker**
```python
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"

container = sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, "1.2-1")

estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket_name}/output",
    sagemaker_session=sagemaker_session,
)

train_input = TrainingInput(s3_data=f"s3://{bucket_name}/train/", content_type="csv")

estimator.fit({"train": train_input})
```

---

## 📡 Deploying the Model with Streamlit

### 1️⃣ **Deploy Model in SageMaker (Real-Time API)**
```python
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

endpoint_name = "sagemaker-xgboost-endpoint"

predictor = Predictor(endpoint_name, serializer=CSVSerializer(), deserializer=JSONDeserializer())
```

### 2️⃣ **Streamlit Web App for Fraud Detection**
Create a `app.py` file with the following:
```python
import streamlit as st
import numpy as np
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

endpoint_name = "sagemaker-xgboost-endpoint"
predictor = Predictor(endpoint_name, serializer=CSVSerializer(), deserializer=JSONDeserializer())

st.title("🚀 Fraud Detection Model")
st.write("Enter transaction details to check if it's fraudulent!")

features = [st.number_input(f"Feature {i+1}", value=0.0) for i in range(30)]

if st.button("🔍 Predict Fraud"):
    csv_payload = ",".join(map(str, features))
    prediction = predictor.predict(csv_payload)
    score = prediction['predictions'][0]['score']

    if score > 0.5:
        st.error(f"🚨 Fraud Alert! Risk Score: {score:.6f}")
    else:
        st.success(f"✅ Transaction is Safe! Risk Score: {score:.6f}")
```

### 3️⃣ **Run Streamlit Locally**
```bash
streamlit run app.py
```

---

## 🔧 Troubleshooting

### ❌ XGBoostError: Label must be in [0,1]
Ensure `Class` column is properly formatted:
```python
df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
df = df[df["Class"].isin([0, 1])]
```

### ❌ SageMaker Model Not Deploying
Ensure your IAM role has correct SageMaker and S3 permissions.

### ❌ Streamlit App Not Running
Ensure `streamlit` and `boto3` are installed:
```bash
pip install streamlit boto3
```

---

## 📜 License
This project is licensed under the **MIT License**.

---
