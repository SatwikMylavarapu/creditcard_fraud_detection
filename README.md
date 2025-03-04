
# **Credit Card Fraud Detection using XGBoost**

This project focuses on detecting fraudulent credit card transactions using the **XGBoost** machine learning algorithm. The dataset used is a reduced version of the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), which contains anonymized credit card transactions labeled as fraudulent or legitimate.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Setup](#setup)
4. [Usage](#usage)
5. [Training on SageMaker](#training-on-sagemaker)
6. [Troubleshooting](#troubleshooting)
7. [License](#license)

---

## **Project Overview**
The goal of this project is to build a machine learning model that can accurately classify credit card transactions as either **fraudulent (1)** or **legitimate (0)**. The model is trained using **XGBoost**, a powerful gradient boosting algorithm, and deployed on **Amazon SageMaker** for scalable and efficient training.

---

## **Dataset**
The dataset (`creditcard_reduced.csv`) contains the following features:
- **Time**: Number of seconds elapsed between the transaction and the first transaction.
- **V1-V28**: Anonymized features obtained through PCA (Principal Component Analysis).
- **Amount**: Transaction amount.
- **Class**: Target variable (`0` for legitimate, `1` for fraudulent).

### **Dataset Statistics**
- Total transactions: 284,807
- Fraudulent transactions: 492 (0.172%)
- Legitimate transactions: 284,315 (99.828%)

---

## **Setup**
To run this project, you need the following:

### **Prerequisites**
- Python 3.7+
- Libraries: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `boto3`, `sagemaker`
- AWS account with SageMaker access

### **Install Dependencies**
```bash
pip install pandas numpy xgboost scikit-learn boto3 sagemaker
```

### **Download the Dataset**
1. Download the dataset from [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud).
2. Save it as `creditcard_reduced.csv` in the project directory.



## **Usage**
### **1. Data Preprocessing**
Clean and preprocess the dataset before training:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv("creditcard_reduced.csv")

# Clean the 'Class' column
df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
df = df[df["Class"].isin([0, 1])  # Keep only 0 and 1

# Save the cleaned dataset
df.to_csv("creditcard_reduced_cleaned.csv", index=False)
```

### **2. Train the Model Locally**
Train the XGBoost model locally:
```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the cleaned dataset
df = pd.read_csv("creditcard_reduced_cleaned.csv")

# Split into features and target
X = df.drop("Class", axis=1)
y = df["Class"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
```

---

## **Training on SageMaker**
To train the model on **Amazon SageMaker**, follow these steps:

### **1. Upload Dataset to S3**
Upload the cleaned dataset (`creditcard_reduced_cleaned.csv`) to an S3 bucket:
```python
import boto3

# Upload to S3
s3 = boto3.client("s3")
bucket_name = "your-bucket-name"
s3.upload_file("creditcard_reduced_cleaned.csv", bucket_name, "train/train.csv")
```

### **2. Configure SageMaker Training Job**
Set up the SageMaker training job:
```python
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator

# SageMaker session and role
sagemaker_session = sagemaker.Session()
role = "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole"

# Define the XGBoost container
container = sagemaker.image_uris.retrieve("xgboost", sagemaker_session.boto_region_name, "1.2-1")

# Configure the estimator
estimator = Estimator(
    image_uri=container,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{bucket_name}/output",
    sagemaker_session=sagemaker_session,
)

# Define input data
train_input = TrainingInput(s3_data=f"s3://{bucket_name}/train/", content_type="csv")

# Train the model
estimator.fit({"train": train_input})
```

---

## **Troubleshooting**
### **1. XGBoostError: Label must be in [0,1] for logistic regression**
This error occurs if the **'Class' column** contains values other than `0` or `1`. To fix:
1. Clean the dataset before uploading to S3:
   ```python
   df["Class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
   df = df[df["Class"].isin([0, 1])
   ```
2. Verify the dataset after uploading to S3:
   ```python
   df_s3 = pd.read_csv(f"s3://{bucket_name}/train/train.csv")
   print(df_s3["Class"].unique())  # Should print [0, 1]
   ```

### **2. Missing Dependencies**
Ensure all required libraries are installed:
```bash
pip install pandas numpy xgboost scikit-learn boto3 sagemaker
```

### **3. SageMaker Permissions**
Ensure your IAM role has the necessary permissions for SageMaker and S3.

---

## **License**
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---
```

---

### **Steps to Use This README**
1. Open a text editor (e.g., VS Code, Notepad++, or any Markdown editor).
2. Create a new file and name it `README.md`.
3. Copy-paste the above raw Markdown content into the file.
4. Save the file.

---

### **Viewing the README**
To view the formatted README:
- Use a Markdown viewer (e.g., GitHub, VS Code with Markdown Preview, or an online Markdown editor like [Dillinger](https://dillinger.io/)).
- Open the `README.md` file in the viewer, and it will render with proper formatting.

---

Let me know if you encounter any issues! 🚀
