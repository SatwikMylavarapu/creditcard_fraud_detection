import streamlit as st
import boto3
import numpy as np
from sagemaker.predictor import Predictor
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# 🚀 **Set Your SageMaker Endpoint Name**
endpoint_name = "sagemaker-xgboost-have your own endpoint"

# ✅ **Initialize SageMaker Predictor**
predictor = Predictor(endpoint_name=endpoint_name, 
                      serializer=CSVSerializer(), 
                      deserializer=JSONDeserializer())

# 🎨 **Streamlit UI**
st.title("🚀 Fraud Detection Model")
st.write("Enter transaction details to check if it's fraudulent!")

# **Input Fields for Transaction Features**
features = []
for i in range(30):
    features.append(st.number_input(f"Feature {i+1}", value=0.0))

# **Predict Button**
if st.button("🔍 Predict Fraud"):
    csv_payload = ",".join(map(str, features))
    try:
        prediction = predictor.predict(csv_payload)
        score = prediction['predictions'][0]['score']

        # **Display result**
        if score > 0.5:
            st.error(f"🚨 Fraud Alert! Risk Score: {score:.6f}")
        else:
            st.success(f"✅ Transaction is Safe! Risk Score: {score:.6f}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
