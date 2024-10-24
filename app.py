import streamlit as st
import pandas as pd
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Title of the dashboard
st.title("Predictive Maintenance Model - Demo with Numerical Features")

# Demo dataset with numerical data (temperature, vibration)
def load_demo_data():
    data = {
        'text': [
            'Motor shows abnormal vibration',
            'Temperature is higher than expected',
            'Routine maintenance required',
            'Oil leakage detected',
            'Unexpected shutdown',
            'Performance degradation observed',
            'Fan making unusual noise',
            'Battery needs replacement',
            'No issue detected during inspection',
            'Filter clogging detected'
        ],
        'temperature': [85, 95, 70, 90, 100, 92, 88, 65, 72, 86],
        'vibration': [3.5, 4.8, 1.2, 5.0, 5.5, 4.0, 3.8, 1.0, 0.5, 4.3],
        'label': [
            'Failure', 'Warning', 'Routine', 'Failure', 'Failure',
            'Warning', 'Warning', 'Routine', 'Routine', 'Warning'
        ]
    }
    return pd.DataFrame(data)

# Load demo data
data = load_demo_data()
st.write("Demo Data Preview:")
st.write(data)

# Encode labels
label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])

# Standardize numerical data (temperature, vibration)
scaler = StandardScaler()
numerical_data = scaler.fit_transform(data[['temperature', 'vibration']])

# Tokenizer and model setup for text
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
text_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(data['label'].unique()))

# Define a simple classifier that combines numerical features and text embeddings
class PredictiveMaintenanceModel(nn.Module):
    def __init__(self, text_model, num_numerical_features):
        super(PredictiveMaintenanceModel, self).__init__()
        self.text_model = text_model
        self.fc = nn.Linear(num_numerical_features + self.text_model.config.hidden_size, len(data['label'].unique()))  # combine text + numerical

    def forward(self, text_inputs, numerical_inputs):
        text_outputs = self.text_model(**text_inputs).last_hidden_state[:, 0, :]  # Extract [CLS] token
        combined_inputs = torch.cat((text_outputs, numerical_inputs), dim=1)
        logits = self.fc(combined_inputs)
        return logits

# Instantiate the model
model = PredictiveMaintenanceModel(text_model, num_numerical_features=2)

# Tokenize text data
def tokenize_data(text_list):
    return tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")

# Tokenize the text data
text_inputs = tokenize_data(data['text'].tolist())

# Convert numerical data to tensors
numerical_inputs = torch.tensor(numerical_data, dtype=torch.float32)

# Predict button
if st.button("Run Prediction"):
    with torch.no_grad():
        outputs = model(text_inputs, numerical_inputs)
        predictions = torch.argmax(outputs, dim=1).numpy()

    # Decode predictions
    decoded_predictions = label_encoder.inverse_transform(predictions)
    data['Predictions'] = decoded_predictions
    st.write("Prediction Results:")
    st.write(data[['text', 'temperature', 'vibration', 'label', 'Predictions']])

