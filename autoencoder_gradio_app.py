# Autoencoder-Anomaly-Detection

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import gradio as gr

# 1. Create synthetic dataset
np.random.seed(1)
normal_data = np.random.normal(loc=0.5, scale=0.1, size=(1000, 3))
anomalies = np.random.uniform(low=0.9, high=1.2, size=(20, 3))
data = np.vstack([normal_data, anomalies])

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# 2. Build and train autoencoder on normal data only
input_dim = data_scaled.shape[1]
autoencoder = models.Sequential([
    layers.Dense(2, activation='relu', input_shape=(input_dim,)),
    layers.Dense(input_dim, activation='sigmoid')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(data_scaled[:1000], data_scaled[:1000], epochs=20, verbose=0)

# 3. Calculate reconstruction errors
recon = autoencoder.predict(data_scaled)
errors = np.mean(np.square(data_scaled - recon), axis=1)
threshold = np.mean(errors[:1000]) + 2*np.std(errors[:1000])

# 4. Define Gradio interface
def detect_anomaly(v1, v2, v3):
    sample = np.array([[v1, v2, v3]])
    sample_scaled = scaler.transform(sample)
    recon = autoencoder.predict(sample_scaled)
    error = np.mean((sample_scaled - recon) ** 2)
    result = "🚨 Anomaly" if error > threshold else "✅ Normal"
    return f"{result}\nReconstruction Error: {error:.4f}"

# Launch Gradio app
iface = gr.Interface(
    fn=detect_anomaly,
    inputs=[gr.Number(label="Sensor 1"), gr.Number(label="Sensor 2"), gr.Number(label="Sensor 3")],
    outputs="text",
    title="Autoencoder-Based Anomaly Detector",
    description="Enter 3 sensor values (e.g., CPU temp, voltage, fan speed)."
)

iface.launch(share=True)
