Autoencoder-Anomaly-Detection 🚨

This project is a lightweight and interactive anomaly detection app built with Gradio and a simple Autoencoder neural network. It detects anomalies based on two-dimensional input data by learning what “normal” data looks like and flagging anything that deviates significantly as an anomaly.

What is it?

An autoencoder is an unsupervised neural network that tries to reconstruct its input. When it can’t reconstruct well, it means the input might be unusual or an anomaly. This app takes two numeric inputs (X1 and X2), processes them through the trained autoencoder, and then predicts if the input is “Normal” or an “Anomaly.”
