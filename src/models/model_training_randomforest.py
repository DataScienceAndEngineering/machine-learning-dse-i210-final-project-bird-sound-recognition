# -*- coding: utf-8 -*-
"""model_training Randomforest.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1VppPFNncwrEnpNM7_SSF9wzG_b49Mqn-
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
from google.colab import drive

def train_rf_model(X_train, y_train, model_path, label_encoder_path):
    # Encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train_encoded)
    joblib.dump(rf, model_path)

    # Save the label encoder
    joblib.dump(le, label_encoder_path)

def main():
    # Mount Google Drive
    drive.mount('/content/drive')

    # MFCC features
    X_train_mfcc = np.load('/content/drive/MyDrive/archive/X_train_mfcc.npy')
    y_train_mfcc = np.load('/content/drive/MyDrive/archive/y_train_mfcc.npy', allow_pickle=True)
    train_rf_model(X_train_mfcc, y_train_mfcc, '/content/drive/MyDrive/archive/rf_model_mfcc.pkl', '/content/drive/MyDrive/archive/label_encoder_mfcc.pkl')

    # Log-mel features
    X_train_log_mel = np.load('/content/drive/MyDrive/archive/X_train_log_mel.npy')
    y_train_log_mel = np.load('/content/drive/MyDrive/archive/y_train_log_mel.npy', allow_pickle=True)
    train_rf_model(X_train_log_mel, y_train_log_mel, '/content/drive/MyDrive/archive/rf_model_log_mel.pkl', '/content/drive/MyDrive/archive/label_encoder_log_mel.pkl')

    # Waveform features
    X_train_waveform = np.load('/content/drive/MyDrive/archive/X_train_waveform.npy')
    y_train_waveform = np.load('/content/drive/MyDrive/archive/y_train_waveform.npy', allow_pickle=True)
    train_rf_model(X_train_waveform, y_train_waveform, '/content/drive/MyDrive/archive/rf_model_waveform.pkl', '/content/drive/MyDrive/archive/label_encoder_waveform.pkl')

if __name__ == "__main__":
    main()