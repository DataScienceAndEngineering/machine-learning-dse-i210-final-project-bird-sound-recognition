#!/usr/bin/env python
# coding: utf-8

# In[15]:


#Importing necessary libraries
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import librosa
import os
import shap
import librosa.display
from skimage.io import imread
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_curve, auc


# In[16]:


import pandas as pd

# Load the CSV file
metadata_path = "C:/Users/19147/Downloads/archive (12)/bird_songs_metadata.csv"
df = pd.read_csv(metadata_path)

# Display the first few rows of the dataframe 
df.head()


# In[17]:


#distribution of species
species_dist = df['species'].value_counts()
print(species_dist)


# In[18]:


# Checking for missing values in each column
missing_values = df.isnull().sum()
print(missing_values)


# The primary input for generating spectrograms is the audio file itself. Missing values in metadata (like subspecies, latitude, longitude, altitude, and remarks) won't affect the generation of spectrograms, as long as the audio files (filename) are intact and not missing. 

# In[19]:


print(df.info())


# **Feature Extraction**

# In[20]:


def load_bird_sound_paths_from_csv(metadata_path, audio_files_path):
    df = pd.read_csv(metadata_path)
    file_paths = []
    labels = []

    for index, row in df.iterrows():
        file_path = os.path.join(audio_files_path, row['filename'])
        if os.path.exists(file_path):
            file_paths.append(file_path)
            labels.append(row['species'])  
        else:
            print(f"File not found: {file_path}")

    return file_paths, labels

# Usage
metadata_path = "C:/Users/19147/Downloads/archive (12)/bird_songs_metadata.csv"
audio_files_path =  "C:/Users/19147/Downloads/archive (12)/wavfiles"

file_paths, labels = load_bird_sound_paths_from_csv(metadata_path, audio_files_path)
print(file_paths[:5], labels[:5])


# In[21]:


# New function to load and process audio files from paths
def load_and_process_audio(file_paths, sr, fixed_length):
    processed_audio = []
    for file_path in file_paths:
        sound, _ = librosa.load(file_path, sr=None)
        if len(sound) > fixed_length:
            sound = sound[:fixed_length]
        else:
            padding = fixed_length - len(sound)
            sound = np.pad(sound, (0, padding), 'constant')
        processed_audio.append(sound)
    return processed_audio



# In[22]:


# Variables
sr = 22050  # Default sampling rate from librosa
fixed_length = 5 * sr  # 5 seconds multiplied by sampling rate

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

def extract_log_mel_spectrogram(file_path, sr, n_fft=2048, hop_length=512, n_mels=128):
    # Load audio file
    audio, sr = librosa.load(file_path,sr=None)
    
    # Calculate Mel spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)  # Convert the Mel spectrogram to logarithmic scale (decibels)
    
    return S_DB  # Return the log Mel spectrogram data directly


log_mel_spectrogram = extract_log_mel_spectrogram(file_paths[0], sr)


# In[23]:


log_mel_features_list = []
# Loop through each file path and extract features
for file_path in file_paths:
    log_mel_features = extract_log_mel_spectrogram(file_path,sr)
    log_mel_features_list.append(log_mel_features)
# Convert lists to numpy arrays for further processing 
log_mel_features_array = np.array(log_mel_features_list)


# In[24]:


# Flatten each Mel spectrogram before adding to the list
log_mel_features_list = [mel_feature.flatten() for mel_feature in log_mel_features_list]
X1_train, X1_test, y1_train, y1_test, files1_train, files1_test = train_test_split(
    log_mel_features_list, y_encoded, file_paths, test_size=0.2, random_state=42
)



# In[27]:


scaler = StandardScaler()
X1_train_scaled = scaler.fit_transform(X1_train)
X1_test_scaled = scaler.transform(X1_test)


# In[28]:


clf_log_Mel = RandomForestClassifier(random_state=42)
clf_log_Mel.fit(X1_train_scaled, y1_train)
clf_log_Mel_predictions = clf_log_Mel.predict(X1_test_scaled)

# Print accuracy and classification report
print(f"Accuracy with Mel features: {accuracy_score(y1_test, clf_log_Mel_predictions)}")
print(f"Classification Report: \n{classification_report(y1_test, clf_log_Mel_predictions, target_names=label_encoder.classes_)}\n")


# In[29]:


cm_Mel = confusion_matrix(y1_test, clf_log_Mel_predictions)

# Plotting using seaborn 
plt.figure(figsize=(10, 7))
sns.heatmap(cm_Mel, annot=True, fmt='g', cmap='viridis', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix(Log Mel)')
plt.show()
print(cm_Mel)


# In[32]:


log_Mel_prob = clf_log_Mel.predict_proba(X1_test)
max_probs = np.max(log_Mel_prob, axis=1)
top_10_indices_log_Mel = np.argsort(-max_probs)[:10]  
print(top_10_indices_log_Mel )


# In[33]:


def plot_log_mel_spectrogram(spectrogram, title, sr=22050):
    print("Spectrogram shape:", spectrogram.shape)  
    if spectrogram.ndim != 2:
        raise ValueError("Spectrogram must be a 2D array.")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, x_axis='time', y_axis='mel', hop_length=512)
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Iterate over the top 10 indices and plot the log Mel spectrogram for each
for index in top_10_indices_log_Mel:
    log_mel_spectrogram = X1_test[index]  
    if log_mel_spectrogram.ndim == 1:  
        num_mels = 128  
        time_steps = len(log_mel_spectrogram) // num_mels
        log_mel_spectrogram = log_mel_spectrogram.reshape(num_mels, time_steps)
    file_path = files1_test[index]
    plot_title = f'Log Mel Spectrogram for Top Prediction: {file_path}'
    plot_log_mel_spectrogram(log_mel_spectrogram, plot_title)


# In[36]:


bottom_10_indices_log_Mel = np.argsort(max_probs)[:10]  
print(bottom_10_indices_log_Mel)


# In[37]:


# Iterate over the bottom 10 indices and plot the log Mel spectrogram for each
for index in bottom_10_indices_log_Mel:
    log_mel_spectrogram = X1_test[index]
    if log_mel_spectrogram.ndim == 1:
        num_mels = 128
        time_steps = len(log_mel_spectrogram) // num_mels
        log_mel_spectrogram = log_mel_spectrogram.reshape(num_mels, time_steps)
    file_path = files1_test[index]
    plot_title = f'Log Mel Spectrogram for Bottom Prediction: {file_path}'
    plot_log_mel_spectrogram(log_mel_spectrogram, plot_title)


# In[38]:


incorrect_indices_log_Mel = np.where(clf_log_Mel_predictions != y1_test)[0]
print(incorrect_indices_log_Mel)


# In[39]:


number_to_display = 5  

for index in incorrect_indices_log_Mel[:number_to_display]:
    log_mel_spectrogram = X1_test[index]
    
    # Reshape the flattened spectrogram if necessary
    if log_mel_spectrogram.ndim == 1:
        num_mels = 128
        # Ensure the total length of the array is divisible by num_mels
        time_steps = len(log_mel_spectrogram) // num_mels
        log_mel_spectrogram = log_mel_spectrogram.reshape(num_mels, time_steps)

    file_path = files1_test[index]
    plot_title = f'Log Mel Spectrogram for Incorrect Prediction: {file_path}'
    plot_log_mel_spectrogram(log_mel_spectrogram, plot_title)


# In[40]:


explainer = shap.TreeExplainer(clf_log_Mel)
shap_values = explainer.shap_values(X1_test_scaled)

shap.summary_plot(shap_values, X1_test_scaled)


# In[41]:


explainer = shap.TreeExplainer(clf_log_Mel)
shap_values = explainer.shap_values(X1_train_scaled)

shap.summary_plot(shap_values, X1_train_scaled)

