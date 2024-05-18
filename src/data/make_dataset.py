# make_dataset.py

import pandas as pd
import numpy as np
import librosa
import os
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
metadata_path = "C:/Users/19147/Downloads/archive (12)/bird_songs_metadata.csv"
audio_files_path = "C:/Users/19147/Downloads/archive (12)/wavfiles"
df = pd.read_csv(metadata_path)

# Display the first few rows of the dataframe
print(df.head())

# Distribution of species
species_dist = df['species'].value_counts()
print("Distribution of species:")
print(species_dist)


fixed_length = 5 * 22050  # 5 seconds multiplied by sampling rate
sr = 22050  # Default sampling rate from librosa

# Function to load bird sound file paths and labels from CSV
def load_bird_sound_paths_from_csv(metadata_path, audio_files_path):
    df = pd.read_csv(metadata_path)
    file_paths = []
    labels = []

    for index, row in df.iterrows():
        file_path = os.path.join(audio_files_path, row['filename'])
        if os.path.exists(file_path):
            file_paths.append(file_path)
            labels.append(row['species'])  # 'species' holds the label
        else:
            print(f"File not found: {file_path}")

    return file_paths, labels

# Function to load and process audio files from paths
def load_and_process_audio(file_paths, sr, fixed_length):
    processed_audio = []
    for file_path in file_paths:
        sound, _ = librosa.load(file_path, sr=sr)
        if len(sound) > fixed_length:
            sound = sound[:fixed_length]
        else:
            padding = fixed_length - len(sound)
            sound = np.pad(sound, (0, padding), 'constant')
        processed_audio.append(sound)
    return processed_audio

# Function to extract MFCC features
def extract_mfcc_features(audio_clips, sr):
    mfcc_features = []
    for sound in audio_clips:
        mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_features.append(mfcc_scaled)
    return np.array(mfcc_features)

# Function to extract log mel spectrogram
def extract_log_mel_spectrogram(file_path, sr, n_fft=2048, hop_length=512, n_mels=128):
    audio, sr = librosa.load(file_path, sr=sr)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    return S_DB.flatten()

# Function to extract waveform features
def extract_waveform_features(audio_clips):
    def process_sound(sound):
        return np.array([np.mean(sound), np.std(sound), np.max(sound), np.min(sound)])

    waveform_features = Parallel(n_jobs=-1)(delayed(process_sound)(sound) for sound in audio_clips)
    return np.array(waveform_features)

# Function to extract combined features (MFCC + Log Mel Spectrogram)
def extract_combined_features(file_path, sr):
    audio, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec)
    log_mel_spec_mean = np.mean(log_mel_spec, axis=1)
    combined_features = np.concatenate((mfcc_mean, log_mel_spec_mean)).flatten()
    return combined_features

# Function to extract combined features (MFCC + Log Mel Spectrogram + Waveform)
def extract_combined_features_with_waveform(file_path, sr):
    audio, sr = librosa.load(file_path, sr=sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec)
    log_mel_spec_mean = np.mean(log_mel_spec, axis=1)
    waveform = np.array([np.mean(audio), np.std(audio), np.max(audio), np.min(audio)])
    combined_features = np.concatenate((mfcc_mean, log_mel_spec_mean, waveform)).flatten()
    return combined_features

# Load file paths and labels
file_paths, labels = load_bird_sound_paths_from_csv(metadata_path, audio_files_path)
processed_audio = load_and_process_audio(file_paths, sr, fixed_length)
mfcc_features = extract_mfcc_features(processed_audio, sr)

# Extract log mel spectrogram features
log_mel_features = [extract_log_mel_spectrogram(fp, sr) for fp in file_paths]

# Extract waveform features
waveform_features = extract_waveform_features(processed_audio)

# Extract combined features (MFCC + Log Mel Spectrogram)
combined_features_mfcc_logmel = [extract_combined_features(fp, sr) for fp in file_paths]

# Extract combined features (MFCC + Log Mel Spectrogram + Waveform)
combined_features_all = [extract_combined_features_with_waveform(fp, sr) for fp in file_paths]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
X_train_mfcc, X_test_mfcc, y_train, y_test, files_train_mfcc, files_test_mfcc = train_test_split(
    mfcc_features, y_encoded, file_paths, test_size=0.2, random_state=42)
X_train_log_mel, X_test_log_mel, y_train, y_test, files_train_log_mel, files_test_log_mel = train_test_split(
    log_mel_features, y_encoded, file_paths, test_size=0.2, random_state=42)
X_train_waveform, X_test_waveform, y_train, y_test, files_train_waveform, files_test_waveform = train_test_split(
    waveform_features, y_encoded, file_paths, test_size=0.2, random_state=42)
X_train_combined_mfcc_logmel, X_test_combined_mfcc_logmel, y_train, y_test, files_train_combined_mfcc_logmel, files_test_combined_mfcc_logmel = train_test_split(
    combined_features_mfcc_logmel, y_encoded, file_paths, test_size=0.2, random_state=42)
X_train_combined_all, X_test_combined_all, y_train, y_test, files_train_combined_all, files_test_combined_all = train_test_split(
    combined_features_all, y_encoded, file_paths, test_size=0.2, random_state=42)

# Save features, labels, and file paths to files
np.save('X_train_mfcc.npy', X_train_mfcc)
np.save('X_test_mfcc.npy', X_test_mfcc)
np.save('X_train_log_mel.npy', X_train_log_mel)
np.save('X_test_log_mel.npy', X_test_log_mel)
np.save('X_train_waveform.npy', X_train_waveform)
np.save('X_test_waveform.npy', X_test_waveform)
np.save('X_train_combined_mfcc_logmel.npy', X_train_combined_mfcc_logmel)
np.save('X_test_combined_mfcc_logmel.npy', X_test_combined_mfcc_logmel)
np.save('X_train_combined_all.npy', X_train_combined_all)
np.save('X_test_combined_all.npy', X_test_combined_all)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
np.save('files_train_mfcc.npy', files_train_mfcc)
np.save('files_test_mfcc.npy', files_test_mfcc)
np.save('files_train_log_mel.npy', files_train_log_mel)
np.save('files_test_log_mel.npy', files_test_log_mel)
np.save('files_train_waveform.npy', files_train_waveform)
np.save('files_test_waveform.npy', files_test_waveform)
np.save('files_train_combined_mfcc_logmel.npy', files_train_combined_mfcc_logmel)
np.save('files_test_combined_mfcc_logmel.npy', files_test_combined_mfcc_logmel)
np.save('files_train_combined_all.npy', files_train_combined_all)
np.save('files_test_combined_all.npy', files_test_combined_all)

print("Dataset created and saved successfully!")
