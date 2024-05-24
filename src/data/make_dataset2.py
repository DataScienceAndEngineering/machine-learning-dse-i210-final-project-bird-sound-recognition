import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from features import extract_mfcc_features, extract_log_mel_spectrogram, extract_waveform_features, extract_combined_features, extract_combined_features_with_waveform # type: ignore

# Load the CSV file
metadata_path = "C:/Users/19147/Downloads/archive (12)/bird_songs_metadata.csv"
audio_files_path = "C:/Users/19147/Downloads/archive (12)/wavfiles"

def create_dataset(metadata_path, audio_files_path, feature_type='mfcc'):
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

    # Load file paths and labels
    file_paths, labels = load_bird_sound_paths_from_csv(metadata_path, audio_files_path)
    processed_audio = load_and_process_audio(file_paths, sr, fixed_length)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)

    # Extract features based on user input
    if feature_type == 'mfcc':
        features = extract_mfcc_features(processed_audio, sr)
    elif feature_type == 'log_mel':
        features = [extract_log_mel_spectrogram(fp, sr) for fp in file_paths]
    elif feature_type == 'waveform':
        features = extract_waveform_features(processed_audio)
    elif feature_type == 'combined_mfcc_logmel':
        features = [extract_combined_features(fp, sr) for fp in file_paths]
    elif feature_type == 'combined_all':
        features = [extract_combined_features_with_waveform(fp, sr) for fp in file_paths]
    else:
        raise ValueError("Invalid feature type. Choose from 'mfcc', 'log_mel', 'waveform', 'combined_mfcc_logmel', 'combined_all'.")

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

print("Dataset created successfully!")

# Call the function and get the dataset
feature_type = 'mfcc'  # or 'log_mel', 'waveform', 'combined_mfcc_logmel', 'combined_all'
X_train, X_test, y_train, y_test = create_dataset(metadata_path, audio_files_path, feature_type)
