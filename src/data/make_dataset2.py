import pandas as pd
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from features import extract_mfcc_features, extract_log_mel_spectrogram, extract_waveform_features, extract_combined_features, extract_combined_features_with_waveform

# Load the CSV file
metadata_path = "C:/Users/19147/Downloads/archive (12)/bird_songs_metadata.csv"
audio_files_path = "C:/Users/19147/Downloads/archive (12)/wavfiles"

def create_dataset(metadata_path, audio_files_path):
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
    X_train_mfcc, X_test_mfcc, y_train, y_test = train_test_split(mfcc_features, y_encoded, test_size=0.2, random_state=42)
    X_train_log_mel, X_test_log_mel, y_train, y_test = train_test_split(log_mel_features, y_encoded, test_size=0.2, random_state=42)
    X_train_waveform, X_test_waveform, y_train, y_test = train_test_split(waveform_features, y_encoded, test_size=0.2, random_state=42)
    X_train_combined_mfcc_logmel, X_test_combined_mfcc_logmel, y_train, y_test = train_test_split(combined_features_mfcc_logmel, y_encoded, test_size=0.2, random_state=42)
    X_train_combined_all, X_test_combined_all, y_train, y_test = train_test_split(combined_features_all, y_encoded, test_size=0.2, random_state=42)

    return {
        'X_train_mfcc': X_train_mfcc, 'X_test_mfcc': X_test_mfcc,
        'X_train_log_mel': X_train_log_mel, 'X_test_log_mel': X_test_log_mel,
        'X_train_waveform': X_train_waveform, 'X_test_waveform': X_test_waveform,
        'X_train_combined_mfcc_logmel': X_train_combined_mfcc_logmel, 'X_test_combined_mfcc_logmel': X_test_combined_mfcc_logmel,
        'X_train_combined_all': X_train_combined_all, 'X_test_combined_all': X_test_combined_all,
        'y_train': y_train, 'y_test': y_test
    }

print("Dataset created successfully!")

# Call the function and get the dataset
dataset = create_dataset(metadata_path, audio_files_path)
