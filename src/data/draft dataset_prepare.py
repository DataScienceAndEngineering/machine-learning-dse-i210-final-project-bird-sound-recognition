import numpy as np
import librosa
from joblib import Parallel, delayed
import os
from pathlib import Path
from google.colab import drive
import pandas as pd
from sklearn.model_selection import train_test_split
import zipfile

def read_metadata(metadata_path):
    return pd.read_csv(metadata_path)

def get_file_paths_and_labels(df, zip_ref):
    file_paths = []
    labels = []
    for filename, species in df[['filename', 'species']].itertuples(index=False):
        file_path = 'wavfiles/' + filename
        if file_path in zip_ref.namelist():
            file_paths.append(file_path)
            labels.append(species)
        else:
            print(f"File not found in zip: {file_path}")
    return file_paths, labels

def load_and_process_audio(zip_ref, file_paths, sr, fixed_length):
    processed_audio = []
    for file_path in file_paths:
        try:
            with zip_ref.open(file_path) as file:
                audio, _ = librosa.load(file, sr=sr)
                if len(audio) > fixed_length:
                    audio = audio[:fixed_length]
                else:
                    audio = np.pad(audio, (0, max(0, fixed_length - len(audio))), 'constant')
                processed_audio.append(audio)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    return processed_audio

def extract_mfcc_features(audio_clips, sr, n_mfcc=13):
    def process_sound(sound):
        mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=n_mfcc)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled

    mfcc_features = Parallel(n_jobs=-1)(delayed(process_sound)(sound) for sound in audio_clips)
    return np.array(mfcc_features)

def extract_log_mel_features(audio_clips, sr, n_mels=128):
    def process_sound(sound):
        mel = librosa.feature.melspectrogram(y=sound, sr=sr, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_scaled = np.mean(mel_db.T, axis=0)
        return mel_scaled

    mel_features = Parallel(n_jobs=-1)(delayed(process_sound)(sound) for sound in audio_clips)
    return np.array(mel_features)

def extract_waveform_features(audio_clips):
    def process_sound(sound):
        return np.mean(sound), np.std(sound), np.max(sound), np.min(sound)

    waveform_features = Parallel(n_jobs=-1)(delayed(process_sound)(sound) for sound in audio_clips)
    return np.array(waveform_features)

def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    return (features - mean) / std

def split_data(features, labels, test_size=0.2, random_state=42):
    return train_test_split(features, labels, test_size=test_size, random_state=random_state)

def main():
    drive.mount('/content/drive')

    metadata_path = '/content/drive/MyDrive/Bird sound classification data/bird_songs_metadata.csv'
    zip_path = '/content/drive/MyDrive/Bird sound classification data/archive (1).zip'
    sample_rate = 22050
    fixed_length = sample_rate * 5  # 5 seconds

    df = read_metadata(metadata_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_paths, labels = get_file_paths_and_labels(df, zip_ref)
        processed_audio_clips = load_and_process_audio(zip_ref, file_paths, sample_rate, fixed_length)

    if not processed_audio_clips:
        print("No audio files processed. Exiting.")
        return

    mfcc_features = extract_mfcc_features(processed_audio_clips, sample_rate)
    log_mel_features = extract_log_mel_features(processed_audio_clips, sample_rate)
    waveform_features = extract_waveform_features(processed_audio_clips)

    normalized_mfcc_features = normalize_features(mfcc_features)
    normalized_log_mel_features = normalize_features(log_mel_features)
    normalized_waveform_features = normalize_features(waveform_features)

    # Ensure all features are 2D
    if normalized_mfcc_features.ndim == 1:
        normalized_mfcc_features = normalized_mfcc_features[:, np.newaxis]
    if normalized_log_mel_features.ndim == 1:
        normalized_log_mel_features = normalized_log_mel_features[:, np.newaxis]
    if normalized_waveform_features.ndim == 1:
        normalized_waveform_features = normalized_waveform_features[:, np.newaxis]

    combined_features = np.concatenate([normalized_mfcc_features, normalized_log_mel_features, normalized_waveform_features], axis=1)
    X_train, X_test, y_train, y_test = split_data(combined_features, labels)

    np.save('/content/drive/MyDrive/X_train.npy', X_train)
    np.save('/content/drive/MyDrive/X_test.npy', X_test)
    np.save('/content/drive/MyDrive/y_train.npy', y_train)
    np.save('/content/drive/MyDrive/y_test.npy', y_test)

if __name__ == "__main__":
    main()
