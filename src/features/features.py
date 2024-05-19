import numpy as np
import librosa
from joblib import Parallel, delayed

# Function to extract MFCC features
def extract_mfcc_features(audio_clips, sr):
    mfcc_features = []
    for sound in audio_clips:
        mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_features.append(mfcc_scaled.flatten())
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
