import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

def plot_mel_spectrogram(audio_path, title, low_confidence=False):
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8192)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', fmax=8192)
    plt.colorbar(format='%+2.0f dB')
    if low_confidence:
        plt.title(f'Low Confidence: Mel Spectrogram of {title}')
    else:
        plt.title(f'Mel Spectrogram of {title}')
    plt.tight_layout()
    plt.show()

def main():
    audio_files = [
        ('/content/wavfiles/109034-2.wav', 'wavfiles/109034-2.wav', False),
        ('/content/wavfiles/173150-0.wav', 'wavfiles/173150-0.wav', False),
        ('/content/wavfiles/358837-5.wav', 'wavfiles/358837-5.wav', True),
        ('/content/wavfiles/363141-2.wav', 'wavfiles/363141-2.wav', True)
    ]

    for file_path, title, low_confidence in audio_files:
        plot_mel_spectrogram(file_path, title, low_confidence)

if __name__ == "__main__":
    main()
