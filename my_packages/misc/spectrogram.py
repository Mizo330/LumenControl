import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load audio file
audio_path = "/home/appuser/lumenai/src/wav/Avicii - Wake Me Up.wav"  # Replace with your file
y, sr = librosa.load(audio_path)

# Compute Short-Time Fourier Transform (STFT)
D = librosa.stft(y)  # Compute the STFT of the audio signal
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)  # Convert amplitude to decibel

# Display the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log', cmap='coolwarm')
plt.colorbar(format='%+2.0f dB')
plt.title('Spektogram (logaritmikus frekvencia skála)')
plt.xlabel('Idő (s)')
plt.ylabel('Frekvencia (Hz)')
plt.tight_layout()
plt.show()
