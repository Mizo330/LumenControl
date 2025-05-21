import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load two audio files (replace with your files)
audio_path_1 = "/home/appuser/lumenai/output_01.wav"
audio_path_2 = "/home/appuser/lumenai/output_02.wav"
y1, sr1 = librosa.load(audio_path_1)
y2, sr2 = librosa.load(audio_path_2)

# Ensure the sampling rates match
assert sr1 == sr2, "The sampling rates of the two files must be the same."
# Function to smooth values using a running average
def smooth(values, window_size=20):
    """Apply a running average (moving average) to smooth the values."""
    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode='same')
# Compute RMS
rms1 = librosa.feature.rms(y=y1)[0]
rms2 = librosa.feature.rms(y=y2)[0]
rms1 = smooth(rms1)
rms2 = smooth(rms2)

# Compute ZCR
zcr1 = librosa.feature.zero_crossing_rate(y=y1)[0]
zcr2 = librosa.feature.zero_crossing_rate(y=y2)[0]

zcr1 = smooth(zcr1)
zcr2 = smooth(zcr2)

# Time axis for features
frames = range(len(rms1))
time = librosa.frames_to_time(frames, sr=sr1)

# Plot RMS
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time, rms1, label="Egyszeres erősítés", color='blue')
plt.plot(time, rms2, label="Kétszeres erősítés", color='orange')
plt.title("Négyzetes középérték")
plt.xlabel("Idő (s)")
plt.ylabel("RMS")
plt.legend()

# Plot ZCR
plt.subplot(2, 1, 2)
plt.plot(time, zcr1, label="Egyszeres erősítés", color='blue')
plt.plot(time, zcr2, label="Kétszeres erősítés", color='orange')
plt.title("Nullátmeneti ráta (ZCR)")
plt.xlabel("Idő (s)")
plt.ylabel("ZCR")
plt.legend()

plt.tight_layout()
plt.show()
