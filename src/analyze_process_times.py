import torchaudio
from beat_this.inference import Audio2Beats
import allin1
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

# Specify the path to your WAV file
wav_file_path = "/home/appuser/lumenai/src/wav/Dimension, Sub Focus & NGHTMRE - Angel (ft. Mougleta).wav"

# Load the audio file
waveform, sample_rate = torchaudio.load(wav_file_path)

# Convert the waveform to a NumPy array
waveform_np = waveform.numpy()

# Convert stereo to mono if necessary
if waveform_np.shape[0] == 2:
    # Average the two channels
    waveform_mono = np.mean(waveform_np, axis=0)
else:
    waveform_mono = waveform_np.squeeze()

# Cut the first 60 seconds
num_samples_60s = sample_rate * 60
waveform_60s = waveform_mono[:num_samples_60s]

# Initialize the models
beathis = Audio2Beats("/home/appuser/lumenai/src/final0.ckpt", "cuda")

# Warm up beathis
beathis(waveform_60s[:sample_rate], sr=sample_rate)

# Warm up allin1.analyze
temp_file_path = 'temp_warmup.wav'
torchaudio.save(temp_file_path, torch.from_numpy(waveform_mono[:sample_rate]).unsqueeze(0), sample_rate)
allin1.analyze(temp_file_path)
os.remove(temp_file_path)

# Arrays to store results
segment_lengths = []
beathis_times = []
allin1_times = []

# Loop over segment lengths from 60 down to 1
for i in range(60, 0, -3):
    segment_length = i  # in seconds
    num_samples = sample_rate * segment_length
    waveform_segment = waveform_60s[:num_samples]

    # Process with beathis
    start_time = time.perf_counter()
    beathis(waveform_segment, sr=sample_rate)
    end_time = time.perf_counter()
    beathis_time = end_time - start_time

    # Process with allin1.analyze
    # Save waveform_segment to temp file
    temp_file_path = 'temp_segment.wav'
    torchaudio.save(temp_file_path, torch.from_numpy(waveform_segment).unsqueeze(0), sample_rate)
    start_time2 = time.perf_counter()
    try:
        allin1.analyze(temp_file_path)
    except:
        print("error")
    end_time2 = time.perf_counter()
    allin1_time = end_time2 - start_time2

    # Record times and segment lengths
    segment_lengths.append(segment_length)
    beathis_times.append(beathis_time)
    allin1_times.append(allin1_time)

    # Remove temp file
    os.remove(temp_file_path)

# Plot for beathis
plt.figure()
plt.plot(segment_lengths, beathis_times)
plt.xlabel('Audio hossza (mp)')
plt.ylabel('Feldolgozási idő (mp)')
plt.title('beathis feldolgozási ideje a hang függvényében')
plt.gca().invert_xaxis()  # Since segment lengths decrease
plt.savefig('beathis_processing_time.png')
plt.show()

# Plot for allin1
plt.figure()
plt.plot(segment_lengths, allin1_times)
plt.xlabel('Audio hossza (mp)')
plt.ylabel('Feldolgozási idő (mp)')
plt.title('allin1 feldolgozási ideje a hang függvényében')
plt.gca().invert_xaxis()
plt.savefig('allin1_processing_time.png')
plt.show()
