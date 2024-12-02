import torchaudio
from beat_this.inference import Audio2Beats
import matplotlib.pyplot as plt
import numpy as np
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

# Initialize the model
beathis = Audio2Beats("/home/appuser/lumenai/src/final0.ckpt", "cuda")

# Number of iterations
num_iterations = 5
shift_seconds = 0.2
segment_length_seconds = 5

# Prepare the figure
fig, axes = plt.subplots(num_iterations, 1, figsize=(12, num_iterations * 2), sharex=True)

for i in range(num_iterations):
    start_time = i * shift_seconds  # in seconds
    start_sample = int(start_time * sample_rate)
    num_samples_per_segment = int(segment_length_seconds * sample_rate)
    end_sample = start_sample + num_samples_per_segment

    # Ensure we don't go beyond the audio length
    if end_sample > len(waveform_mono):
        print(f"Not enough audio data for iteration {i}.")
        break

    waveform_segment = waveform_mono[start_sample:end_sample]

    # Process with beathis
    # Assuming beathis returns a dictionary with 'beats' and 'downbeats' in seconds
    beat_times, downbeat_times = beathis(waveform_segment, sr=sample_rate)

    # Adjust beat times to absolute scale
    beat_times_absolute = beat_times + start_time
    downbeat_times_absolute = downbeat_times + start_time

    # Plotting
    ax = axes[i]
    ax.set_title(f"{i+1}. iteráció, kezdő idő: {start_time:.1f}s")
    ax.set_ylabel("Amplitúdó")

   # Plot the waveform segment
    time_axis = np.linspace(start_time, start_time + segment_length_seconds, num_samples_per_segment)
    ax.plot(time_axis, waveform_segment, color='gray', alpha=0.5)

    # Plot beats as points on the x-axis
    beat_amplitudes = np.interp(beat_times_absolute, time_axis, waveform_segment)
    ax.scatter(beat_times_absolute, beat_amplitudes, marker='o', color='blue', label='Ütemek')

    # Highlight downbeats
    downbeat_amplitudes = np.interp(downbeat_times_absolute, time_axis, waveform_segment)
    ax.scatter(downbeat_times_absolute, downbeat_amplitudes, marker='x', color='red', s=100, label='Kezdő ütem')

    # Set x-limits to show the 5-second segment
    ax.set_xlim(start_time, start_time + segment_length_seconds)

    # Add legend only to the first subplot
    if i == 0:
        ax.legend(loc='upper right')

# Common X label
plt.xlabel("Idő (s)")
plt.tight_layout()
plt.show()
