import os
import torchaudio
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from xumx_slicq_v2.separator import Separator
from xumx_slicq_v2 import data# Configuration
import time
from tqdm import trange, tqdm

FOLDER_PATH = '/home/appuser/lumencontrol/src/my_packages/wav'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_RATE = 44100  # Replace with your NN’s expected rate
DURATIONS = np.arange(0.1, 2.1, 0.1)  # Seconds


def separate(
    audio,
    separator,
    rate=None,
    device=None,
):
    if rate is None:
        raise Exception("rate` must be provided.")

    if device:
        audio = audio.to(device)
    audio = data.preprocess_audio(audio, rate, separator.sample_rate)

    # getting the separated signals
    start_time = time.time()

    estimates = separator(audio)
    time_delta = time.time() - start_time
    estimates = separator.to_dict(estimates)
    return estimates, time_delta

# Load separator NN
separator = Separator.load(
    device=DEVICE,
    runtime_backend='torch-cuda',
    realtime=True,
    warmup=2,
    model_path=None,
)

separator.quiet = True

# Helper: load and resample audio
def load_audio(path):
    audio, sr = torchaudio.load(path)
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio

# Collect results
time_results = {round(dur, 2): [] for dur in DURATIONS}
wav_files = list(Path(FOLDER_PATH).rglob("*.wav"))

for wav_file_num, wav_file in enumerate(tqdm(os.listdir(FOLDER_PATH))):
    input_file = os.path.join(FOLDER_PATH, wav_file)

    for dur in DURATIONS:
        audio, rate = data.load_audio(input_file, start=0.1, dur=dur)
        num_samples = audio.shape[1]
        try:
            estimates, time_taken = separate(
                audio=audio,
                rate=rate,
                separator=separator,
                device=DEVICE,
            )
            
            outdir = Path("saved_outputs") / Path(wav_file).stem
            outdir.mkdir(parents=True, exist_ok=True)

            for name, estimate in estimates.items():
                target_path = str(outdir / Path(name).with_suffix('.wav'))
                torchaudio.save(
                    target_path,
                    torch.squeeze(estimate).detach().cpu(),
                    encoding="PCM_F",  # pcm float for dtype=float32 wav
                    sample_rate=separator.sample_rate,
                )

            time_results[round(dur, 2)].append(time_taken)
        except Exception as e:
            print(f"Error processing {wav_file.name} at {dur}s: {e}")

# Compute averages
avg_times = [np.mean(time_results[round(dur, 2)]) for dur in DURATIONS]

# Ensure arrays are numpy
lengths = np.array(DURATIONS)
avg_times = np.array(avg_times)

# Create mask for points above y = x
mask = avg_times > lengths

# Plot
plt.figure(figsize=(8, 6))
plt.plot(lengths, lengths, 'k--', label='y = x')  # Reference line

# Scatter plot with conditional coloring
plt.scatter(lengths[~mask], avg_times[~mask], label='x>y', color='blue')
plt.scatter(lengths[mask], avg_times[mask], label='x<y', color='red')

plt.xlabel("Audio hossz (s)")
plt.ylabel("Demixelési idő (s)")
plt.title("Demixelési idő a bemeneti hossz függvényében.")
plt.legend()
plt.grid(True)
plt.ylim(bottom=0)  # Start y-axis at 0
plt.tight_layout()
plt.show()