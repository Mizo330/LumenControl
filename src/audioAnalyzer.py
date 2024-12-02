import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from beat_this.inference import File2File

# Load the audio file
def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

# Apply filters using torchaudio.functional
def apply_filters(waveform, sample_rate):
    filters = {
        "Aluláteresztő szűrő (130 Hz)": lambda w: F.lowpass_biquad(w, sample_rate, cutoff_freq=130),
        "Sáváteresztő szűrő (260-1024 Hz)": lambda w: F.bandpass_biquad(w, sample_rate, central_freq=642, Q=1),
        "Sáváteresztő szűrő (512-2048 Hz)": lambda w: F.bandpass_biquad(w, sample_rate, central_freq=1250, Q=1),
        "Felüláteresztő szűrő (2048 Hz)": lambda w: F.highpass_biquad(w, sample_rate, cutoff_freq=2000),

    }
    filtered_waveforms = {name: filt(waveform) for name, filt in filters.items()}
    return filtered_waveforms

# Compute the Short-Time Fourier Transform (STFT)
def compute_stft(waveform, n_fft=1024, hop_length=None):
    stft_transform = T.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None)
    stft_result = stft_transform(waveform)
    magnitude = torch.abs(stft_result)
    return magnitude

# Generalized Plotter for Features
def plot_features_subplots(features, feature_times, duration, smoothing="moving_average", ylabel="Feature Value", deviation_window=None):
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 10), sharex=True)

    for ax, (label, feature) in zip(axes, features.items()):
        # Apply smoothing
        if smoothing == "moving_average":
            smoothed_feature = smooth_moving_average(feature)
        elif smoothing == "gaussian":
            smoothed_feature = smooth_gaussian(feature)
        else:
            smoothed_feature = feature  # No smoothing

        # Plot the smoothed feature
        ax.plot(feature_times, smoothed_feature[: len(feature_times)], label=label)
        
        # Compute and plot standard deviation if deviation_window is specified
        if deviation_window:
            rolling_std = np.array([
                smoothed_feature[max(0, i - deviation_window):i + 1].std()
                for i in range(len(smoothed_feature))
            ])
            ax.plot(feature_times[: len(rolling_std)], rolling_std, label=f"{label} (Szórás)", color="orange")

        ax.set_title(label)
        ax.set_ylabel(ylabel)
        ax.legend()
        
    axes[-1].set_xlabel("Time (s)")
    xticks = np.arange(0, duration + 1, 10)
    for ax in axes:
        ax.set_xticks(xticks)
        ax.set_xlim(0, duration)

    plt.tight_layout()
    plt.show()


# Compute Spectral Flux
def compute_spectral_flux(magnitude_spectrogram):
    flux = torch.diff(magnitude_spectrogram, dim=-1).pow(2).sum(dim=1).sqrt()
    return flux.mean(dim=0).numpy()


# Compute Zero-Crossing Rate (ZCR)
def compute_zcr(waveform, frame_length, hop_length):
    waveform_shifted = waveform[:, 1:]
    zero_crossings = ((waveform[:, :-1] * waveform_shifted) < 0).float()
    zcr = torch.nn.functional.avg_pool1d(zero_crossings, frame_length, stride=hop_length).squeeze()
    return zcr.numpy()


# Compute Root Mean Square Energy (RMS)
def compute_rms(waveform, frame_length, hop_length):
    frames = waveform.unfold(1, frame_length, hop_length)
    rms = frames.pow(2).mean(dim=-1).sqrt()
    return rms.squeeze().numpy()

# Compute Root Mean Square Energy (RMS) for causal processing
def compute_rms_causal(waveform, frame_length, hop_length):
    num_frames = (waveform.shape[1] - frame_length) // hop_length + 1
    frames = torch.zeros((waveform.shape[0], num_frames, frame_length), device=waveform.device)

    for i in range(num_frames):
        start = i * hop_length
        end = start + frame_length
        frames[:, i, :] = waveform[:, max(0, start):end]

    rms = frames.pow(2).mean(dim=-1).sqrt()
    return rms.squeeze().numpy()


# Compute RMS Difference
def compute_rms_difference(rms):
    d_rms = np.diff(rms, prepend=rms[0])  # Prepend the first value to maintain length
    return d_rms

def flux_difference(flux):
    d_s = np.abs(np.diff(flux, prepend=flux[0]))
    return d_s

# Compute Novelty Curve
def compute_novelty_curve(flux, zcr):
    novelty = np.abs(np.diff(flux, prepend=flux[0])) + np.abs(np.diff(zcr, prepend=zcr[0]))
    return novelty


# Normalize waveform for proper visualization
def normalize_waveform(waveform):
    max_val = torch.max(torch.abs(waveform))
    return waveform / max_val


# Plot results
def plot_results(times, audio_times, waveform, zcr, d_rms, flux, novelty, duration):
    fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
    
     # Plot audio waveform
    ax[0].plot(audio_times, waveform.squeeze().numpy(), label="Hanghullám", color="blue")
    ax[0].set_title("Hanghullám")
    ax[0].legend()
    ax[0].set_ylim(-1.1, 1.1)  # Ensure waveform fits within [-1, 1]
    
    # Plot ZCR
    ax[1].plot(times, zcr, label="Zéró átlépési arány (ZCR)")
    ax[1].set_title("Zéró átlépési arány")
    ax[1].legend()
    
    # Plot RMS Difference
    ax[2].plot(times, d_rms, label="RMS", color="orange")
    ax[2].set_title("RMS (Négyzetes közép)")
    ax[2].legend()
    
    # Plot Spectral Flux
    ax[3].plot(times, flux, label="Spektrális fluxus", color="green")
    ax[3].set_title("Spektrális fluxus")
    ax[3].legend()
    
    # Plot Novelty Curve
    ax[4].plot(times, novelty, label="Újdonsági Görbe", color="purple")
    ax[4].set_title("Újdonsági Görbe")
    ax[4].legend()
    
    # Set x-axis labels every 10 seconds
    xticks = np.arange(0, duration + 1, 5)
    for a in ax:
        a.set_xticks(xticks)
        a.set_xlim(0, duration)
    
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

# Smoothing: Moving Average
def smooth_moving_average(data, window_size=30):
    print(np.shape(data))
    return np.convolve(data, np.ones(window_size) / window_size, mode="same")


# Smoothing: Gaussian Smoothing
def smooth_gaussian(data, sigma=4):
    return gaussian_filter1d(data, sigma=sigma)

# Main function
def main(file_path):
    waveform, sample_rate = load_audio(file_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono if stereo

    # Apply filters
    filtered_waveforms = apply_filters(waveform, sample_rate)
    
    # # Normalize waveform for better visualization
    # waveform = normalize_waveform(waveform)
    
    # Audio duration
    duration = waveform.shape[1] / sample_rate
    # Parameters
    n_fft = 1024
    hop_length = 512
    frame_length = 1024

    # Compute STFT and magnitude
    magnitude_spectrogram = compute_stft(waveform, n_fft=n_fft, hop_length=hop_length)

    # Initialize dictionaries to store features for filtered waveforms
    filtered_flux = {}
    filtered_rms = {}
    filtered_zcr = {}

    for label, filtered_waveform in filtered_waveforms.items():
        rms = compute_rms_causal(filtered_waveform, frame_length, hop_length)
        zcr = compute_zcr(filtered_waveform, frame_length, hop_length)
        magnitude_spectrogram = compute_stft(filtered_waveform, n_fft=n_fft, hop_length=hop_length)
        flux = compute_spectral_flux(magnitude_spectrogram)
        # Store the features
        filtered_flux[label] = flux
        filtered_rms[label] = rms
        filtered_zcr[label] = zcr
        
            
    feature_times = np.linspace(0, duration, len(list(filtered_flux.values())[0]))

    # Plot each feature
    plot_features_subplots(filtered_flux, feature_times, duration, smoothing="moving_average", ylabel="Spektrális fluxus",deviation_window=15)
    plot_features_subplots(filtered_rms, feature_times[:-1], duration, smoothing="moving_average", ylabel="RMS",deviation_window=15)
    plot_features_subplots(filtered_zcr, feature_times[:-2], duration, smoothing="moving_average", ylabel="Zéró átlépési arány (ZCR)",deviation_window=15)
    if True: 
        # Compute features
        flux = compute_spectral_flux(magnitude_spectrogram)
        zcr = compute_zcr(waveform, frame_length, hop_length)
        rms = compute_rms(waveform, frame_length, hop_length)

        # Compute RMS difference
        d_rms = compute_rms_difference(rms)

        # Compute Novelty Curve
        novelty = compute_novelty_curve(flux[: len(zcr)], zcr)

        # Align dimensions with time axis
        flux = flux[: len(zcr)]  # Align flux length with zcr

        # Compute time axis for features
        feature_times = np.linspace(0, waveform.shape[1] / sample_rate, len(zcr))

        # Compute time axis for audio waveform
        audio_times = np.linspace(0, waveform.shape[1] / sample_rate, waveform.shape[1])
        plot_results(feature_times, audio_times, waveform, zcr, rms[:-1], flux, novelty, duration)






# Run the script
file_path = "/home/appuser/lumenai/segment_avicii.wav"  # Replace with your WAV file path
main(file_path)
