import torch
import numpy as np
import torch 
import time
import rclpy
from rclpy.node import Node
from audio_stream_msgs.msg import AudioStream, AudioFormat
from collections import deque
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from lumen_msgs.msg import AudioFeatures
from std_msgs.msg import Int16
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import torchaudio
import librosa
from std_msgs.msg import Float32

class Spectrogram(Node):
    def __init__(self):
        super().__init__('spectrogram')

        self.create_subscription(AudioFormat, "/audio/format", self._audio_format_cb, 10)
        self.format: AudioFormat = None
        while self.format is None:
            rclpy.spin_once(self, timeout_sec=2)
            self.get_logger().info("Waiting on format topic..")

        self.callback_group = ReentrantCallbackGroup()
        self.create_subscription(AudioStream, "audio/filtered/bass", self.process_raw, 10, callback_group=self.callback_group)

        # Parameters
        self.n_fft = 1024
        self.hop_length = 512
        self.buffer_size = 10240

        # Setup buffer
        self.buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # Mel spectrogram
        self.spectral_centroid = T.SpectralCentroid(
        sample_rate=self.format.sample_rate,
        n_fft=self.n_fft,
        win_length=self.n_fft,
        hop_length= self.hop_length,
        pad=0,
        window_fn=torch.hann_window).to("cuda")
        
        self.fluxpub = self.create_publisher(Float32,"/spectral/flux",10)

        self.get_logger().info("Init done")

    def _audio_format_cb(self, msg: AudioFormat):
        self.format = msg  
        
    def process_raw(self, msg: AudioStream):
        start = time.time()
        if self.format.channels == 2:
            samples = np.frombuffer(msg.data, dtype=self.format.dtype).reshape(-1, 2).mean(axis=1)
        else:
            samples = np.frombuffer(msg.data, dtype=self.format.dtype)

        samples = samples.astype(np.float32) / 32768.0
        # Update buffer
        self.buffer = np.roll(self.buffer, -len(samples))
        self.buffer[-len(samples):] = samples

        # Spectrogram
        torch_samples = torch.from_numpy(self.buffer).to("cuda")
        centroid = self.spectral_centroid(torch_samples).cpu().numpy()  # (n_mels, time)
        
        flux = self.compute_spectral_flux(torch_samples)
        self.get_logger().debug(f"C is {centroid[0]}")
        self.get_logger().debug(f"Flux is {flux.mean()}")
        
        self.fluxpub.publish(Float32(data=float(flux.mean())))

        dt = time.time()-start
        self.get_logger().debug(f"Time taken: {dt}")
        
    def compute_spectral_flux(self, waveform):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Ensure shape (1, time)

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=512,
            win_length=self.n_fft,
            window=torch.hann_window(self.n_fft).to(waveform.device),
            return_complex=True
        )  # shape: (batch, freq_bins, time_frames)

        magnitude = torch.abs(stft)  # shape: (batch, freq_bins, time_frames)

        # Frame-wise difference and norm over frequency bins
        delta = magnitude[:, :, 1:] - magnitude[:, :, :-1]
        flux = torch.norm(delta, dim=1)  # shape: (batch, time_frames - 1)

        return flux.squeeze(0)  # Remove batch dim if only one channel

        
def main(args=None):
    rclpy.init(args=args)
    node = Spectrogram()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()