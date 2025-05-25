import time
import torchaudio
import torch
import numpy as np
import pyaudio
import threading
import torch 
from collections import deque
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from audio_stream_msgs.msg import AudioStream, AudioFormat
from std_msgs.msg import Bool
import torchaudio.sox_effects as sox
from collections import defaultdict
import soundfile as sf
import torchaudio.functional as F
class Filterer(Node):
    def __init__(self):
        super().__init__('demixer')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using torch device {self.device} for backend torch-cuda")

        self.format : AudioFormat = None
        self.create_subscription(AudioFormat,"/audio/format",self.audio_format_callback,10)
        while self.format is None:
            rclpy.spin_once(self,timeout_sec=2)
            self.get_logger().info("Waiting on format topic..")
        
        self.create_subscription(AudioStream,"/audio/input_raw",self.audio_callback,10)
        
        self.bass_publisher = self.create_publisher(AudioStream, "/audio/filtered/bass", 10)
        self.lowmid_publisher = self.create_publisher(AudioStream, "/audio/filtered/lowmid", 10)
        self.high_publisher = self.create_publisher(AudioStream, "/audio/filtered/high", 10)
        # Publish to corresponding topic
        self.publisher_map = {
            "Bass": self.bass_publisher,
            "Lowmid": self.lowmid_publisher,
            "High": self.high_publisher,
        }
        self.get_logger().info("Init done")
        self.filtered_buffers = defaultdict(list)  # at init
        
    def audio_format_callback(self, msg):
        self.format = msg
        
    def audio_callback(self,input_msg:AudioStream):
        a = time.time()
        # Convert bytes to float32 waveform
        samples = np.frombuffer(input_msg.data, dtype=self.format.dtype)
        if self.format.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        samples = samples.astype(np.float32) / 32768.0  # Normalize 16-bit PCM

        # Move to CUDA and add batch dimension
        waveform = torch.from_numpy(samples).unsqueeze(0).to(self.device)  # Shape: (1, N)
        
        filtered_waveforms = self.apply_filters(waveform, self.format.sample_rate)
        #time.sleep(0.05)
        for name, filtered_waveform in filtered_waveforms.items():
            peak = filtered_waveform.abs().max()
            if peak > 1.0:
                filtered_waveform = filtered_waveform / peak
            self.get_logger().debug(f"{name}: shape={filtered_waveform.shape}, min={filtered_waveform.min().item()}, max={filtered_waveform.max().item()}")
            # Clamp to [-1, 1], convert to int16 PCM format
            filtered_clamped = filtered_waveform.squeeze(0).clamp(-0.99, 0.99)
            filtered_int16 = (filtered_clamped * 32767.0).to(torch.int16)

            # Convert mono → stereo if needed
            if self.format.channels == 2:
                # Duplicate each sample into L & R channel
                filtered_int16 = filtered_int16.repeat_interleave(2).view(-1, 2).flatten()

            # Convert to bytes
            data_bytes = filtered_int16.cpu().numpy().tobytes()

            # Create AudioStream message
            msg = AudioStream()
            msg.stamp = input_msg.stamp
            msg.data = list(data_bytes)
            self.publisher_map[name].publish(msg)
            
        d = time.time()-a
        self.get_logger().debug(f"Filtered time taken:{d}")           
    
    def apply_filters(self, waveform, sample_rate):
        N = waveform.size(-1)
        # create a Hann window
        window = torch.hann_window(N, device=waveform.device)

        # your filter definitions
        filters = {
            "Bass":     lambda w: F.lowpass_biquad(w, sample_rate, cutoff_freq=130),
            "Lowmid":   lambda w: F.bandpass_biquad(w, sample_rate, central_freq=642, Q=1),
            "High":     lambda w: F.highpass_biquad(w, sample_rate, cutoff_freq=3000),
        }

        out = {}
        for name, filt in filters.items():
            # apply filter, then window
            w = filt(waveform.clone())
            w = w * window.unsqueeze(0)  # multiply each frame by the window
            out[name] = w

        return out
        
def main(args=None):
    rclpy.init(args=args)
    node = Filterer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()