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
        
        #declare and get params for the filters..
        self.declare_parameter("filter_names", ['Bass', 'Mid', 'High'])
        self.declare_parameter("filter_frequencies", [80, 1000, 5000])
        self.declare_parameter("filter_gains", [1.0, 1.0, 2.0])
        
        names = self.get_parameter("filter_names").get_parameter_value().string_array_value
        freqs = self.get_parameter("filter_frequencies").get_parameter_value().integer_array_value
        gains = self.get_parameter("filter_gains").get_parameter_value().double_array_value

        self.filter_defs = {
            name: {"f": f, "gain": g}
            for name, f, g in zip(names, freqs, gains)
        }

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.get_logger().info(f"Using torch device {self.device} for backend torch-cuda")

        self.format : AudioFormat = None
        self.create_subscription(AudioFormat,"/audio/format",self.audio_format_callback,10)
        while self.format is None:
            rclpy.spin_once(self,timeout_sec=2)
            self.get_logger().info("Waiting on format topic..")
        
        self.create_subscription(AudioStream,"/audio/input_raw",self.audio_callback,10)
        
        # Sorted bands by central frequency
        self.bands = sorted(self.filter_defs.items(), key=lambda x: x[1]['f'])

        self.filters = {}
        self.publisher_map = {}

        for i, (name, params) in enumerate(self.bands):
            fc = params['f']
            gain = params.get('gain', 1.0)

            if i == 0:
                # Lowpass
                cutoff = (fc + self.bands[i + 1][1]['f']) / 2
                self.filters[name] = lambda w, c=cutoff, g=gain: g * F.lowpass_biquad(w, self.format.sample_rate, cutoff_freq=c, Q=1.5)
            elif i == len(self.bands) - 1:
                # Highpass
                cutoff = (fc + self.bands[i - 1][1]['f']) / 2
                self.filters[name] = lambda w, c=cutoff, g=gain: g * F.highpass_biquad(w,  self.format.sample_rate, cutoff_freq=c, Q=1.5)
            else:
                # Bandpass
                f_prev = self.bands[i - 1][1]['f']
                f_next = self.bands[i + 1][1]['f']
                bandwidth = f_next - f_prev
                Q = fc / bandwidth
                self.filters[name] = lambda w, f=fc, q=Q, g=gain: g * F.bandpass_biquad(w,  self.format.sample_rate, central_freq=f, Q=q)

            # Create ROS publisher
            topic_name = f"/audio/filtered/{name.lower()}"
            self.publisher_map[name] = self.create_publisher(AudioStream, topic_name, 10)
            
        self.get_logger().info(f"Init done,")
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

        out = {}
        for name, filt in self.filters.items():
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