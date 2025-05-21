from xumx_slicq_v2.separator import Separator
from xumx_slicq_v2 import data
import time
import torchaudio
import torch
import numpy as np
import pyaudio
import threading
import torch  # Added import for torch
from collections import deque
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from audio_stream_msgs.msg import AudioStream, AudioFormat
from audio_stream_msgs.srv import GetAudioFormat
from std_msgs.msg import Bool
import torchaudio.functional as F


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

class Demixer(Node):
    def __init__(self):
        super().__init__('demixer')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using torch device {self.device} for backend torch-cuda")

        # create separator only once to reduce model loading
        # when using multiple files
        self.separator = Separator.load(
            device=self.device,
            runtime_backend='torch-cuda',
            realtime=True,
            warmup=2,
            model_path=None,
        )
        self.demixed_tracks = {}  # key: str, value: torch.Tensor
        self.demix_format = None  
        self.demix_index = 0      # optional, to cycle through tracks
        self.format_client = self.create_client(GetAudioFormat,"/audio/get_format")
        while not self.format_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /audio/get_format service...')
        format = self.get_audio_format()    
        self.format = format
        self.sample_rate = format.sample_rate
        self.input_channels = format.channels
        self.dtype = format.dtype
        self.buffer = deque(maxlen= self.sample_rate * 5)
        self.lock = threading.Lock()
        
        self.create_subscription(AudioStream,"/audio/input_raw",self.audio_callback,10)
        #self.beat_publisher = self.create_publisher(Bool, "/lumenctrl/beat",10)
        # Process buffer periodically (e.g., 10 Hz)
        #self.create_timer(0.05, self.process_buffer)
        
        self.publisher = self.create_publisher(AudioStream, "demixed_audio", 10)
        self.create_timer(0.5, self.publish_demixed_estimate)
        self.create_timer(0.5, self.process_buffer)

    def audio_callback(self,msg:AudioStream):
        if  self.input_channels == 2:
            # Reshape to (n_frames, 2) and average across channels
            samples = np.frombuffer(msg.data, dtype=self.dtype).reshape(-1, 2).mean(axis=1)
        else:
            samples = np.frombuffer(msg.data, dtype=self.dtype)
        samples = samples.astype(np.float32) / 32768.0 

        with self.lock:
            self.buffer.extend(samples)
            self.current_stamp = rclpy.time.Time.from_msg(msg.stamp)
            
    def process_buffer(self):
        with self.lock:
            if len(self.buffer) < 2000:
                return
            buffer_copy = np.array(self.buffer)

        audio= torch.from_numpy(buffer_copy)
        estimates, time_taken = separate(
            audio=audio,
            rate=self.sample_rate,
            separator=self.separator,
            device=self.device,
            )
        # Save format for publishing
        self.demix_format = self.dtype
            # write out estimates
        # Save estimates
        self.demixed_tracks = estimates
        self.demix_index = 0  # reset index

        self.get_logger().info(f"Time taken:{time_taken}")
    
    def publish_demixed_estimate(self):
        if not self.demixed_tracks:
            return  # Nothing to publish yet

        keys = list(self.demixed_tracks.keys())
        target = keys[self.demix_index % len(keys)]
        estimate = self.demixed_tracks[target]  # shape: [channels, time]
        self.demix_index += 1

        # Convert back to int16
        estimate_np = (estimate.clamp(-1.0, 1.0) * 32768.0).to(torch.int16).cpu().numpy()

        if estimate_np.shape[0] == 1:
            int16_data = estimate_np[0]  # mono
        else:
            int16_data = estimate_np.T.reshape(-1)  # [time, channels] → flat

        msg = AudioStream()
        msg.stamp = self.get_clock().now().to_msg()
        msg.format = self.format
        msg.data = list(int16_data.tobytes())

        self.publisher.publish(msg)
        self.get_logger().info(f"Published demixed track: {target}")
    
    def get_audio_format(self):
        req = GetAudioFormat.Request()
        self.future = self.format_client.call_async(req)
        self.get_logger().info('Request sent, waiting for response...')

        rclpy.spin_until_future_complete(self, self.future)

        if self.future.result() is not None:
            format_resp = self.future.result()
            self.get_logger().info(f'Got format')
        else:
            self.get_logger().error('Service call failed.')
        return format_resp.format    
    
    # Apply filters using torchaudio.functional
    def apply_filters(self, waveform, sample_rate):
        filters = {
            "Bass": lambda w: F.lowpass_biquad(w, sample_rate, cutoff_freq=130),
            "Lowmid": lambda w: F.bandpass_biquad(w, sample_rate, central_freq=642, Q=1),
            "High": lambda w: F.highpass_biquad(w, sample_rate, cutoff_freq=2000),
        }
        filtered_waveforms = {name: filt(waveform) for name, filt in filters.items()}
        return filtered_waveforms
    
def main(args=None):
    rclpy.init(args=args)
    node = Demixer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()