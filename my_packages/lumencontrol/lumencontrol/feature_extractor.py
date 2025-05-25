import torch
import numpy as np
import torch 
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from audio_stream_msgs.msg import AudioStream, AudioFormat
import torchaudio.sox_effects as sox
import soundfile as sf
import torch.nn.functional as nnF
import librosa
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

class FeatureExtractor(Node):
    def __init__(self):
        super().__init__('demixer')

        self.create_subscription(AudioFormat,"/audio/format",self._audio_format_cb,10)
        self.format:AudioFormat = None
        while self.format is None:
            rclpy.spin_once(self,timeout_sec=2)
            self.get_logger().info("Waiting on format topic..")

        self.callback_group = ReentrantCallbackGroup()
        self.create_subscription(AudioStream, "/audio/filtered/high",self.process_high, 10,callback_group=self.callback_group)
        self.create_subscription(AudioStream, "/audio/filtered/lowmid",self.process_mid, 10,callback_group=self.callback_group)
        self.create_subscription(AudioStream, "/audio/filtered/bass",self.process_bass, 10,callback_group=self.callback_group)
      
    def _audio_format_cb(self, msg: AudioFormat):
        self.format = msg  
        
    def process_high(self,msg:AudioStream):
        zcr, rms = self.get_features(msg.data)
        self.get_logger().info(f"ZCR:{zcr},RMS:{rms}")
        
    def process_mid(self,msg):
        zcr, rms = self.get_features(msg.data)
        
    def process_bass(self,msg):
        zcr, rms = self.get_features(msg.data)

    def get_features(self,data):
        if self.format.channels == 2:
        # Reshape to (n_frames, 2) and average channels
            samples = np.frombuffer(data, dtype=self.format.dtype).reshape(-1, 2).mean(axis=1)
        else:
            samples = np.frombuffer(data, dtype=self.format.dtype)

        # Normalize samples to float32 in [-1,1]
        samples = samples.astype(np.float32) / 32768.0

        # Compute RMS
        rms = np.sqrt(np.mean(samples**2))

        # Zero Crossing Rate (ZCR) calculation
        samples_tensor = torch.from_numpy(samples)  # Convert to tensor
        samples_tensor = samples_tensor.float()

        waveform_shifted = samples_tensor[1:]
        zero_crossings = ((samples_tensor[:-1] * waveform_shifted) < 0).float()

        # Reshape zero_crossings to (N=1, C=1, L)
        zero_crossings = zero_crossings.unsqueeze(0).unsqueeze(0)

        input_length = zero_crossings.shape[-1]
        frame_size = self.format.frame_size
        stride = frame_size // 2

        if frame_size > input_length:
            # Option 1: Pad zero_crossings to frame_size length
            pad_amount = frame_size - input_length
            zero_crossings = torch.nn.functional.pad(zero_crossings, (0, pad_amount))


        # Apply avg_pool1d
        zcr = nnF.avg_pool1d(zero_crossings, kernel_size=frame_size, stride=stride).squeeze()
        
        return zcr,rms
    
def main(args=None):
    rclpy.init()
    executor = MultiThreadedExecutor()
    node = FeatureExtractor()

    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        executor.shutdown()

if __name__ == '__main__':
    main()