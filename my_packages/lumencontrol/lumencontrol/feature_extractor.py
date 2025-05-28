import torch
import numpy as np
import torch 
import rclpy
from rclpy.node import Node
from audio_stream_msgs.msg import AudioStream, AudioFormat
import torchaudio.sox_effects as sox
import torch.nn.functional as nnF
import librosa
from collections import deque
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from lumen_msgs.msg import AudioFeatures
from std_msgs.msg import Int16

class FeatureExtractor(Node):
    def __init__(self):
        super().__init__('demixer')

        #declare and get params for the filters..
        #TODO instead of declaring, get it from the filtering node with param.get
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


        self.create_subscription(AudioFormat,"/audio/format",self._audio_format_cb,10)
        self.format = None
        while self.format is None:
            rclpy.spin_once(self,timeout_sec=2)
            self.get_logger().info("Waiting on format topic..")

        self.callback_group = ReentrantCallbackGroup()
        
        self.current_bpm = 120
        self.create_subscription(AudioStream, "/audio/bpm",self.bpm_cb, 10,callback_group=self.callback_group)
                
        self.audio_subscribers = {}
        self.audio_subscriber_cbs = {}
        self.featue_publishers = {}
        self.rms_history = {}
        self.zcr_history = {}
        self.rms_smoothed_history = {}
        self.zcr_smoothed_history = {}
        
        #Create all subs, pubs and containers from the filter definitions
        for name in self.filter_defs:
            
            self.featue_publishers[name] = self.create_publisher(
                AudioFeatures,
                f"/features/{name.lower()}",
                10,
                callback_group=self.callback_group
            )
            
            # Define callback inside a function to capture 'name'
            def make_callback(filter_name):
                def callback(msg):
                    # Process msg here
                    feature_msg = self.get_features(msg.data, filter_name)
                    self.featue_publishers[filter_name].publish(feature_msg)
                return callback

            callback_fn = make_callback(name)
            self.audio_subscribers[name] = self.create_subscription(
                AudioStream,
                f"/audio/filtered/{name.lower()}",
                callback_fn,
                10,
                callback_group=self.callback_group
            )
            self.audio_subscriber_cbs[name] = callback_fn
            
            self.rms_history[name] = deque(maxlen=1000)
            self.zcr_history[name] = deque(maxlen=1000)
            self.rms_smoothed_history[name] = deque(maxlen=1000)
            self.zcr_smoothed_history[name] = deque(maxlen=1000)

        self.get_logger().info("Init done")
        
    def _audio_format_cb(self, msg: AudioFormat):
        self.format = msg  
    
    def bpm_cb(self,msg:Int16):
        self.current_bpm = msg.data

    def get_features(self,data,band):
        if self.format.channels == 2:
        # Reshape to (n_frames, 2) and average channels
            samples = np.frombuffer(data, dtype=self.format.dtype).reshape(-1, 2).mean(axis=1)
        else:
            samples = np.frombuffer(data, dtype=self.format.dtype)

        samples = samples.astype(np.float32) / 32768.0

        rms = np.sqrt(np.mean(samples**2))

        zcr = np.mean(np.abs(np.diff(np.sign(samples)))) / 2
        zcr = zcr*10
        
        self.rms_history[band].append(rms)
        self.zcr_history[band].append(zcr)
        
        #smooth features
        avg_size = round((60 / self.current_bpm) * (self.format.sample_rate / self.format.frame_size))
        rms_array = np.fromiter(self.rms_history[band], dtype=np.float32)
        rms_smoothed = rms_array[-avg_size:].mean() 
        zcr_array = np.fromiter(self.zcr_history[band], dtype=np.float32)
        zcr_smoothed = zcr_array[-avg_size:].mean() 
        
        self.rms_smoothed_history[band].append(rms_smoothed)
        self.zcr_smoothed_history[band].append(zcr_smoothed)
        
        rms_s_array = np.fromiter(self.rms_smoothed_history[band], dtype=np.float32)
        rms_z_score = self.get_z_score(rms_smoothed,rms_s_array)
        
        zcr_s_array = np.fromiter(self.zcr_smoothed_history[band], dtype=np.float32)
        zcr_z_score = self.get_z_score(zcr_smoothed,zcr_s_array)
                
        features = AudioFeatures(rms = float(rms_smoothed),
                                 zcr=float(zcr_smoothed),
                                 rms_z_score=float(rms_z_score),
                                 zcr_z_score=float(zcr_z_score))
        return features
    
    def get_z_score(self,val, arr):
        if len(arr) < 2:
            return 0.0
        std = np.std(arr)
        if not np.isfinite(std) or std < 1e-6:
            return 0.0
        return (val - np.mean(arr)) / std

def main(args=None):
    rclpy.init()
    executor = MultiThreadedExecutor()
    node = FeatureExtractor()

    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.destroy_node()
        executor.shutdown()

if __name__ == '__main__':
    main()