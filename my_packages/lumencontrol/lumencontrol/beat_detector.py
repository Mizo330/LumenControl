import numpy as np
import pyaudio
import threading
from beat_this.inference import Audio2Beats
import torch  # Added import for torch
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy import time
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from audio_stream_msgs.msg import AudioStream, AudioFormat
from audio_stream_msgs.srv import GetAudioFormat
from std_msgs.msg import Bool

class BeatDetector(Node):

    def __init__(self):
        super().__init__('beat_detector')

        self.get_logger().info("Waiting for input format service..")        
        self.format_client = self.create_client(GetAudioFormat,"/audio/get_format")
        while not self.format_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /audio/get_format service...')
        
        format = self.get_audio_format()    
        self.sample_rate = format.sample_rate
        self.input_channels = format.channels
        self.dtype = format.dtype
        self.buffer = deque(maxlen= self.sample_rate * 5)
        self.current_stamp = rclpy.time.Time()
        self.last_beat = rclpy.time.Time()
        self.lock = threading.Lock()
        
        self.beatThis = Audio2Beats("/home/appuser/lumencontrol/src/my_packages/lumencontrol/checkpoints/final0.ckpt", "cuda")

        self.create_subscription(AudioStream,"/audio/input_raw",self.audio_callback,10)
        self.beat_publisher = self.create_publisher(Bool, "/lumenctrl/beat",10)
        # Process buffer periodically (e.g., 10 Hz)
        self.create_timer(0.05, self.process_buffer)
        
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
            current_stamp = self.current_stamp

        start = self.get_clock().now()
        beats, upbeats = self.beatThis(buffer_copy, self.sample_rate)
        delta = self.get_clock().now() - start
        is_new_beat = self.filter_for_new_beat(beats,current_stamp)
        self.get_logger().debug(f"Beat processing time: {delta} s")
        if is_new_beat:
            self.beat_publisher.publish(Bool(data=True))
        #self.get_logger().info(f"beats:{beats}")
        
    def filter_for_new_beat(self, beats: np.ndarray, timestamp) -> bool:
        """
        Filters for the latest beat and checks if it's newer than self.last_beat.
        If yes, updates self.last_beat and returns True.
        """
        if len(beats) == 0:
            return False
        tol_ns = int(0.2 * 1e9) #lets have a 0.2s tolerance window -> cut at 300BPM
        latest_beat_sec = beats[-1]

        # Convert beat offset (in seconds) to nanoseconds
        beat_time_ns = timestamp.nanoseconds + int(latest_beat_sec * 1e9)
        beat_time = rclpy.time.Time(nanoseconds=beat_time_ns)

        # If beat is newer than last published one
        if beat_time.nanoseconds > self.last_beat.nanoseconds+tol_ns:
            self.get_logger().info(f"New beat at: {beat_time.nanoseconds / 1e9:.2f}s")
            self.last_beat = beat_time
            return True

        return False
            
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
    
def main(args=None):
    rclpy.init(args=args)
    node = BeatDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()