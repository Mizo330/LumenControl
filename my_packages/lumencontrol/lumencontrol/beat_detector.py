import numpy as np
import threading
from beat_this.inference import Audio2Beats
import math
from collections import deque
import rclpy
from rclpy.node import Node
from rclpy import time
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from audio_stream_msgs.msg import AudioStream, AudioFormat
from std_msgs.msg import Bool, Int16

class BeatDetector(Node):

    def __init__(self):
        super().__init__('beat_detector')

        self.create_subscription(AudioFormat,"/audio/format",self._audio_format_cb,10)

        self.format:AudioFormat = None
        while self.format is None:
            rclpy.spin_once(self,timeout_sec=2)
            self.get_logger().info("Waiting on format topic..")
        
        self.buffer = deque(maxlen= self.format.sample_rate * 5)
        self.current_stamp = rclpy.time.Time()
        self.last_beat = rclpy.time.Time()
        self.recent_beats = deque(maxlen=40)  #TODO make this a param
        self.bpm = 0.0
        self.lock = threading.Lock()
        
        self.beatThis = Audio2Beats("/home/appuser/lumencontrol/src/my_packages/lumencontrol/checkpoints/final0.ckpt", "cuda")

        self.create_subscription(AudioStream,"/audio/input_raw",self.audio_callback,10)
        self.beat_publisher = self.create_publisher(Bool, "/lumenctrl/beat",10)
        self.bpm_publisher = self.create_publisher(Int16,"/audio/bpm",10)
        # Process buffer periodically (e.g., 10 Hz)
        self.create_timer(0.05, self.process_buffer)
        
    def audio_callback(self,msg:AudioStream):
        if self.format.channels == 2:
            # Reshape to (n_frames, 2) and average across channels
            samples = np.frombuffer(msg.data, dtype=self.format.dtype).reshape(-1, 2).mean(axis=1)
        else:
            samples = np.frombuffer(msg.data, dtype=self.format.dtype)
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
        beats, upbeats = self.beatThis(buffer_copy, self.format.sample_rate)
        delta = self.get_clock().now() - start
        is_new_beat = self.filter_for_new_beat(beats,current_stamp)
        self.get_logger().debug(f"Beat processing time: {delta} s")
        self.get_logger().debug(f"beats:{beats}")
        
    def filter_for_new_beat(self, beats: np.ndarray, timestamp) -> bool:
        """
        Filters for the latest beat and checks if it's newer than self.last_beat.
        If yes, updates self.last_beat and returns True. Also calculates BPM from last N beats.
        """
        if len(beats) == 0:
            return False

        tol_ns = int(0.3 * 1e9)  # 0.2s tolerance -> 200 BPM max
        latest_beat_sec = beats[-1]
        beat_time_ns = timestamp.nanoseconds + int(latest_beat_sec * 1e9)
        beat_time = rclpy.time.Time(nanoseconds=beat_time_ns)

        if beat_time.nanoseconds > self.last_beat.nanoseconds + tol_ns:
            self.last_beat = beat_time
            self.recent_beats.append(beat_time.nanoseconds)
            self.beat_publisher.publish(Bool(data=True))
            # Calculate BPM from recent beats
            if len(self.recent_beats) >= 2:
                intervals_sec = np.diff(self.recent_beats) / 1e9  # Convert ns to sec
                avg_interval = np.mean(intervals_sec)
                if avg_interval > 0:
                    self.bpm = 60.0 / avg_interval
                    self.get_logger().debug(f"Estimated BPM: {self.bpm:.2f}")
                    self.bpm_publisher.publish(Int16(data=round(self.bpm)))
            
    def _audio_format_cb(self, msg: AudioFormat):
        self.format = msg
    
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