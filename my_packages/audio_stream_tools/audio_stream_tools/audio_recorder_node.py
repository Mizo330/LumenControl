import rclpy
from rclpy.node import Node
from audio_stream_msgs.msg import AudioStream, AudioFormat
import wave
import time

class AudioRecorder(Node):
    def __init__(self):
        super().__init__('audio_recorder')
        self.subscription = self.create_subscription(
            AudioStream,
            '/audio/input_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Audio parameters (adjust as needed)
        self.sample_rate = 44100  # Hz
        self.channels = 2
        self.sample_width = 2  # bytes (16-bit audio)

        self.buffer = bytearray()
        self.record_duration = 1  # seconds
        self.start_time = time.time()

    def listener_callback(self, msg):
        self.buffer.extend(msg.data)
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= self.record_duration:
            self.save_wav()
            self.buffer.clear()
            self.start_time = time.time()

    def save_wav(self):
        filename = f'audio_{int(time.time())}.wav'
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.sample_width)
            wf.setframerate(self.sample_rate)
            wf.writeframes(self.buffer)
        self.get_logger().info(f'Saved {filename}')

def main(args=None):
    rclpy.init(args=args)
    audio_recorder = AudioRecorder()
    rclpy.spin(audio_recorder)
    audio_recorder.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
