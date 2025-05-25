import rclpy
from rclpy.node import Node
from audio_stream_msgs.msg import AudioStream, AudioFormat
import pyaudio
import numpy as np
from audio_stream_tools.helpers import NP_DTYPE_TO_PYAUDIO
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import SetParametersResult
from std_msgs.msg import Float32

class AudioOutputNode(Node):
    def __init__(self):
        super().__init__('audio_output')
        self.declare_parameter("input_topic", "audio/filtered/bass")
        
        self.add_on_set_parameters_callback(self._input_topic_changed)
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.format = None
        self.create_subscription(AudioFormat,"/audio/format",self._audio_format_cb,10)
        self.gain = 0.8
        
        while self.format is None:
            rclpy.spin_once(self,timeout_sec=2)
            self.get_logger().info("Waiting on format topic..")
        
        self.create_subscription(Float32,"/audio/output_gain",self.gain_cb,10)
        
        self.subscription = self.create_subscription(
            AudioStream,
            self.get_parameter("input_topic").value,
            self.audio_callback,
            10
        )
        self.get_logger().info("AudioOutputNode started. Listening for audio...")

    def gain_cb(self,msg:Float32):
        self.gain = msg.data

    def audio_callback(self, msg: AudioStream):
        audio_bytes = bytes(msg.data)
        # Convert to NumPy array (assuming 16-bit signed int PCM)
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)

        # Apply gain
        audio_np = (audio_np * self.gain).astype(np.int16)

        # Convert back to bytes and write to output stream
        self.stream.write(audio_np.tobytes())

    def destroy_node(self):
        self.get_logger().info("Shutting down audio output.")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        super().destroy_node()

    def _input_topic_changed(self,params):
        for param in params:
            if param.name == 'input_topic':
                self.get_logger().info(f"Changing to {param.value}")
                self.destroy_subscription(self.subscription)
                self.subscription = self.create_subscription(
                    AudioStream,
                    param.value,
                    self.audio_callback,
                    10
                )
        
        return SetParametersResult(successful=True)

    def _audio_format_cb(self, msg: AudioFormat):
        if msg != self.format:
            self.format = msg
            if self.stream:
                self.stream.close()
            self.stream = self.p.open(
                format=NP_DTYPE_TO_PYAUDIO[self.format.dtype],
                channels=self.format.channels,
                rate=self.format.sample_rate,
                output=True,
                frames_per_buffer=self.format.frame_size
            )
    
def main(args=None):
    rclpy.init(args=args)
    node = AudioOutputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
