import rclpy.time
import numpy as np
import pyaudio
import time
import rclpy
from rclpy.node import Node
from audio_stream_msgs.msg import AudioStream, AudioFormat
from audio_stream_msgs.srv import GetAudioFormat
from builtin_interfaces.msg import Time
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange
from audio_stream_tools.helpers import NP_DTYPE_TO_PYAUDIO

class InputRecorder(Node):
    """Simple, cross-platform node to listen from the default input device."""

    def __init__(self):
        super().__init__('audio_input')

        self.sample_rate = 44100
        self.frame_count = 1024 #2^10
        self.dytpe = "int16"
        self.channels = 2

        self.format_msg = AudioFormat()
        self.format_msg.sample_rate = self.sample_rate
        self.format_msg.channels = self.channels
        self.format_msg.frame_size = self.frame_count
        self.format_msg.dtype = self.dytpe
        # channels_descriptor = ParameterDescriptor(
        #     description="Number of audio channels: 1 = mono, 2 = stereo",
        #     integer_range=[IntegerRange(from_value=1, to_value=2, step=1)]
        # )
        # self.declare_parameter('channels', 2, descriptor=channels_descriptor)
        
        self.getformat_srv = self.create_service(GetAudioFormat,"/audio/get_format",self.getformat_callback)
        
        self.audio_publisher = self.create_publisher(AudioStream,"/audio/input_raw",10)
        
        self.p = pyaudio.PyAudio()

        # Use the default input device
        default_input_device_info = self.p.get_default_input_device_info()
        self.get_logger().info(f"Using default input device: {default_input_device_info['name']}")
        self.in_stream = self.p.open(
            format=NP_DTYPE_TO_PYAUDIO["int16"],
            channels=self.channels,
            rate=self.sample_rate,
            input_device_index=default_input_device_info["index"],
            input=True,
            frames_per_buffer=self.frame_count,
            stream_callback=self.pyaudio_callback
        )
        self.get_logger().info("PyAdio steam opened")

    def pyaudio_callback(self, in_data, frame_count, time_info, status):
        msg = AudioStream()
        msg.stamp = rclpy.time.Time(seconds=time_info['input_buffer_adc_time']).to_msg()
        msg.format = self.format_msg
        msg.data = list(in_data)
        
        self.audio_publisher.publish(msg)

        return (None, pyaudio.paContinue)
    
    def getformat_callback(self,request,respone):
        self.get_logger().info("Got request for format")
        respone = GetAudioFormat.Response()
        respone.format = self.format_msg
        return respone
        
    def destroy_node(self):
        if self.in_stream is not None:
            self.in_stream.close()
        if self.p is not None:
            self.p.terminate()
        super().destroy_node()
            
def main(args=None):
    rclpy.init(args=args)
    node = InputRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()