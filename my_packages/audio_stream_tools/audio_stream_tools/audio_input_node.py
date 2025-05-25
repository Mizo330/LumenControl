import rclpy.time
import numpy as np
import pyaudio
import time
import rclpy
from rclpy.node import Node
from audio_stream_msgs.msg import AudioStream, AudioFormat
from audio_stream_msgs.srv import AudioPlaybackControl
from builtin_interfaces.msg import Time
from rcl_interfaces.msg import ParameterDescriptor, IntegerRange
from audio_stream_tools.helpers import NP_DTYPE_TO_PYAUDIO

class InputRecorder(Node):
    """Simple, cross-platform node to listen from the default input device."""

    def __init__(self):
        super().__init__('audio_input')

        self.format = AudioFormat()
        self.format.sample_rate = 41000
        self.format.channels = 2
        self.format.frame_size = 1024
        self.format.dtype = "int16"
        
        self.format_publisher = self.create_publisher(AudioFormat, '/audio/format',10)        
        self.audio_publisher = self.create_publisher(AudioStream,"/audio/input_raw",10)
        self.create_service(AudioPlaybackControl,'/audio/playback_control',self.playback_control_cb)

        self.p = pyaudio.PyAudio()

        # Use the default input device
        default_input_device_info = self.p.get_default_input_device_info()
        self.get_logger().info(f"Using default input device: {default_input_device_info['name']}")
        self.in_stream = self.p.open(
            format=NP_DTYPE_TO_PYAUDIO["int16"],
            channels=self.format.channels,
            rate=self.format.sample_rate,
            input_device_index=default_input_device_info["index"],
            input=True,
            frames_per_buffer=self.format.frame_size,
            stream_callback=self.pyaudio_callback
        )
        self.publish_format()
        self.create_timer(1,self.publish_format)

        self.get_logger().info("PyAdio steam opened")

    def pyaudio_callback(self, in_data, frame_count, time_info, status):
        msg = AudioStream()
        msg.stamp = rclpy.time.Time(seconds=time_info['input_buffer_adc_time']).to_msg()
        msg.data = list(in_data)
        self.audio_publisher.publish(msg)

        return (None, pyaudio.paContinue)
    
    def publish_format(self):
        self.format_publisher.publish(self.format)
        
    def destroy_node(self):
        if self.in_stream is not None:
            self.in_stream.close()
        if self.p is not None:
            self.p.terminate()
        super().destroy_node()
            
    def playback_control_cb(self,request:AudioPlaybackControl.Request,response:AudioPlaybackControl.Response):
        if request.command == "play":
            if self.in_stream.is_stopped():
                self.in_stream.start_stream()
            response.success = True
        elif request.command == "pause":
            if self.in_stream.is_active():
                self.in_stream.stop_stream()
            response.success = True
        else:
            response.success == False
            response.message == "Unrecognized command."
        return response        
            
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