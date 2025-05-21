import rclpy
from rclpy.node import Node
from audio_stream_msgs.msg import AudioStream, AudioFormat
from audio_stream_msgs.srv import GetAudioFormat
import pyaudio
import numpy as np
from audio_stream_tools.helpers import NP_DTYPE_TO_PYAUDIO

class AudioOutputNode(Node):
    def __init__(self):
        super().__init__('audio_output_node')

        self.get_logger().info("Waiting for input format service..")        
        self.format_client = self.create_client(GetAudioFormat,"/audio/get_format")
        while not self.format_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /audio/get_format service...')
        
        format = self.get_audio_format()    
        
        self.p = pyaudio.PyAudio()
        
        self.stream = self.p.open(
            format=NP_DTYPE_TO_PYAUDIO[format.dtype],
            channels=format.channels,
            rate=format.sample_rate,
            output=True,
            frames_per_buffer=format.frame_size
        )

        self.subscription = self.create_subscription(
            AudioStream,
            '/demixed_audio',
            self.callback,
            10
        )
        self.get_logger().info("AudioOutputNode started. Listening for audio...")

    def callback(self, msg: AudioStream):
        # Convert list of uint8 to bytes
        audio_bytes = bytes(msg.data)

        # Play audio to output stream
        self.stream.write(audio_bytes)

    def destroy_node(self):
        self.get_logger().info("Shutting down audio output.")
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        super().destroy_node()

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
