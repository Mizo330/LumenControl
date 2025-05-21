import rclpy
from rclpy.node import Node
from audio_stream_msgs.msg import AudioStream, AudioFormat
from audio_stream_msgs.srv import GetAudioFormat
import wave
import numpy as np
import scipy.signal
import pyaudio

class AudioPlaybackNode(Node):
    def __init__(self, wav_path, loop=False):
        super().__init__('audio_playback')
        self.publisher = self.create_publisher(AudioStream, '/audio/input_raw', 10)

        self.loop = loop
        self.wav_path = wav_path


        # Open wave file once
        self.wf = wave.open(self.wav_path, 'rb')
        self.sample_rate = self.wf.getframerate()
        self.channels = self.wf.getnchannels()
        self.sampwidth = self.wf.getsampwidth()
        self.frame_count = 1024 #2^10
        self.dytpe = "int16"
        
        self.format_msg = AudioFormat()
        self.format_msg.sample_rate = self.sample_rate
        self.format_msg.channels = self.channels
        self.format_msg.frame_size = self.frame_count
        self.format_msg.dtype = self.dytpe

        if self.sampwidth != 2:
            self.get_logger().error("Only 16-bit PCM WAV files are supported.")
            raise RuntimeError("Unsupported WAV sample width")

        self.p = pyaudio.PyAudio()
        self.getformat_srv = self.create_service(GetAudioFormat,"/audio/get_format",self.getformat_callback)

        # Timer interval based on playback rate
        self.timer_period = self.frame_count / self.sample_rate
        self.timer = self.create_timer(self.timer_period, self.play_audio)

        self.get_logger().info(
            f"Initialized playback: {self.wav_path} @ {self.sample_rate}Hz, {self.channels} channels"
        )

    def play_audio(self):
        data = self.wf.readframes(self.frame_count)
        if not data:
            if self.loop:
                self.wf.rewind()
                data = self.wf.readframes(self.frame_count)
            else:
                self.get_logger().info("Playback finished.")
                self.timer.cancel()
                self.wf.close()
                rclpy.shutdown()
                return

        # Convert bytes to numpy int16
        audio = np.frombuffer(data, dtype=np.int16)

        # If stereo, reshape accordingly
        if self.channels > 1:
            audio = np.reshape(audio, (-1, self.channels))


        # Flatten and convert to uint8 for publishing
        audio_bytes = audio.astype(np.int16).tobytes()
        msg = AudioStream()
        msg.stamp = self.get_clock().now().to_msg()
        msg.format = self.format_msg
        msg.data = list(audio_bytes)  # still bytes, but packed as list of ints

        self.publisher.publish(msg)

    def getformat_callback(self,request,respone):
        self.get_logger().info("Got request for format")
        respone = GetAudioFormat.Response()
        respone.format = self.format_msg
        return respone
    
def main(args=None):
    rclpy.init(args=args)
    wav_path = "/home/appuser/lumencontrol/src/my_packages/wav/On The Drums.wav"  # replace as needed
    loop = True  
    node = AudioPlaybackNode(wav_path=wav_path, loop=loop)
    rclpy.spin(node)

if __name__ == '__main__':
    main()
