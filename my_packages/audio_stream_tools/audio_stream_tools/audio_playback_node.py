import rclpy
from rclpy.node import Node
from audio_stream_msgs.msg import AudioStream, AudioFormat
from audio_stream_msgs.srv import AudioPlaybackControl
import wave
import numpy as np
import scipy.signal
import pyaudio
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter
from audio_stream_tools.helpers import NP_DTYPE_TO_PYAUDIO

class AudioPlaybackNode(Node):
    def __init__(self, wav_path, loop=False):
        super().__init__('audio_playback')
        self.declare_parameter('audio_path',"/home/appuser/lumencontrol/src/my_packages/wav/On The Drums.wav")
        
        self.add_on_set_parameters_callback(self.path_changed_cb)
        
        self.audio_publisher = self.create_publisher(AudioStream, '/audio/input_raw', 10)
        self.format_publisher = self.create_publisher(AudioFormat, '/audio/format',10)
        self.create_service(AudioPlaybackControl,'/audio/playback_control',self.playback_control_cb)
        
        self.loop = loop
        self.timer = None
        
        self.format = AudioFormat()
        self.format.frame_size = 1024 #2^10
        self.format.dtype = "int16"
        self.p = pyaudio.PyAudio()
        
        self.create_timer(1,self.publish_format)
        self.open_new_wf(self.get_parameter('audio_path').value)

        self.get_logger().info("Initialized playbacker.")

    def open_new_wf(self, audio_path):
        if self.timer is not None:
            self.timer.cancel()

        self.wf = wave.open(audio_path, 'rb')
        self.sampwidth = self.wf.getsampwidth()
    
        if self.sampwidth != 2:
            self.get_logger().error("Only 16-bit PCM WAV files are supported.")
            raise RuntimeError("Unsupported WAV sample width")
        
        format_temp = self.format
        format_temp.sample_rate = self.wf.getframerate()
        format_temp.channels = self.wf.getnchannels()
        self.timer_period = format_temp.frame_size / format_temp.sample_rate
        
        self.format = format_temp
        self.publish_format() #make sure the new format is published before change.

        self.timer = self.create_timer(self.timer_period, self.play_audio)
                
        self.get_logger().info(
            f"Opened new stream: {audio_path} @ {self.format.sample_rate}Hz, {self.format.channels} channels"
        )
        
    def play_audio(self):
        data = self.wf.readframes(self.format.frame_size)
        if not data:
            if self.loop:
                self.wf.rewind()
                data = self.wf.readframes(self.format.frame_size)
            else:
                self.get_logger().info("Playback finished.")
                self.timer.cancel()
                self.wf.close()
                rclpy.shutdown()
                return

        # Convert bytes to numpy int16
        audio = np.frombuffer(data, dtype=np.int16)

        # If stereo, reshape accordingly
        if self.format.channels > 1:
            audio = np.reshape(audio, (-1, self.format.channels))

        audio_bytes = audio.astype(self.format.dtype).tobytes()
        msg = AudioStream()
        msg.stamp = self.get_clock().now().to_msg()
        msg.data = list(audio_bytes)  # still bytes, but packed as list of ints

        self.audio_publisher.publish(msg)
    
    def publish_format(self):
        self.format_publisher.publish(self.format)
    
    def path_changed_cb(self,params):
        for param in params:
            if param.name == 'audio_path':
                try:
                    self.open_new_wf(param.value)
                except Exception as e:
                    self.get_logger().info(f"Failed to change track! Error:{e}")
                    return SetParametersResult(successful=False,reason=str(e))
        
        return SetParametersResult(successful=True)
    
    def playback_control_cb(self,request:AudioPlaybackControl.Request,response:AudioPlaybackControl.Response):
        if request.command == "play":
            if  self.timer.is_canceled():
                response.success = True
                self.timer.reset()
            else:
                response.success = True
                response.message = "Already playing."
        elif request.command == "pause":
            if not self.timer:
                response.success = True
                response.message = "Already paused."
            else:
                response.success = True
                self.timer.cancel()
        else:
            response.success == False
            response.message == "Unrecognized command."
        return response
            
    
def main(args=None):
    rclpy.init(args=args)
    wav_path = "/home/appuser/lumencontrol/src/my_packages/wav/On The Drums.wav"  # replace as needed
    loop = True  
    node = AudioPlaybackNode(wav_path=wav_path, loop=loop)
    rclpy.spin(node)

if __name__ == '__main__':
    main()
