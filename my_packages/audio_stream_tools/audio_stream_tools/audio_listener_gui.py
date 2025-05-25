import sys
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox, QHBoxLayout, QSlider)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QObject
import pyqtgraph as pg
import numpy as np

from audio_stream_msgs.msg import AudioStream, AudioFormat
from audio_stream_msgs.srv import AudioPlaybackControl
from rcl_interfaces.srv import SetParameters
from rclpy.topic_endpoint_info import TopicEndpointInfo
from std_msgs.msg import Float32
class RosWorker(QThread):
    
    topics_updated = pyqtSignal(list)
    audio_signal = pyqtSignal(AudioStream)
    
    def __init__(self):
        super().__init__()
        rclpy.init()
        self.node = Node("audio_gui")
        self.available_topics = []
        self.subscriber = None
        self._running = True
        self.playback_srv = self.node.create_client(AudioPlaybackControl,'/audio/playback_control')
        self.volume_pub = self.node.create_publisher(Float32,"/audio/output_gain",10)
        self.node.get_logger().info("Wating for input node.")
        self.playback_srv.wait_for_service()

        self.audio_node_name = self.get_input_node_name()
        self.node.create_timer(1,self.update_topic_list)

    def update_topic_list(self):
        topics = self.node.get_topic_names_and_types()
        filtered = [name for name, types in topics if 'audio_stream_msgs/msg/AudioStream' in types] 
        self.topics_updated.emit(filtered)

    def get_input_node_name(self):
        try:
            infos = self.node.get_publishers_info_by_topic("/audio/input_raw")
            return infos[0].node_name
        except Exception:
            return None        
        
    def subscribe_to_audio(self, topic):
        if self.subscriber:
            self.node.destroy_subscription(self.subscriber)
        self.subscriber = self.node.create_subscription(
            AudioStream,
            topic,
            self.audio_callback,
            10)

    def publish_gain(self,gain):
        msg = Float32(data=gain)
        self.volume_pub.publish(msg)

    def audio_callback(self, msg:AudioStream):
        # Simulate audio signal (replace with actual msg.data)
        self.audio_signal.emit(msg)

    def set_parameters_via_service(self, node_name, parameters):
        client = self.node.create_client(SetParameters, f"/{node_name}/set_parameters")
        if not client.wait_for_service(timeout_sec=2.0):
            self.node.get_logger().error(f"Service {node_name}/set_parameters not available")
            return

        request = SetParameters.Request()
        request.parameters = parameters

        client.call_async(request) #handle this somehow
    
    def play_pause_cb(self,command):
        self.node.get_logger().info(f"Sending {command}")
        req = AudioPlaybackControl.Request(command=command)
        self.playback_srv.call_async(req)
        
    def shutdown(self):
        self._running = False

    def run(self):
        #====EXECUTOR====
        executor = rclpy.get_global_executor()
        executor.add_node(self.node)
        while rclpy.ok() and self._running:
            executor.spin_once()
        self.node.destroy_node()
        rclpy.shutdown()


class AudioSelectorApp(QMainWindow):
    control_pressed = pyqtSignal(str)
    volumeChanged = pyqtSignal(float)
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio Stream Selector")
        self.input_topic = " "

        self.worker = RosWorker()
        self.init_ui()
        self.worker.start()
        self.worker.topics_updated.connect(self.update_topic_dropdown)
        self.worker.audio_signal.connect(self.update_plot)
        self.worker.update_topic_list()
        self.control_pressed.connect(self.worker.play_pause_cb)
        self.volumeChanged.connect(self.worker.publish_gain)

    def init_ui(self):
        central = QWidget()
        layout = QVBoxLayout()

        self.file_label = QLabel("Selected File:")
        self.file_edit = QLineEdit()
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)

        self.topic_label = QLabel("AudioStream Topic:")
        self.topic_dropdown = QComboBox()
        self.topic_dropdown.currentIndexChanged.connect(self.topic_changed)

        self.play_button = QPushButton('Play')
        self.stop_button = QPushButton('Pause')
        self.play_button.clicked.connect(lambda: self.control_pressed.emit("play"))
        self.stop_button.clicked.connect(lambda: self.control_pressed.emit("pause"))
        playpauselayout = QHBoxLayout()
        playpauselayout.addWidget(self.play_button)
        playpauselayout.addWidget(self.stop_button)
        
        self.output_label = QLabel("Current Output:")
        self.plot_widget = pg.PlotWidget()
        self.plot_data = self.plot_widget.plot([], pen=pg.mkPen(color='c', width=2))
        self.plot_widget.setYRange(-1, 1, padding=0.1)


        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)  # 0..100 slider steps
        self.volume_slider.setValue(100)     # default max volume

        self.volume_label = QLabel("Volume: 100%")

        volume_layout = QVBoxLayout()
        volume_layout.addWidget(self.volume_slider)
        volume_layout.addWidget(self.volume_label)

        self.volume_slider.valueChanged.connect(self._on_slider_change)

        layout.addWidget(self.file_label)
        layout.addWidget(self.file_edit)
        layout.addWidget(self.file_button)
        layout.addWidget(self.topic_label)
        layout.addWidget(self.topic_dropdown)
        layout.addLayout(playpauselayout)
        layout.addLayout(volume_layout)
        layout.addWidget(self.output_label)
        layout.addWidget(self.plot_widget)

        central.setLayout(layout)
        self.setCentralWidget(central)

    def select_file(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Select Input File")
        if filename:
            self.file_edit.setText(filename)
            self.set_parameter('playbacker', 'audio_path', filename)

    def topic_changed(self):
        topic = self.topic_dropdown.currentText()
        if topic:
            if topic != self.input_topic:
                self.set_parameter('output', 'input_topic', topic)
                self.worker.subscribe_to_audio(topic)
                self.input_topic = topic

    def update_topic_dropdown(self, topics):
        current_text = self.topic_dropdown.currentText()

        self.topic_dropdown.blockSignals(True)  # Optional: prevent signals during update
        self.topic_dropdown.clear()
        self.topic_dropdown.addItems(topics)
        self.topic_dropdown.blockSignals(False)

        # Restore selection if still available
        if current_text in topics:
            index = topics.index(current_text)
            self.topic_dropdown.setCurrentIndex(index)
        else:
            self.topic_dropdown.setCurrentIndex(0 if topics else -1)  # Default to first item if available


    def set_parameter(self, node_name, name, value):
        param = Parameter(name=name, value=value)
        self.worker.set_parameters_via_service(node_name, [param.to_parameter_msg()])

    def update_plot(self, stream:AudioStream):
        samples = np.frombuffer(stream.data, dtype=np.int16).reshape(-1, 2).mean(axis=1)
        samples = samples.astype(np.float32) / 32768.0
        self.plot_data.setData(samples)
    
    def update_current_audio_node(self,node_name:str):
        self.input_node_name = node_name

    def closeEvent(self, event):
        self.worker.shutdown()
        self.worker.wait()
        super().closeEvent(event)

    def _on_slider_change(self, pos: int):
        # Map slider position [0..100] to volume [0..1] logarithmically (dB scale)
        # volume = 10^(dB/20), let's define dB range from -60 dB (mute) to 0 dB (max)
        # min_db = -60.0
        # max_db = 0.0

        norm_pos = pos / 100
        #db = min_db + (max_db - min_db) * norm_pos

        # # Convert dB to linear volume scale [0..1]
        # volume = 10 ** (db / 20)
        # # Clamp volume between 0 and 1 (should already be in range)
        # volume = max(0.0, min(1.0, volume))

        self.volume_label.setText(f"Volume: {int(norm_pos * 100)}%")
        self.volumeChanged.emit(norm_pos)
def main():
    app = QApplication(sys.argv)
    window = AudioSelectorApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
