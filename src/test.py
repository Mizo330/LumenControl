import pyaudio

p = pyaudio.PyAudio()
print(p.get_device_count())
for i in range(p.get_device_count()):
    print(p.get_device_info_by_index(i))