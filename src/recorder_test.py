import pyaudio
import numpy as np
import torch
import torchaudio

# Parameters
duration = 3  # Seconds to store in buffer
sample_rate = 16000  # Hz
channels = 1  # Mono audio
chunk_size = 1024  # Number of frames per buffer (small chunk for faster processing)

# Create a ring buffer to store 3 seconds of audio
buffer_size = duration * sample_rate
ring_buffer = np.zeros(buffer_size, dtype='float32')

# Initialize PyAudio
p = pyaudio.PyAudio()
print(p.get_host_api_info_by_index(0))

# Define the audio stream
stream = p.open(format=pyaudio.paFloat32,  # 32-bit float format
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk_size)

# Function to save the current buffer as a .wav file
def save_audio(buffer, filename="output.wav"):
    # Convert buffer to a PyTorch tensor
    audio_tensor = torch.tensor(buffer).unsqueeze(0)
    
    # Save the audio as a WAV file using torchaudio
    torchaudio.save(filename, audio_tensor, sample_rate)
    print(f"Audio saved to {filename}")

print("Recording... Press Ctrl+C to stop.")
try:
    while True:
        # Read audio data from the microphone
        audio_data = stream.read(chunk_size)
        
        # Convert the byte data to a numpy array
        audio_np = np.frombuffer(audio_data, dtype='float32')
        
        # Shift the buffer to the left and append the new data
        ring_buffer = np.roll(ring_buffer, -len(audio_np))
        ring_buffer[-len(audio_np):] = audio_np
        
        # Optionally, save the current buffer every few seconds
        save_audio(ring_buffer, "output.wav")

except KeyboardInterrupt:
    print("\nRecording stopped.")
    
finally:
    # Close the stream and terminate PyAudio
    stream.stop_stream()
    stream.close()
    p.terminate()
