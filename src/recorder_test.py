import pyaudio
import wave
import numpy as np

def record_audio(filename, duration, gain=1.0, sample_rate=44100, channels=1):
    # Initialize PyAudio
    audio = pyaudio.PyAudio()

    # Open stream
    stream = audio.open(format=pyaudio.paInt16,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=1024)

    print("Recording...")

    frames = []

    for _ in range(int(sample_rate / 1024 * duration)):
        data = stream.read(1024)
        
        # Convert byte data to numpy array
        audio_data = np.frombuffer(data, dtype=np.int16)
        
        # Apply gain
        audio_data = audio_data * gain
        
        # Clip the values to ensure they remain within valid range
        audio_data = np.clip(audio_data, -32768, 32767).astype(np.int16)

        # Convert back to bytes and store in frames
        frames.append(audio_data.tobytes())

    print("Recording finished.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))

# Usage example
gain_value = 1.0  # Set gain value (e.g., 2.0 for doubling the volume)
record_audio("output.wav", duration=5, gain=gain_value)  # Record for 5 seconds
