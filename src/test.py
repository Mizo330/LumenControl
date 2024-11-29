import torchaudio
from beat_this.inference import File2File



# # Specify the path to your WAV file
wav_file_path = "/home/appuser/lumenai/rockit.wav"

# # Load the audio file
waveform, sample_rate = torchaudio.load(wav_file_path)

# # Define the start and end times for the segment (in seconds)
start_time = 110  # Start at 5 seconds
end_time = 115  # End at 10 seconds

# # Convert times to sample indices
start_sample = int(start_time * sample_rate)
end_sample = int(end_time * sample_rate)

# # Cut the segment
segment = waveform[:, start_sample:end_sample]

# # Print details about the segment
print(f"Segment shape: {segment.shape}")
print(f"Duration: {(end_sample - start_sample) / sample_rate} seconds")

# # Optionally, save the segment to a new file
torchaudio.save("segment5.wav", segment, sample_rate)

asd = File2File("/home/appuser/lumenai/src/final0.ckpt","cuda")
asd("/home/appuser/lumenai/src/segment5.wav","/home/appuser/lumenai/beats_5.beat")
