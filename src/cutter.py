import torchaudio
from beat_this.inference import Audio2Beats
from allin1 import analyze

analyze()
# # Specify the path to your WAV file
wav_file_path = "/home/appuser/lumenai/src/wav/Dimension, Sub Focus & NGHTMRE - Angel (ft. Mougleta).wav"

# # Load the audio file
waveform, sample_rate = torchaudio.load(wav_file_path)

# # Define the start and end times for the segment (in seconds)
start_time = 60  # Start 
end_time = 140  # End 

# # Convert times to sample indices
start_sample = int(start_time * sample_rate)
end_sample = int(end_time * sample_rate)

# # Cut the segment
segment = waveform[:, start_sample:end_sample]

# # Print details about the segment
print(f"Segment shape: {segment.shape}")
print(f"Duration: {(end_sample - start_sample) / sample_rate} seconds")

savepath = "segment_angel.wav"
# # Optionally, save the segment to a new file
torchaudio.save(savepath, segment, sample_rate)

asd = File2File("/home/appuser/lumenai/src/final0.ckpt","cuda")
asd(savepath,"/home/appuser/lumenai/asd.beat")
