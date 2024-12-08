import os
import torchaudio

def convert_wav_to_mp3(folder_path):
    # Ensure the provided path exists
    if not os.path.exists(folder_path):
        print(f"The folder {folder_path} does not exist.")
        return

    # Get all wav files in the folder
    wav_files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    
    if not wav_files:
        print("No WAV files found in the folder.")
        return

    # Create an output directory for mp3 files
    output_folder = os.path.join(folder_path, "mp3_output")
    os.makedirs(output_folder, exist_ok=True)
    
    for wav_file in wav_files:
        wav_path = os.path.join(folder_path, wav_file)
        mp3_path = os.path.join(output_folder, os.path.splitext(wav_file)[0] + ".mp3")
        
        try:
            # Load the WAV file
            waveform, sample_rate = torchaudio.load(wav_path)
            # Save as MP3
            torchaudio.save(mp3_path, waveform, sample_rate, format="mp3")
            print(f"Converted: {wav_file} -> {mp3_path}")
        except Exception as e:
            print(f"Failed to convert {wav_file}: {e}")

if __name__ == "__main__":
    convert_wav_to_mp3("/home/appuser/lumenai/src/segmented")
