import subprocess
import imageio_ffmpeg

input_file = "dataset/raw/normal/normal_001.m4a"
output_file = "dataset/raw/normal/normal_001.wav"

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

command = [
    ffmpeg,
    "-y",
    "-i", input_file,
    "-ar", "16000",
    "-ac", "1",
    output_file
]

subprocess.run(command, check=True)

print("Audio conversion successful!")
print("Created:", output_file)