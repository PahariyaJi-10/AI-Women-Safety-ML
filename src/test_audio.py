import librosa
from pathlib import Path

# Find the project root
project_root = Path(__file__).resolve().parent.parent

# Audio file path
file_path = project_root / "dataset" / "raw" / "normal" / "normal_001.wav"

print("Loading:", file_path)

# Load audio
audio, sample_rate = librosa.load(file_path, sr=16000)

print("Audio loaded successfully!")
print("Sample rate:", sample_rate)
print("Number of samples:", len(audio))
print("Duration:", round(len(audio) / sample_rate, 2), "seconds")
print("Minimum amplitude:", audio.min())
print("Maximum amplitude:", audio.max())