import librosa
import librosa.display
import matplotlib.pyplot as plt
from pathlib import Path

# Find project root
project_root = Path(__file__).resolve().parent.parent

# Audio file
file_path = project_root / "dataset" / "raw" / "normal" / "normal_001.wav"

# Load audio
audio, sample_rate = librosa.load(file_path, sr=16000)

# Extract 40 MFCC features
mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sample_rate,
    n_mfcc=40
)

print("MFCC extraction successful!")
print("MFCC shape:", mfcc.shape)
print("Number of MFCC coefficients:", mfcc.shape[0])
print("Number of time frames:", mfcc.shape[1])

# Create MFCC visualization
plt.figure(figsize=(10, 4))

librosa.display.specshow(
    mfcc,
    x_axis="time",
    sr=sample_rate
)

plt.colorbar()
plt.title("MFCC - Normal Audio")
plt.tight_layout()

# Save result
result_path = project_root / "results" / "normal_001_mfcc.png"
plt.savefig(result_path)

print("MFCC image saved to:", result_path)

plt.show()