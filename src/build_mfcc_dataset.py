import librosa
import numpy as np
from pathlib import Path

# ==============================
# 1. PROJECT PATHS
# ==============================

project_root = Path(__file__).resolve().parent.parent

normal_folder = project_root / "dataset" / "raw" / "normal"
distress_folder = project_root / "dataset" / "raw" / "distress"

processed_folder = project_root / "dataset" / "processed"
processed_folder.mkdir(parents=True, exist_ok=True)


# ==============================
# 2. MFCC SETTINGS
# ==============================

SAMPLE_RATE = 16000
N_MFCC = 40


# ==============================
# 3. FUNCTION TO EXTRACT MFCC
# ==============================

def extract_mfcc(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC
    )

    # Take the average across time
    mfcc_mean = np.mean(mfcc, axis=1)

    return mfcc_mean


# ==============================
# 4. BUILD DATASET
# ==============================

features = []
labels = []

print("Starting MFCC extraction...")
print()


# ==============================
# NORMAL = 0
# ==============================

normal_files = list(normal_folder.glob("*.wav"))

print("Normal files:", len(normal_files))

for i, file_path in enumerate(normal_files):

    try:
        mfcc = extract_mfcc(file_path)

        features.append(mfcc)
        labels.append(0)

        if (i + 1) % 25 == 0:
            print(f"Normal processed: {i + 1}/{len(normal_files)}")

    except Exception as e:
        print("Error:", file_path.name)
        print(e)


# ==============================
# DISTRESS = 1
# ==============================

distress_files = list(distress_folder.glob("*.wav"))

print()
print("Distress files:", len(distress_files))

for i, file_path in enumerate(distress_files):

    try:
        mfcc = extract_mfcc(file_path)

        features.append(mfcc)
        labels.append(1)

        if (i + 1) % 25 == 0:
            print(f"Distress processed: {i + 1}/{len(distress_files)}")

    except Exception as e:
        print("Error:", file_path.name)
        print(e)


# ==============================
# 5. CONVERT TO NUMPY ARRAYS
# ==============================

X = np.array(features)
y = np.array(labels)


# ==============================
# 6. SAVE DATASET
# ==============================

X_path = processed_folder / "X_mfcc.npy"
y_path = processed_folder / "y_labels.npy"

np.save(X_path, X)
np.save(y_path, y)


# ==============================
# 7. SUMMARY
# ==============================

print()
print("====================================")
print("MFCC DATASET CREATED SUCCESSFULLY")
print("====================================")

print("Feature shape:", X.shape)
print("Label shape:", y.shape)

print("Normal samples:", np.sum(y == 0))
print("Distress samples:", np.sum(y == 1))

print()
print("Features saved to:")
print(X_path)

print()
print("Labels saved to:")
print(y_path)