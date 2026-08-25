from pathlib import Path
import shutil

# ==============================
# 1. PATHS
# ==============================

# Our project
project_root = Path(__file__).resolve().parent.parent

# RAVDESS dataset downloaded from Kaggle
ravdess_root = Path(
    r"C:\Users\Lenovo\Downloads\RAVDESS\audio_speech_actors_01-24"
)

# Our project dataset folders
normal_folder = project_root / "dataset" / "raw" / "normal"
distress_folder = project_root / "dataset" / "raw" / "distress"

# Create folders if they don't exist
normal_folder.mkdir(parents=True, exist_ok=True)
distress_folder.mkdir(parents=True, exist_ok=True)


# ==============================
# 2. RAVDESS EMOTION CODES
# ==============================

emotion_codes = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}


# ==============================
# 3. OUR CLASSIFICATION
# ==============================

# Normal / safe speech
normal_emotions = {
    "neutral",
    "calm"
}

# Distress-related speech
distress_emotions = {
    "sad",
    "angry",
    "fearful"
}


# ==============================
# 4. FIND ALL AUDIO FILES
# ==============================

audio_files = list(ravdess_root.rglob("*.wav"))

print("Total RAVDESS WAV files found:", len(audio_files))


# ==============================
# 5. COPY AND LABEL FILES
# ==============================

normal_count = 0
distress_count = 0
ignored_count = 0

for audio_file in audio_files:

    # Example RAVDESS filename:
    # 03-01-05-01-02-01-12.wav

    parts = audio_file.stem.split("-")

    if len(parts) != 7:
        ignored_count += 1
        continue

    emotion_code = parts[2]

    emotion = emotion_codes.get(emotion_code)

    if emotion is None:
        ignored_count += 1
        continue

    # --------------------------
    # NORMAL
    # --------------------------

    if emotion in normal_emotions:

        normal_count += 1

        destination = normal_folder / f"normal_{normal_count:04d}.wav"

        shutil.copy2(audio_file, destination)

    # --------------------------
    # DISTRESS
    # --------------------------

    elif emotion in distress_emotions:

        distress_count += 1

        destination = distress_folder / f"distress_{distress_count:04d}.wav"

        shutil.copy2(audio_file, destination)

    else:
        ignored_count += 1


# ==============================
# 6. RESULTS
# ==============================

print()
print("========== DATASET PREPARATION COMPLETE ==========")
print("Normal files:", normal_count)
print("Distress files:", distress_count)
print("Ignored files:", ignored_count)

print()
print("Normal folder:")
print(normal_folder)

print()
print("Distress folder:")
print(distress_folder)