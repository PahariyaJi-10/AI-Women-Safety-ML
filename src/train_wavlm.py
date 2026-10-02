import librosa
import numpy as np
import torch

from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModel

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

project_root = Path(__file__).resolve().parent.parent

# ORIGINAL RAVDESS DATASET
ravdess_root = Path(
    r"C:\Users\Lenovo\Downloads\RAVDESS\audio_speech_actors_01-24"
)

MODEL_NAME = "microsoft/wavlm-base-plus"


# ============================================================
# 2. LOAD WAVLM
# ============================================================

print("=" * 60)
print("Loading WavLM...")
print("=" * 60)

feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_NAME
)

wavlm = AutoModel.from_pretrained(
    MODEL_NAME
)

wavlm.eval()

device = torch.device("cpu")
wavlm.to(device)

print("WavLM loaded successfully.")
print("Device:", device)


# ============================================================
# 3. FUNCTION TO EXTRACT WAVLM EMBEDDING
# ============================================================

def get_embedding(file_path):

    audio, sr = librosa.load(
        file_path,
        sr=16000,
        mono=True
    )

    inputs = feature_extractor(
        audio,
        sampling_rate=16000,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = wavlm(**inputs)

    # Last hidden state
    # Shape approximately:
    # [1, time_steps, 768]

    embedding = outputs.last_hidden_state.mean(
        dim=1
    )

    # Final shape:
    # [768]

    return embedding.squeeze().cpu().numpy()


# ============================================================
# 4. RAVDESS EMOTION MAPPING
# ============================================================

# RAVDESS emotion codes:
#
# 01 = neutral
# 02 = calm
# 03 = happy
# 04 = sad
# 05 = angry
# 06 = fearful
# 07 = disgust
# 08 = surprised
#
# Our project:
#
# NORMAL:
# 01 neutral
# 02 calm
#
# DISTRESS:
# 04 sad
# 05 angry
# 06 fearful
#
# IGNORE:
# 03 happy
# 07 disgust
# 08 surprised


normal_emotions = {
    "01",
    "02"
}

distress_emotions = {
    "04",
    "05",
    "06"
}


# ============================================================
# 5. FIND RAVDESS FILES
# ============================================================

all_files = list(
    ravdess_root.glob("Actor_*/*.wav")
)

print()
print("=" * 60)
print("RAVDESS DATASET")
print("=" * 60)

print("Total WAV files found:", len(all_files))


# ============================================================
# 6. EXTRACT FEATURES
# ============================================================

features = []
labels = []
groups = []

processed = 0
ignored = 0


for file_path in all_files:

    filename = file_path.stem

    parts = filename.split("-")

    # Expected:
    # 03-01-05-01-02-01-12

    if len(parts) != 7:
        print("Skipping invalid file:", file_path.name)
        continue

    emotion_code = parts[2]

    # Actor ID comes from folder
    actor_folder = file_path.parent.name

    # Example:
    # Actor_12

    actor_id = actor_folder.replace(
        "Actor_",
        ""
    )

    # Determine label

    if emotion_code in normal_emotions:

        label = 0

    elif emotion_code in distress_emotions:

        label = 1

    else:

        ignored += 1
        continue


    try:

        embedding = get_embedding(file_path)

        features.append(embedding)
        labels.append(label)
        groups.append(actor_id)

        processed += 1

        if processed % 20 == 0:

            print(
                f"Processed: {processed}"
            )

    except Exception as e:

        print(
            "Error:",
            file_path.name
        )

        print(e)


# ============================================================
# 7. CONVERT TO NUMPY
# ============================================================

X = np.array(features)

y = np.array(labels)

groups = np.array(groups)


print()
print("=" * 60)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 60)

print("Feature shape:", X.shape)

print("Labels shape:", y.shape)

print("Actors:", len(np.unique(groups)))

print("Ignored files:", ignored)

print("Normal samples:", np.sum(y == 0))

print("Distress samples:", np.sum(y == 1))


# ============================================================
# 8. SPEAKER-INDEPENDENT TRAIN/TEST SPLIT
# ============================================================

print()
print("=" * 60)
print("CREATING SPEAKER-INDEPENDENT SPLIT")
print("=" * 60)


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

groups_train = groups[train_idx]
groups_test = groups[test_idx]


print("Training samples:", len(X_train))

print("Testing samples:", len(X_test))

print(
    "Training actors:",
    sorted(np.unique(groups_train))
)

print(
    "Testing actors:",
    sorted(np.unique(groups_test))
)


# ============================================================
# 9. CHECK ACTOR OVERLAP
# ============================================================

overlap = set(
    groups_train
).intersection(
    set(groups_test)
)


print()

if len(overlap) == 0:

    print(
        "SUCCESS: No actor overlap between train and test."
    )

else:

    print(
        "WARNING: Actor overlap found:",
        overlap
    )


# ============================================================
# 10. CHECK CLASS DISTRIBUTION
# ============================================================

print()
print("Training Normal:",
      np.sum(y_train == 0))

print("Training Distress:",
      np.sum(y_train == 1))

print("Testing Normal:",
      np.sum(y_test == 0))

print("Testing Distress:",
      np.sum(y_test == 1))


# ============================================================
# 11. STANDARD SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 12. LOGISTIC REGRESSION
# ============================================================

print()
print("=" * 60)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 60)


classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)


classifier.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# 13. PREDICTION
# ============================================================

y_pred = classifier.predict(
    X_test_scaled
)


# ============================================================
# 14. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print()
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)


print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)


print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Distress"
        ]
    )
)


print("Confusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 15. FINAL INFORMATION
# ============================================================

print()
print("=" * 60)
print("MODEL PIPELINE")
print("=" * 60)

print(
    "Audio"
    " -> 16 kHz"
    " -> WavLM Base+"
    " -> 768-D Embedding"
    " -> StandardScaler"
    " -> Logistic Regression"
    " -> Normal / Distress"
)

print()
print("Speaker-independent evaluation completed.")