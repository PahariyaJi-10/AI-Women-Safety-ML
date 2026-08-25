import librosa
import numpy as np
import torch

from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. PATHS
# ==========================================

project_root = Path(__file__).resolve().parent.parent

normal_folder = project_root / "dataset" / "raw" / "normal"
distress_folder = project_root / "dataset" / "raw" / "distress"

MODEL_NAME = "microsoft/wavlm-base-plus"

# ==========================================
# 2. LOAD WAVLM
# ==========================================

print("Loading WavLM...")

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
wavlm = AutoModel.from_pretrained(MODEL_NAME)

wavlm.eval()

device = torch.device("cpu")
wavlm.to(device)

print("WavLM loaded!")
print("Device:", device)

# ==========================================
# 3. FUNCTION TO EXTRACT WAVLM EMBEDDING
# ==========================================

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

    # Average across time
    embedding = outputs.last_hidden_state.mean(dim=1)

    return embedding.squeeze().cpu().numpy()


# ==========================================
# 4. LOAD FILES
# ==========================================

normal_files = list(normal_folder.glob("*.wav"))
distress_files = list(distress_folder.glob("*.wav"))

print()
print("Normal files:", len(normal_files))
print("Distress files:", len(distress_files))


# ==========================================
# 5. EXTRACT FEATURES
# ==========================================

features = []
labels = []

print()
print("Extracting WavLM features...")
print()

# NORMAL = 0

for i, file_path in enumerate(normal_files):

    try:

        embedding = get_embedding(file_path)

        features.append(embedding)
        labels.append(0)

        if (i + 1) % 10 == 0:
            print(
                f"Normal processed: "
                f"{i + 1}/{len(normal_files)}"
            )

    except Exception as e:

        print("Error:", file_path.name)
        print(e)


# DISTRESS = 1

for i, file_path in enumerate(distress_files):

    try:

        embedding = get_embedding(file_path)

        features.append(embedding)
        labels.append(1)

        if (i + 10) % 10 == 0:
            print(
                f"Distress processed: "
                f"{i + 1}/{len(distress_files)}"
            )

    except Exception as e:

        print("Error:", file_path.name)
        print(e)


# ==========================================
# 6. CONVERT TO NUMPY
# ==========================================

X = np.array(features)
y = np.array(labels)

print()
print("Feature extraction complete!")
print("Feature shape:", X.shape)
print("Label shape:", y.shape)


# ==========================================
# 7. TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 8. STANDARDIZE FEATURES
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 9. CLASSIFIER
# ==========================================

print()
print("Training WavLM classifier...")

classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

classifier.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 10. PREDICTION
# ==========================================

y_pred = classifier.predict(X_test)


# ==========================================
# 11. EVALUATION
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("==========================================")
print("WAVLM MODEL EVALUATION")
print("==========================================")

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