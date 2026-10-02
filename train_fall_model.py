import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# 1. PATHS
# ============================================================

project_root = Path(
    r"C:\Users\Lenovo\AI-Women-Safety-ML"
)

dataset_folder = (
    project_root
    / "dataset"
    / "accelerometer"
)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 60)
print("FALL DETECTION ML MODEL")
print("=" * 60)

print("\nLoading dataset...")

X = np.load(
    dataset_folder / "X_umafall.npy"
)

y = np.load(
    dataset_folder / "y_umafall.npy"
)

groups = np.load(
    dataset_folder / "groups_umafall.npy"
)

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Groups shape:", groups.shape)


# ============================================================
# 3. CHECK DATA
# ============================================================

print("\nNumber of subjects:",
      len(np.unique(groups)))

print("Normal samples:",
      np.sum(y == 0))

print("Fall samples:",
      np.sum(y == 1))


# ============================================================
# 4. SUBJECT-INDEPENDENT SPLIT
# ============================================================

print("\nCreating subject-independent train/test split...")

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


# ============================================================
# 5. DISPLAY SPLIT INFORMATION
# ============================================================

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Training subjects:",
    np.unique(groups_train)
)

print(
    "Testing subjects:",
    np.unique(groups_test)
)


# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# 7. PREDICTION
# ============================================================

print("\nMaking predictions...")

y_pred = model.predict(X_test)


# ============================================================
# 8. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print()
print("=" * 60)
print("MODEL RESULTS")
print("=" * 60)

print(
    f"Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "NORMAL",
            "FALL"
        ]
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("Confusion Matrix:")

print(cm)


# ============================================================
# 11. SAVE MODEL
# ============================================================

import joblib

model_path = (
    project_root
    / "fall_detection_model.pkl"
)

joblib.dump(
    model,
    model_path
)

print()
print("Model saved at:")

print(model_path)

print()
print("=" * 60)
print("FALL DETECTION TRAINING COMPLETE")
print("=" * 60)