
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from scipy.stats import skew, kurtosis

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    recall_score
)


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

project_root = Path(
    r"C:\Users\Lenovo\AI-Women-Safety-ML"
)

dataset_root = Path(
    r"C:\Users\Lenovo\Downloads\UMAFall_Dataset\UMAFall_Dataset"
)

model_path = project_root / "fall_detection_model_v2.pkl"

WINDOW_SIZE = 450
STEP_SIZE = 50

print("=" * 65)
print("FALL DETECTION MODEL V2 - TRAINING")
print("=" * 65)


# ============================================================
# 2. EXTRACT FEATURES FROM ONE WINDOW
# ============================================================

def extract_features(window):
    """
    Input: 450 samples with X, Y, Z measurements.
    Output: 38 numerical motion features.
    """

    features = []

    for axis in range(3):
        values = window[:, axis]

        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            np.ptp(values),
            np.sqrt(np.mean(values ** 2)),
            np.mean(np.abs(np.diff(values))),
            np.std(np.diff(values)),
            skew(values),
            kurtosis(values)
        ])

    magnitude = np.sqrt(
        np.sum(window ** 2, axis=1)
    )

    features.extend([
        np.mean(magnitude),
        np.std(magnitude),
        np.min(magnitude),
        np.max(magnitude),
        np.ptp(magnitude),
        np.sqrt(np.mean(magnitude ** 2)),
        np.mean(np.abs(np.diff(magnitude))),
        np.std(np.diff(magnitude))
    ])

    return np.nan_to_num(
        np.asarray(features, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


# ============================================================
# 3. DETERMINE RECORDING LABEL
# ============================================================

def get_recording_label(file_path):
    """
    Return 1 for FALL and 0 for ADL.
    Return None if the filename does not identify either class.
    """

    filename = file_path.stem.upper()

    # Match FALL as a filename component, regardless of case.
    if "_FALL_" in filename or filename.endswith("_FALL"):
        return 1

    # Match normal Activities of Daily Living.
    if "_ADL_" in filename or filename.endswith("_ADL"):
        return 0

    return None


# ============================================================
# 4. READ AND PREPROCESS A RECORDING
# ============================================================

def load_recording(file_path):

    df = pd.read_csv(
        file_path,
        sep=";",
        header=None,
        comment="%",
        low_memory=False
    )

    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 7:
        raise ValueError(
            f"Expected at least 7 columns; found {df.shape[1]}"
        )

    # Keep the same sensor filter as the previous pipeline.
    sensor_5 = pd.to_numeric(
        df.iloc[:, 5],
        errors="coerce"
    )

    sensor_6 = pd.to_numeric(
        df.iloc[:, 6],
        errors="coerce"
    )

    selected = df[
        (sensor_5 == 0) &
        (sensor_6 == 0)
    ]

    acceleration_df = selected.iloc[:, 2:5].copy()

    acceleration_df = acceleration_df.apply(
        pd.to_numeric,
        errors="coerce"
    ).dropna()

    return acceleration_df.to_numpy(
        dtype=np.float32
    )


# ============================================================
# 5. PROCESS ALL CSV RECORDINGS
# ============================================================

X_features = []
y_labels = []
groups = []

csv_files = sorted(dataset_root.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in: {dataset_root}"
    )

processed_files = 0
skipped_files = 0
fall_files = 0
adl_files = 0

print("\nCSV files found:", len(csv_files))

for file_number, file_path in enumerate(csv_files, start=1):

    label = get_recording_label(file_path)

    if label is None:
        skipped_files += 1
        print(
            "Skipping unrecognized filename:",
            file_path.name
        )
        continue

    try:
        acceleration = load_recording(file_path)

        if len(acceleration) < WINDOW_SIZE:
            skipped_files += 1
            print(
                "Skipping short recording:",
                file_path.name
            )
            continue

        # Subject ID is the number after "Subject_".
        parts = file_path.stem.split("_")

        subject_index = next(
            (
                i for i, part in enumerate(parts)
                if part.upper() == "SUBJECT"
            ),
            None
        )

        if subject_index is None or subject_index + 1 >= len(parts):
            skipped_files += 1
            print(
                "Could not identify subject:",
                file_path.name
            )
            continue

        try:
            subject_id = int(parts[subject_index + 1])
        except ValueError:
            skipped_files += 1
            print(
                "Invalid subject ID:",
                file_path.name
            )
            continue

        # Create overlapping windows.
        for start in range(
            0,
            len(acceleration) - WINDOW_SIZE + 1,
            STEP_SIZE
        ):
            window = acceleration[
                start:start + WINDOW_SIZE
            ]

            features = extract_features(window)

            X_features.append(features)
            y_labels.append(label)
            groups.append(subject_id)

        processed_files += 1

        if label == 1:
            fall_files += 1
        else:
            adl_files += 1

        if file_number % 50 == 0:
            print(
                f"Processed {file_number}/{len(csv_files)} files..."
            )

    except Exception as error:
        skipped_files += 1
        print(
            "Error processing:",
            file_path.name,
            "|",
            error
        )


# ============================================================
# 6. CONVERT DATA TO NUMPY ARRAYS
# ============================================================

X = np.asarray(
    X_features,
    dtype=np.float32
)

y = np.asarray(
    y_labels,
    dtype=np.int32
)

groups = np.asarray(
    groups,
    dtype=np.int32
)

print("\n" + "=" * 65)
print("DATASET SUMMARY")
print("=" * 65)

print("Processed recordings:", processed_files)
print("FALL recordings:", fall_files)
print("ADL recordings:", adl_files)
print("Skipped recordings:", skipped_files)
print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)
print("Groups shape:", groups.shape)
print("Unique subjects:", np.unique(groups))

if len(y) > 0:
    print("NORMAL windows (0):", int(np.sum(y == 0)))
    print("FALL windows (1):", int(np.sum(y == 1)))
    print("Unique labels:", np.unique(y))

if len(X) == 0:
    raise ValueError(
        "No windows were created. Check the dataset and sensor filter."
    )

if len(np.unique(y)) != 2:
    raise ValueError(
        "Both NORMAL and FALL labels are required. "
        "Check the filename patterns and label counts above."
    )

if len(np.unique(groups)) < 2:
    raise ValueError(
        "At least two subjects are required for evaluation."
    )


# ============================================================
# 7. SPLIT DATA BY SUBJECT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    splitter.split(X, y, groups=groups)
)

X_train = X[train_idx]
X_test = X[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

train_subjects = np.unique(groups[train_idx])
test_subjects = np.unique(groups[test_idx])

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training subjects:", train_subjects)
print("Testing subjects:", test_subjects)

if len(np.unique(y_train)) != 2:
    raise ValueError(
        "Training split lacks one class. Try a different random_state."
    )

if len(np.unique(y_test)) != 2:
    raise ValueError(
        "Testing split lacks one class. Try a different random_state."
    )


# ============================================================
# 8. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest model...")

model.fit(X_train, y_train)


# ============================================================
# 9. EVALUATE MODEL
# ============================================================

predictions = model.predict(X_test)

print("\n" + "=" * 65)
print("MODEL EVALUATION")
print("=" * 65)

accuracy = accuracy_score(
    y_test,
    predictions
)

fall_recall = recall_score(
    y_test,
    predictions,
    pos_label=1,
    zero_division=0
)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Fall recall: {fall_recall * 100:.2f}%")

print("\nClassification report:")

print(
    classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["NORMAL", "FALL"],
        zero_division=0
    )
)

print("Confusion matrix (rows = actual, columns = predicted):")
print(
    confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )
)


# ============================================================
# 10. SAVE MODEL
# ============================================================

joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully:")
print(model_path)

print("\nTRAINING COMPLETE")
print("=" * 65)
