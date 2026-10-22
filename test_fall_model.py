
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# 1. PATHS
# ============================================================

project_root = Path(
    r"C:\Users\Lenovo\AI-Women-Safety-ML"
)

dataset_root = Path(
    r"C:\Users\Lenovo\Downloads\UMAFall_Dataset\UMAFall_Dataset"
)

model_path = project_root / "fall_detection_model.pkl"


# ============================================================
# 2. SELECT TEST RECORDING
# ============================================================

# Choose "FALL" or "ADL"
TEST_TYPE = "FALL"

if TEST_TYPE == "FALL":
    test_files = sorted(
        dataset_root.glob("*[Ff][Aa][Ll][Ll]*.csv")
    )
elif TEST_TYPE == "ADL":
    test_files = sorted(
        dataset_root.glob("*[Aa][Dd][Ll]*.csv")
    )
else:
    raise ValueError(
        "TEST_TYPE must be 'FALL' or 'ADL'."
    )

if not test_files:
    raise FileNotFoundError(
        f"No {TEST_TYPE} recordings found in: {dataset_root}"
    )

test_file = test_files[0]


# ============================================================
# 3. LOAD TRAINED MODEL
# ============================================================

print("=" * 60)
print("FALL DETECTION MODEL TEST")
print("=" * 60)

if not model_path.exists():
    raise FileNotFoundError(
        f"Trained model not found: {model_path}"
    )

model = joblib.load(model_path)

print("\nModel loaded successfully.")
print("Test recording:", test_file.name)


# ============================================================
# 4. DETERMINE ACTUAL RECORDING CLASS
# ============================================================

filename_upper = test_file.name.upper()

if "_FALL_" in filename_upper or "FALL_" in filename_upper:
    actual_class = "FALL"
elif "_ADL_" in filename_upper:
    actual_class = "NORMAL (ADL)"
else:
    actual_class = "UNKNOWN"


# ============================================================
# 5. READ SENSOR DATA
# ============================================================

df = pd.read_csv(
    test_file,
    sep=";",
    header=None,
    comment="%"
)

df = df.dropna(axis=0, how="all")
df = df.dropna(axis=1, how="all")

if df.shape[1] < 7:
    raise ValueError(
        f"Expected at least 7 columns; found {df.shape[1]}."
    )


# ============================================================
# 6. SELECT SENSOR STREAM (0, 0)
# ============================================================

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


# ============================================================
# 7. EXTRACT X, Y, Z ACCELERATION
# ============================================================

acceleration_df = selected.iloc[:, 2:5].copy()

acceleration_df = acceleration_df.apply(
    pd.to_numeric,
    errors="coerce"
)

acceleration_df = acceleration_df.dropna()

acceleration = acceleration_df.to_numpy(
    dtype=np.float32
)

print("\nValid accelerometer samples:", len(acceleration))

if len(acceleration) < 450:
    raise ValueError(
        "Not enough valid samples to create a 450-sample window."
    )


# ============================================================
# 8. CREATE WINDOWS
# ============================================================

WINDOW_SIZE = 450
STEP_SIZE = 50

windows = []

for start in range(
    0,
    len(acceleration) - WINDOW_SIZE + 1,
    STEP_SIZE
):
    window = acceleration[
        start:start + WINDOW_SIZE
    ]

    windows.append(window.reshape(-1))


X_test = np.asarray(
    windows,
    dtype=np.float32
)

print("Windows created:", len(X_test))
print("Input shape:", X_test.shape)


# ============================================================
# 9. PREDICT ALL WINDOWS
# ============================================================

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

fall_column = np.where(model.classes_ == 1)[0]

if len(fall_column) == 0:
    raise ValueError(
        "The loaded model has no FALL class."
    )

fall_probabilities = probabilities[:, fall_column[0]]


# ============================================================
# 10. CALCULATE RESULTS
# ============================================================

fall_count = int(np.sum(predictions == 1))
normal_count = int(np.sum(predictions == 0))

total_windows = len(predictions)

fall_window_percentage = (
    fall_count / total_windows * 100
)

normal_window_percentage = (
    normal_count / total_windows * 100
)


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)

print("Actual recording class:", actual_class)
print("Total windows tested:", total_windows)

print(
    f"Predicted FALL windows: {fall_count} "
    f"({fall_window_percentage:.2f}%)"
)

print(
    f"Predicted NORMAL windows: {normal_count} "
    f"({normal_window_percentage:.2f}%)"
)

print(
    f"Highest fall probability: "
    f"{fall_probabilities.max() * 100:.2f}%"
)

print(
    f"Average fall probability: "
    f"{fall_probabilities.mean() * 100:.2f}%"
)


# ============================================================
# 12. PRELIMINARY INTERPRETATION
# ============================================================

if actual_class == "FALL":
    if fall_count > 0:
        print(
            "\nPreliminary result: At least one window "
            "was predicted as FALL."
        )
    else:
        print(
            "\nPreliminary result: No window was predicted as FALL."
        )

elif actual_class == "NORMAL (ADL)":
    if fall_count == 0:
        print(
            "\nPreliminary result: All windows were predicted "
            "as NORMAL for this ADL recording."
        )
    else:
        print(
            "\nPreliminary result: Potential false-positive "
            "FALL predictions occurred in this ADL recording."
        )

else:
    print(
        "\nPreliminary result: Actual class is unknown."
    )


# ============================================================
# 13. SHOW FIRST 10 WINDOW PREDICTIONS
# ============================================================

print("\nFirst 10 window predictions:")

for i in range(min(10, total_windows)):
    label = (
        "FALL"
        if predictions[i] == 1
        else "NORMAL"
    )

    print(
        f"Window {i + 1}: {label} | "
        f"Fall probability: "
        f"{fall_probabilities[i] * 100:.2f}%"
    )


# ============================================================
# 14. FINISH
# ============================================================

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
