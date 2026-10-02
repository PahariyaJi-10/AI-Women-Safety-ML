import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. DATASET PATH
# ============================================================

dataset_root = Path(
    r"C:\Users\Lenovo\Downloads\UMAFall_Dataset\UMAFall_Dataset"
)


# ============================================================
# 2. OUTPUT PATH
# ============================================================

output_folder = Path(
    r"C:\Users\Lenovo\AI-Women-Safety-ML\dataset\accelerometer"
)

output_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. SETTINGS
# ============================================================

WINDOW_SIZE = 450
STEP_SIZE = 50

X_windows = []
y_labels = []

processed_files = 0
skipped_files = 0


# ============================================================
# 4. FIND CSV FILES
# ============================================================

csv_files = sorted(dataset_root.glob("*.csv"))

print("=" * 60)
print("UMAFall DATASET PREPARATION")
print("=" * 60)

print("Total CSV files:", len(csv_files))


# ============================================================
# 5. PROCESS FILES
# ============================================================

for count, file_path in enumerate(csv_files, start=1):

    try:

        # ----------------------------------------------------
        # Read file
        # ----------------------------------------------------

        df = pd.read_csv(
            file_path,
            sep=";",
            header=None,
            comment="%"
        )

        # ----------------------------------------------------
        # Remove completely empty columns
        # ----------------------------------------------------

        df = df.dropna(axis=1, how="all")

        # ----------------------------------------------------
        # Remove completely empty rows
        # ----------------------------------------------------

        df = df.dropna(axis=0, how="all")

        # ----------------------------------------------------
        # We need at least 7 columns
        # ----------------------------------------------------

        if df.shape[1] < 7:

            print(
                "Skipped:",
                file_path.name,
                "| insufficient columns"
            )

            skipped_files += 1
            continue

        # ----------------------------------------------------
        # Select sensor stream 0,0
        # ----------------------------------------------------

        df_selected = df[
            (pd.to_numeric(df.iloc[:, 5], errors="coerce") == 0) &
            (pd.to_numeric(df.iloc[:, 6], errors="coerce") == 0)
        ]

        # ----------------------------------------------------
        # Extract X, Y, Z
        # ----------------------------------------------------

        acceleration = df_selected.iloc[:, 2:5].copy()

        # Convert each column to numeric
        acceleration = acceleration.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Remove invalid rows
        acceleration = acceleration.dropna()

        # Convert to NumPy
        acceleration = acceleration.values.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Check enough data
        # ----------------------------------------------------

        if len(acceleration) < WINDOW_SIZE:

            skipped_files += 1
            continue

        # ----------------------------------------------------
        # Determine label
        # ----------------------------------------------------

        filename = file_path.name.upper()

        if "_ADL_" in filename:

            label = 0

        elif "_FALL_" in filename:

            label = 1

        else:

            skipped_files += 1
            continue

        # ----------------------------------------------------
        # Create windows
        # ----------------------------------------------------

        for start in range(
            0,
            len(acceleration) - WINDOW_SIZE + 1,
            STEP_SIZE
        ):

            window = acceleration[
                start:start + WINDOW_SIZE
            ]

            # 450 × 3 → 1350 values
            window_flat = window.reshape(-1)

            X_windows.append(window_flat)
            y_labels.append(label)

        processed_files += 1

    except Exception as e:

        print(
            "Error:",
            file_path.name,
            "|",
            e
        )

        skipped_files += 1

    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if count % 50 == 0:

        print(
            f"Processed {count}/{len(csv_files)} files"
        )


# ============================================================
# 6. CREATE NUMPY ARRAYS
# ============================================================

X = np.array(
    X_windows,
    dtype=np.float32
)

y = np.array(
    y_labels,
    dtype=np.int64
)


# ============================================================
# 7. SAVE
# ============================================================

X_path = output_folder / "X_umafall.npy"
y_path = output_folder / "y_umafall.npy"

np.save(X_path, X)
np.save(y_path, y)


# ============================================================
# 8. RESULTS
# ============================================================

print()
print("=" * 60)
print("DATASET PREPARATION COMPLETE")
print("=" * 60)

print("Processed files:", processed_files)
print("Skipped files:", skipped_files)

print()

print("X shape:", X.shape)
print("y shape:", y.shape)

print()

print("NORMAL windows:", np.sum(y == 0))
print("FALL windows:", np.sum(y == 1))

print()

print("Saved:")
print(X_path)
print(y_path)

print("=" * 60)