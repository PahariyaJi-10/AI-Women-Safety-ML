import pandas as pd
from pathlib import Path


# ============================================================
# 1. DATASET PATH
# ============================================================

dataset_root = Path(
    r"C:\Users\Lenovo\Downloads\UMAFall_Dataset\UMAFall_Dataset"
)


# ============================================================
# 2. FIND ONE ADL FILE AND ONE FALL FILE
# ============================================================

adl_files = list(
    dataset_root.glob("*_ADL_*.csv")
)

fall_files = list(
    dataset_root.glob("*_FALL_*.csv")
)

print("=" * 60)
print("UMAFall DATASET INSPECTION")
print("=" * 60)

print("ADL files found:", len(adl_files))
print("Fall files found:", len(fall_files))


# ============================================================
# 3. CHECK FILES EXIST
# ============================================================

if len(adl_files) == 0:
    print("No ADL files found.")
    exit()

if len(fall_files) == 0:
    print("No FALL files found.")
    exit()


adl_file = adl_files[0]
fall_file = fall_files[0]


print()
print("ADL file:")
print(adl_file.name)

print()
print("FALL file:")
print(fall_file.name)


# ============================================================
# 4. READ ADL FILE
# ============================================================

adl_data = pd.read_csv(
    adl_file,
    sep=";",
    header=None,
    comment="%"
)


# ============================================================
# 5. READ FALL FILE
# ============================================================

fall_data = pd.read_csv(
    fall_file,
    sep=";",
    header=None,
    comment="%"
)


# ============================================================
# 6. DISPLAY BASIC INFORMATION
# ============================================================

print()
print("=" * 60)
print("ADL DATA")
print("=" * 60)

print("Rows:", len(adl_data))
print("Columns:", len(adl_data.columns))

print()
print("First 5 rows:")

print(adl_data.head())


print()
print("=" * 60)
print("FALL DATA")
print("=" * 60)

print("Rows:", len(fall_data))
print("Columns:", len(fall_data.columns))

print()
print("First 5 rows:")

print(fall_data.head())


# ============================================================
# 7. CHECK ACCELEROMETER VALUES
# ============================================================

print()
print("=" * 60)
print("ACCELEROMETER INFORMATION")
print("=" * 60)


# X, Y, Z are columns 2, 3 and 4
# because Python starts counting from 0

acc_x_adl = pd.to_numeric(
    adl_data.iloc[:, 2],
    errors="coerce"
)

acc_y_adl = pd.to_numeric(
    adl_data.iloc[:, 3],
    errors="coerce"
)

acc_z_adl = pd.to_numeric(
    adl_data.iloc[:, 4],
    errors="coerce"
)


print()
print("ADL Accelerometer:")

print("X min:", acc_x_adl.min())
print("X max:", acc_x_adl.max())

print("Y min:", acc_y_adl.min())
print("Y max:", acc_y_adl.max())

print("Z min:", acc_z_adl.min())
print("Z max:", acc_z_adl.max())


# ============================================================
# 8. FALL ACCELEROMETER
# ============================================================

acc_x_fall = pd.to_numeric(
    fall_data.iloc[:, 2],
    errors="coerce"
)

acc_y_fall = pd.to_numeric(
    fall_data.iloc[:, 3],
    errors="coerce"
)

acc_z_fall = pd.to_numeric(
    fall_data.iloc[:, 4],
    errors="coerce"
)


print()
print("FALL Accelerometer:")

print("X min:", acc_x_fall.min())
print("X max:", acc_x_fall.max())

print("Y min:", acc_y_fall.min())
print("Y max:", acc_y_fall.max())

print("Z min:", acc_z_fall.min())
print("Z max:", acc_z_fall.max())


print()
print("=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)

print()
print("=" * 60)
print("SENSOR STREAM CHECK")
print("=" * 60)

print("\nADL sensor combinations:")
print(adl_data.groupby([5, 6]).size())

print("\nFALL sensor combinations:")
print(fall_data.groupby([5, 6]).size())