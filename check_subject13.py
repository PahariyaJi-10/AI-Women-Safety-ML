import pandas as pd
from pathlib import Path

dataset_root = Path(
    r"C:\Users\Lenovo\Downloads\UMAFall_Dataset\UMAFall_Dataset"
)

file_path = next(
    dataset_root.glob(
        "*Subject_13_ADL_Bending_1*.csv"
    )
)

print("=" * 60)
print("SUBJECT 13 FILE")
print("=" * 60)

print("File:")
print(file_path.name)

df = pd.read_csv(
    file_path,
    sep=";",
    header=None,
    comment="%"
)

print()
print("Shape:", df.shape)

print()
print("First 20 rows:")
print(df.head(20).to_string(index=False))

print()
print("Data types:")
print(df.dtypes)