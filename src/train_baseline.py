import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================================
# 1. PROJECT PATH
# ==========================================

project_root = Path(__file__).resolve().parent.parent

X_path = project_root / "dataset" / "processed" / "X_mfcc.npy"
y_path = project_root / "dataset" / "processed" / "y_labels.npy"


# ==========================================
# 2. LOAD DATASET
# ==========================================

X = np.load(X_path)
y = np.load(y_path)

print("Dataset loaded successfully!")
print("Feature shape:", X.shape)
print("Label shape:", y.shape)


# ==========================================
# 3. TRAIN / TEST SPLIT
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
# 4. FEATURE STANDARDIZATION
# ==========================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# 5. RANDOM FOREST MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42
)

print()
print("Training Random Forest model...")

model.fit(X_train, y_train)

print("Training completed!")


# ==========================================
# 6. PREDICTION
# ==========================================

y_pred = model.predict(X_test)


# ==========================================
# 7. EVALUATION
# ==========================================

accuracy = accuracy_score(y_test, y_pred)

print()
print("==========================================")
print("MODEL EVALUATION")
print("==========================================")

print("Accuracy:", round(accuracy * 100, 2), "%")

print()
print("Classification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Normal", "Distress"]
    )
)


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)