"""
CPU5006: Artificial Intelligence - Assessment 2
Machine Learning Scientific Research Paper

Title:
Comparing Decision Tree and k-Nearest Neighbours for Predicting
Breast Cancer Diagnosis using the Wisconsin Diagnostic Dataset

NOTE: Parts of this code (structure, comments, and some implementations)
were written with assistance from ChatGPT (OpenAI, 2025).
"""

# =========================
# 1. Imports
# =========================

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =========================
# 2. Configuration
# =========================

RANDOM_SEED = 42
TEST_SIZE = 0.2
K_FOR_KNN = 5
MAX_DEPTH_DT = None  # e.g. set to 5 to limit tree depth


# =========================
# 3. Load Dataset
# =========================

def load_breast_cancer_dataset():
    """
    Load the Breast Cancer Wisconsin (Diagnostic) dataset
    from scikit-learn's built-in datasets.

    Returns:
        X (pd.DataFrame): feature matrix
        y (np.ndarray): target vector (0/1)
        target_mapping (dict): mapping of class index -> label
    """
    bc = load_breast_cancer(as_frame=True)

    # Features (DataFrame) and target (0 or 1)
    X = bc.data
    y = bc.target.values

    # Class names, e.g. ['malignant', 'benign']
    target_names = bc.target_names
    target_mapping = {i: name for i, name in enumerate(target_names)}

    print("Target classes:", list(target_names))
    print("Class mapping (index -> label):", target_mapping)

    return X, y, target_mapping


# =========================
# 4. Split & Scale
# =========================

def split_and_scale_data(X: pd.DataFrame, y: np.ndarray):
    """
    Split data into train/test and scale features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


# =========================
# 5. Model Training
# =========================

def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray):
    """
    Train a Decision Tree classifier using Gini impurity.
    """
    dt = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH_DT,
        random_state=RANDOM_SEED,
    )
    dt.fit(X_train, y_train)
    return dt


def train_knn(X_train_scaled: np.ndarray, y_train: np.ndarray):
    """
    Train a K-Nearest Neighbours classifier.
    """
    knn = KNeighborsClassifier(
        n_neighbors=K_FOR_KNN,
        metric="euclidean",
    )
    knn.fit(X_train_scaled, y_train)
    return knn


# =========================
# 6. Evaluation
# =========================

def evaluate_model(name: str, model, X_test, y_test):
    """
    Evaluate and print metrics for a model.
    Returns metrics as dict.
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="binary", zero_division=0)
    rec = recall_score(y_test, y_pred, average="binary", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print(f"Model: {name}")
    print("=" * 60)
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}\n")

    print("Classification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0,
            target_names=["malignant", "benign"],
        )
    )
    print("Confusion matrix:")
    print(cm)
    print()

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


# =========================
# 7. Main Experiment
# =========================

def main():
    print("Loading Breast Cancer Wisconsin (Diagnostic) dataset...")
    X, y, target_mapping = load_breast_cancer_dataset()
    print("Dataset shape:", X.shape)
    print("First few rows of X:")
    print(X.head(), "\n")

    print("Splitting and scaling data...")
    (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
    ) = split_and_scale_data(X, y)

    print("Training Decision Tree...")
    dt_model = train_decision_tree(X_train, y_train)

    print("Training KNN...")
    knn_model = train_knn(X_train_scaled, y_train)

    print("Evaluating models...\n")
    dt_results = evaluate_model("Decision Tree", dt_model, X_test, y_test)
    knn_results = evaluate_model(
        "K-Nearest Neighbours", knn_model, X_test_scaled, y_test
    )

    comparison_df = pd.DataFrame([dt_results, knn_results])
    print("Summary of model performance:")
    print(comparison_df)

    comparison_df.to_csv("breast_cancer_model_comparison.csv", index=False)
    print("\nSaved breast_cancer_model_comparison.csv in the current directory.")


if __name__ == "__main__":
    main()
