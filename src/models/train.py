
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def load_feature_data(file_path):
    return pd.read_csv(file_path)


def prepare_data(df):
    """
    Separate model features and target labels.
    """

    excluded_columns = ["SubjectID", "Group"]

    X = df.drop(columns=excluded_columns)
    y = df["Group"]

    return X, y


def train_baseline_model(X_train, y_train):
    """
    Train the Random Forest baseline classifier.
    """

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X, y, dataset_name):
    """
    Evaluate the trained model.
    """

    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)

    print(f"\n--- {dataset_name} RESULTS ---")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, predictions, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))


if __name__ == "__main__":

    print("\n--- BASELINE MODEL TRAINING STARTED ---")

    project_root = Path(__file__).resolve().parents[2]

    feature_folder = project_root / "data" / "processed" / "features"

    model_folder = project_root / "models" / "saved"
    model_folder.mkdir(parents=True, exist_ok=True)

    train_df = load_feature_data(
        feature_folder / "train_features.csv"
    )

    validation_df = load_feature_data(
        feature_folder / "validation_features.csv"
    )

    X_train, y_train = prepare_data(train_df)
    X_validation, y_validation = prepare_data(validation_df)

    print("\nTraining participants:", len(X_train))
    print("Validation participants:", len(X_validation))
    print("Number of input features:", X_train.shape[1])

    print("\nTraining Random Forest...")

    model = train_baseline_model(X_train, y_train)

    print("Training completed!")

    evaluate_model(
        model,
        X_train,
        y_train,
        "TRAINING"
    )

    evaluate_model(
        model,
        X_validation,
        y_validation,
        "VALIDATION"
    )

    model_path = model_folder / "baseline_random_forest.pkl"

    joblib.dump(model, model_path)

    print("\nModel saved successfully:")
    print(model_path)

    print("\n--- BASELINE MODEL TRAINING COMPLETED ---")
