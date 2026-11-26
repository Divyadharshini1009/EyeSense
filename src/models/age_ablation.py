
import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def load_data(path):
    return pd.read_csv(path)


def prepare_data(df):
    # Use only eye-movement features; exclude Age
    X = df.drop(columns=["SubjectID", "Group", "Age"])
    y = df["Group"]
    return X, y


def evaluate(model, X, y, dataset_name):
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    print(f"\n--- {dataset_name} RESULTS (WITHOUT AGE) ---")

    print(f"Accuracy: {accuracy_score(y, predictions):.4f}")

    print(
        "Balanced Accuracy:",
        f"{balanced_accuracy_score(y, predictions):.4f}"
    )

    print("\nClassification Report:")
    print(classification_report(y, predictions, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))

    # Group 1 is the positive class (Dyslexic)
    positive_index = list(model.classes_).index(1)

    y_binary = (y == 1).astype(int)

    auc = roc_auc_score(
        y_binary,
        probabilities[:, positive_index]
    )

    print(f"\nROC-AUC (Group 1 = Positive): {auc:.4f}")


if __name__ == "__main__":

    print("\n--- AGE ABLATION EXPERIMENT STARTED ---")

    root = Path(__file__).resolve().parents[2]

    feature_folder = root / "data" / "processed" / "features"
    model_folder = root / "models" / "saved"
    model_folder.mkdir(parents=True, exist_ok=True)

    train_df = load_data(feature_folder / "train_features.csv")
    validation_df = load_data(feature_folder / "validation_features.csv")
    test_df = load_data(feature_folder / "test_features.csv")

    X_train, y_train = prepare_data(train_df)
    X_validation, y_validation = prepare_data(validation_df)
    X_test, y_test = prepare_data(test_df)

    assert list(X_train.columns) == list(X_validation.columns)
    assert list(X_train.columns) == list(X_test.columns)

    print("\nTraining participants:", len(X_train))
    print("Validation participants:", len(X_validation))
    print("Test participants:", len(X_test))
    print("Input features:", X_train.shape[1])

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining Random Forest without Age...")
    model.fit(X_train, y_train)

    print("Training completed!")

    evaluate(model, X_validation, y_validation, "VALIDATION")
    evaluate(model, X_test, y_test, "TEST")

    model_path = model_folder / "random_forest_without_age.pkl"
    joblib.dump(model, model_path)

    print("\nModel saved successfully:")
    print(model_path)

    print("\n--- AGE ABLATION EXPERIMENT COMPLETED ---")
