
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def load_test_data(file_path):
    return pd.read_csv(file_path)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    print("\n--- TEST SET EVALUATION ---")

    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

    print(
        "Balanced Accuracy:",
        f"{balanced_accuracy_score(y_test, predictions):.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    # ROC-AUC for binary classification
    if len(model.classes_) == 2:
        positive_class = model.classes_[1]
        positive_index = list(model.classes_).index(positive_class)

        auc = roc_auc_score(
            y_test,
            probabilities[:, positive_index],
            labels=model.classes_
        )

        print(f"\nROC-AUC: {auc:.4f}")


def display_feature_importance(model, feature_names):

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\n--- FEATURE IMPORTANCE ---")
    print(importance_df.to_string(index=False))

    return importance_df


if __name__ == "__main__":

    print("\n--- MODEL EVALUATION STARTED ---")

    project_root = Path(__file__).resolve().parents[2]

    feature_folder = (
        project_root / "data" / "processed" / "features"
    )

    model_path = (
        project_root / "models" / "saved" /
        "baseline_random_forest.pkl"
    )

    test_path = feature_folder / "test_features.csv"

    model = joblib.load(model_path)
    test_df = load_test_data(test_path)

    X_test = test_df.drop(
        columns=["SubjectID", "Group"]
    )

    y_test = test_df["Group"]

    # Ensure the test features match the training feature order
    X_test = X_test[model.feature_names_in_]

    print("Test participants:", len(X_test))
    print("Input features:", X_test.shape[1])

    evaluate_model(model, X_test, y_test)

    importance_df = display_feature_importance(
        model,
        X_test.columns
    )

    output_folder = project_root / "data" / "processed" / "evaluation"
    output_folder.mkdir(parents=True, exist_ok=True)

    importance_path = (
        output_folder / "baseline_feature_importance.csv"
    )

    importance_df.to_csv(importance_path, index=False)

    print("\nFeature importance saved:")
    print(importance_path)

    print("\n--- MODEL EVALUATION COMPLETED ---")
