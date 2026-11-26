
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


def create_age_matched_data(df, dataset_name):

    matched_groups = []

    print(f"\n--- {dataset_name}: AGE MATCHING ---")

    for age in sorted(df["Age"].unique()):

        age_data = df[df["Age"] == age]

        group_1 = age_data[age_data["Group"] == 1]
        group_2 = age_data[age_data["Group"] == 2]

        # Keep only ages represented in both groups
        if len(group_1) == 0 or len(group_2) == 0:
            print(f"Age {age}: excluded (one group missing)")
            continue

        # Match both groups to the smaller group size
        sample_count = min(len(group_1), len(group_2))

        matched_groups.append(
            group_1.sample(
                n=sample_count,
                random_state=42
            )
        )

        matched_groups.append(
            group_2.sample(
                n=sample_count,
                random_state=42
            )
        )

        print(
            f"Age {age}: {sample_count} participants per group"
        )

    if not matched_groups:
        raise ValueError("No common ages found for matching.")

    matched_df = pd.concat(
        matched_groups,
        ignore_index=True
    )

    matched_df = matched_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    return matched_df


def prepare_data(df):

    X = df.drop(
        columns=["SubjectID", "Group", "Age"]
    )

    y = df["Group"]

    return X, y


def evaluate_model(model, X, y, dataset_name):

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)

    positive_index = list(model.classes_).index(1)

    y_binary = (y == 1).astype(int)

    auc = roc_auc_score(
        y_binary,
        probabilities[:, positive_index]
    )

    print(f"\n--- {dataset_name} RESULTS ---")

    print(
        f"Accuracy: {accuracy_score(y, predictions):.4f}"
    )

    print(
        "Balanced Accuracy:",
        f"{balanced_accuracy_score(y, predictions):.4f}"
    )

    print(f"ROC-AUC (Group 1 = Positive): {auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, predictions))


if __name__ == "__main__":

    print("\n--- AGE-MATCHED EXPERIMENT STARTED ---")

    root = Path(__file__).resolve().parents[2]

    feature_folder = (
        root / "data" / "processed" / "features"
    )

    output_folder = feature_folder / "age_matched"
    output_folder.mkdir(parents=True, exist_ok=True)

    model_folder = root / "models" / "saved"
    model_folder.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(
        feature_folder / "train_features.csv"
    )

    validation_df = pd.read_csv(
        feature_folder / "validation_features.csv"
    )

    test_df = pd.read_csv(
        feature_folder / "test_features.csv"
    )

    # Match ages separately within each existing split
    matched_train = create_age_matched_data(
        train_df, "TRAINING"
    )

    matched_validation = create_age_matched_data(
        validation_df, "VALIDATION"
    )

    matched_test = create_age_matched_data(
        test_df, "TEST"
    )

    # Save matched datasets
    matched_train.to_csv(
        output_folder / "train_age_matched.csv",
        index=False
    )

    matched_validation.to_csv(
        output_folder / "validation_age_matched.csv",
        index=False
    )

    matched_test.to_csv(
        output_folder / "test_age_matched.csv",
        index=False
    )

    print("\nMatched participant totals:")
    print("Training:", len(matched_train))
    print("Validation:", len(matched_validation))
    print("Test:", len(matched_test))

    X_train, y_train = prepare_data(matched_train)
    X_validation, y_validation = prepare_data(matched_validation)
    X_test, y_test = prepare_data(matched_test)

    assert list(X_train.columns) == list(X_validation.columns)
    assert list(X_train.columns) == list(X_test.columns)

    print("\nEye-movement input features:", X_train.shape[1])

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("\nTraining age-matched Random Forest...")
    model.fit(X_train, y_train)

    print("Training completed!")

    evaluate_model(
        model,
        X_validation,
        y_validation,
        "AGE-MATCHED VALIDATION"
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        "AGE-MATCHED TEST"
    )

    model_path = (
        model_folder / "age_matched_random_forest.pkl"
    )

    joblib.dump(model, model_path)

    print("\nModel saved:")
    print(model_path)

    print("\n--- AGE-MATCHED EXPERIMENT COMPLETED ---")
