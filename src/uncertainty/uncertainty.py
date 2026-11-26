
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.stats import entropy

from sklearn.calibration import (
    CalibratedClassifierCV,
    calibration_curve
)
from sklearn.metrics import (
    brier_score_loss,
    log_loss
)


# ==========================================
# 1. EXISTING UNCERTAINTY FUNCTIONS
# ==========================================

def calculate_uncertainty(probabilities):

    confidence = np.max(probabilities, axis=1)

    prediction_entropy = entropy(
        probabilities.T,
        base=2
    )

    sorted_probabilities = np.sort(
        probabilities,
        axis=1
    )

    margin = (
        sorted_probabilities[:, -1]
        - sorted_probabilities[:, -2]
    )

    return confidence, prediction_entropy, margin


def assign_uncertainty_level(confidence, prediction_entropy):

    if confidence >= 0.80 and prediction_entropy <= 0.50:
        return "Low"

    elif confidence >= 0.60 and prediction_entropy <= 0.85:
        return "Moderate"

    else:
        return "High"


def generate_uncertainty_report(
    model, X, subject_ids, calibrated=False
):

    probabilities = model.predict_proba(X)
    predictions = model.predict(X)

    confidence, prediction_entropy, margin = (
        calculate_uncertainty(probabilities)
    )

    results = pd.DataFrame({
        "SubjectID": subject_ids,
        "Predicted_Group": predictions,
        "Prediction_Confidence": confidence,
        "Prediction_Entropy": prediction_entropy,
        "Probability_Margin": margin,
        "Uncertainty_Level": [
            assign_uncertainty_level(c, e)
            for c, e in zip(confidence, prediction_entropy)
        ]
    })

    if calibrated:
        results["Group2_Probability"] = probabilities[
            :, list(model.classes_).index(2)
        ]

    return results


# ==========================================
# 2. EXPECTED CALIBRATION ERROR
# ==========================================

def calculate_ece(y_true, probabilities, positive_class=2, bins=10):

    positive_index = list(range(len(probabilities[0])))

    # Convert actual labels to binary indicators for Group 2
    actual = (np.asarray(y_true) == positive_class).astype(int)

    predicted = probabilities[:, positive_index.index(
        list(sorted(set(y_true))).index(positive_class)
    )]

    bin_edges = np.linspace(0, 1, bins + 1)

    ece = 0.0

    for i in range(bins):

        if i == bins - 1:
            mask = (
                (predicted >= bin_edges[i]) &
                (predicted <= bin_edges[i + 1])
            )
        else:
            mask = (
                (predicted >= bin_edges[i]) &
                (predicted < bin_edges[i + 1])
            )

        if np.any(mask):

            average_confidence = predicted[mask].mean()
            average_accuracy = actual[mask].mean()

            ece += (
                np.sum(mask) / len(predicted)
            ) * abs(
                average_confidence - average_accuracy
            )

    return ece


# ==========================================
# 3. CALIBRATION METRICS
# ==========================================

def evaluate_calibration(y_true, probabilities, classes):

    group2_index = list(classes).index(2)

    group2_probabilities = probabilities[:, group2_index]

    actual_group2 = (
        np.asarray(y_true) == 2
    ).astype(int)

    brier = brier_score_loss(
        actual_group2,
        group2_probabilities
    )

    loss = log_loss(
        y_true,
        probabilities,
        labels=classes
    )

    ece = calculate_ece(
        y_true,
        probabilities,
        positive_class=2
    )

    return {
        "Brier_Score": brier,
        "Log_Loss": loss,
        "ECE": ece
    }


# ==========================================
# 4. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("\n--- UNCERTAINTY AND CALIBRATION ---")

    project_root = Path(__file__).resolve().parents[2]

    model_path = (
        project_root / "models" / "saved" /
        "baseline_random_forest.pkl"
    )

    feature_folder = (
        project_root / "data" / "processed" / "features"
    )

    output_folder = (
        project_root / "data" / "processed" /
        "uncertainty"
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    # --------------------------------------
    # LOAD ORIGINAL MODEL AND DATA
    # --------------------------------------

    original_model = joblib.load(model_path)

    feature_names = list(
        original_model.feature_names_in_
    )

    train_df = pd.read_csv(
        feature_folder / "train_features.csv"
    )

    validation_df = pd.read_csv(
        feature_folder / "validation_features.csv"
    )

    test_df = pd.read_csv(
        feature_folder / "test_features.csv"
    )

    X_train = train_df[feature_names]
    y_train = train_df["Group"]

    X_validation = validation_df[feature_names]
    y_validation = validation_df["Group"]

    X_test = test_df[feature_names]
    y_test = test_df["Group"]

    print("Training participants:", len(train_df))
    print("Validation participants:", len(validation_df))
    print("Test participants:", len(test_df))

    # --------------------------------------
    # RAW MODEL PROBABILITIES
    # --------------------------------------

    raw_validation_probabilities = (
        original_model.predict_proba(X_validation)
    )

    raw_test_probabilities = (
        original_model.predict_proba(X_test)
    )

    # --------------------------------------
    # FIT CALIBRATED MODEL
    # --------------------------------------

    print("\nFitting sigmoid calibration...")

    try:
        calibrated_model = CalibratedClassifierCV(
            estimator=original_model,
            method="sigmoid",
            cv=5
        )

    except TypeError:
        # Compatibility with older scikit-learn versions
        calibrated_model = CalibratedClassifierCV(
            base_estimator=original_model,
            method="sigmoid",
            cv=5
        )

    # The calibration process trains cloned estimators
    # using only training data.
    calibrated_model.fit(X_train, y_train)

    calibrated_validation_probabilities = (
        calibrated_model.predict_proba(X_validation)
    )

    calibrated_test_probabilities = (
        calibrated_model.predict_proba(X_test)
    )

    # --------------------------------------
    # CALIBRATION COMPARISON
    # --------------------------------------

    results = []

    for split_name, y_true, raw_probs, calibrated_probs in [
        (
            "Validation",
            y_validation,
            raw_validation_probabilities,
            calibrated_validation_probabilities
        ),
        (
            "Test",
            y_test,
            raw_test_probabilities,
            calibrated_test_probabilities
        )
    ]:

        raw_metrics = evaluate_calibration(
            y_true,
            raw_probs,
            original_model.classes_
        )

        calibrated_metrics = evaluate_calibration(
            y_true,
            calibrated_probs,
            original_model.classes_
        )

        for model_name, metrics in [
            ("Original", raw_metrics),
            ("Calibrated", calibrated_metrics)
        ]:

            results.append({
                "Split": split_name,
                "Model": model_name,
                **metrics
            })

    comparison_df = pd.DataFrame(results)

    comparison_path = (
        output_folder / "calibration_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    print("\n--- CALIBRATION COMPARISON ---")
    print(comparison_df.round(4).to_string(index=False))

    # --------------------------------------
    # SAVE CALIBRATION CURVE DATA
    # --------------------------------------

    curve_rows = []

    for split_name, y_true, raw_probs, calibrated_probs in [
        (
            "Validation",
            y_validation,
            raw_validation_probabilities,
            calibrated_validation_probabilities
        ),
        (
            "Test",
            y_test,
            raw_test_probabilities,
            calibrated_test_probabilities
        )
    ]:

        for model_name, probabilities in [
            ("Original", raw_probs),
            ("Calibrated", calibrated_probs)
        ]:

            group2_index = list(
                original_model.classes_
            ).index(2)

            group2_probabilities = probabilities[
                :, group2_index
            ]

            actual_group2 = (
                np.asarray(y_true) == 2
            ).astype(int)

            fraction_positive, mean_predicted = (
                calibration_curve(
                    actual_group2,
                    group2_probabilities,
                    n_bins=10,
                    strategy="uniform"
                )
            )

            for actual_rate, predicted_rate in zip(
                fraction_positive,
                mean_predicted
            ):

                curve_rows.append({
                    "Split": split_name,
                    "Model": model_name,
                    "Mean_Predicted_Probability": predicted_rate,
                    "Actual_Group2_Rate": actual_rate
                })

    curve_df = pd.DataFrame(curve_rows)

    curve_path = (
        output_folder / "calibration_curve_data.csv"
    )

    curve_df.to_csv(curve_path, index=False)

    # --------------------------------------
    # GENERATE UNCERTAINTY REPORTS
    # --------------------------------------

    raw_validation_report = generate_uncertainty_report(
        original_model,
        X_validation,
        validation_df["SubjectID"]
    )

    calibrated_validation_report = generate_uncertainty_report(
        calibrated_model,
        X_validation,
        validation_df["SubjectID"],
        calibrated=True
    )

    calibrated_test_report = generate_uncertainty_report(
        calibrated_model,
        X_test,
        test_df["SubjectID"],
        calibrated=True
    )

    raw_validation_report.to_csv(
        output_folder / "validation_uncertainty.csv",
        index=False
    )

    calibrated_validation_report.to_csv(
        output_folder / "calibrated_validation_uncertainty.csv",
        index=False
    )

    calibrated_test_report.to_csv(
        output_folder / "calibrated_test_uncertainty.csv",
        index=False
    )

    # --------------------------------------
    # SAVE CALIBRATED MODEL
    # --------------------------------------

    calibrated_model_path = (
        project_root / "models" / "saved" /
        "calibrated_random_forest.pkl"
    )

    joblib.dump(
        calibrated_model,
        calibrated_model_path
    )

    print("\nCalibration reports saved.")
    print("Comparison:", comparison_path)
    print("Calibration curve data:", curve_path)
    print("Calibrated model:", calibrated_model_path)

    print("\n--- CALIBRATION COMPLETED ---")
