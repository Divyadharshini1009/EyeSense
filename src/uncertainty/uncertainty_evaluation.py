
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    roc_auc_score
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
    PROJECT_ROOT / "data" / "processed" /
    "uncertainty" /
    "contrastive_validation_uncertainty.csv"
)

OUTPUT_FOLDER = (
    PROJECT_ROOT / "data" / "processed" /
    "uncertainty"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

N_BINS = 10


# --------------------------------------------------
# EXPECTED CALIBRATION ERROR
# --------------------------------------------------

def calculate_ece(confidence, correct, n_bins=10):

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (confidence >= bin_edges[i]) &
                (confidence <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidence >= bin_edges[i]) &
                (confidence < bin_edges[i + 1])
            )

        if np.any(mask):

            bin_accuracy = correct[mask].mean()
            bin_confidence = confidence[mask].mean()

            bin_weight = mask.mean()

            ece += bin_weight * abs(
                bin_accuracy - bin_confidence
            )

    return float(ece)


# --------------------------------------------------
# RISK-COVERAGE ANALYSIS
# --------------------------------------------------

def calculate_risk_coverage(df):

    sorted_df = df.sort_values(
        "Prediction_Confidence",
        ascending=False
    ).reset_index(drop=True)

    results = []

    total = len(sorted_df)

    for coverage in [0.25, 0.50, 0.75, 1.00]:

        count = max(1, int(np.ceil(total * coverage)))

        selected = sorted_df.iloc[:count]

        accuracy = accuracy_score(
            selected["Actual_Group"],
            selected["Predicted_Group"]
        )

        results.append({
            "Coverage": coverage,
            "Participants_Retained": count,
            "Accuracy": accuracy,
            "Risk": 1.0 - accuracy
        })

    return pd.DataFrame(results)


# --------------------------------------------------
# MAIN EVALUATION
# --------------------------------------------------

def main():

    print("\n======================================")
    print(" EYESENSE UNCERTAINTY EVALUATION")
    print("======================================")

    if not REPORT_PATH.exists():

        raise FileNotFoundError(
            f"Uncertainty report not found: {REPORT_PATH}"
        )

    df = pd.read_csv(REPORT_PATH)

    required_columns = [
        "Actual_Group",
        "Predicted_Group",
        "Dyslexia_Probability",
        "Prediction_Confidence",
        "Predictive_Entropy",
        "Mutual_Information"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        raise ValueError("Uncertainty report is empty.")

    if df[required_columns].isnull().any().any():
        raise ValueError(
            "Missing values found in evaluation columns."
        )

    if not np.isfinite(
        df[required_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite values found in evaluation columns."
        )

    print("\nParticipants loaded:", len(df))

    # Group 1 is the positive class (Dyslexic).
    actual = (
        df["Actual_Group"].to_numpy() == 1
    ).astype(int)

    predicted = (
        df["Predicted_Group"].to_numpy() == 1
    ).astype(int)

    dyslexia_probability = df[
        "Dyslexia_Probability"
    ].to_numpy(dtype=float)

    confidence = df[
        "Prediction_Confidence"
    ].to_numpy(dtype=float)

    if (
        np.any(dyslexia_probability < 0) or
        np.any(dyslexia_probability > 1) or
        np.any(confidence < 0) or
        np.any(confidence > 1)
    ):
        raise ValueError(
            "Probabilities must be between 0 and 1."
        )

    if not np.allclose(
        df["Dyslexia_Probability"] +
        df["Non_Dyslexia_Probability"],
        1.0,
        atol=1e-4
    ):
        raise ValueError(
            "Class probabilities do not sum to 1."
        )

    # Prediction correctness
    correct = (actual == predicted).astype(int)

    # Accuracy
    accuracy = accuracy_score(
        actual,
        predicted
    )

    # Brier score
    brier = brier_score_loss(
        actual,
        dyslexia_probability
    )

    # Expected Calibration Error
    ece = calculate_ece(
        confidence,
        correct,
        N_BINS
    )

    # ROC-AUC
    auc = roc_auc_score(
        actual,
        dyslexia_probability
    )

    # Risk-coverage
    risk_coverage = calculate_risk_coverage(df)

    # Uncertainty versus correctness
    uncertainty_summary = df.copy()

    uncertainty_summary["Correct"] = correct

    correctness_analysis = (
        uncertainty_summary.groupby("Correct")
        .agg(
            Participants=("Correct", "size"),
            Average_Confidence=(
                "Prediction_Confidence", "mean"
            ),
            Average_Entropy=(
                "Predictive_Entropy", "mean"
            ),
            Average_Mutual_Information=(
                "Mutual_Information", "mean"
            )
        )
        .reset_index()
    )

    correctness_analysis["Result"] = (
        correctness_analysis["Correct"]
        .map({
            1: "Correct predictions",
            0: "Incorrect predictions"
        })
    )

    # Save risk-coverage results
    risk_path = (
        OUTPUT_FOLDER /
        "risk_coverage_results.csv"
    )

    risk_coverage.to_csv(
        risk_path,
        index=False
    )

    # Save correctness analysis
    correctness_path = (
        OUTPUT_FOLDER /
        "uncertainty_correctness_analysis.csv"
    )

    correctness_analysis.to_csv(
        correctness_path,
        index=False
    )

    # Save summary metrics
    summary = pd.DataFrame([{
        "Participants": len(df),
        "Accuracy": accuracy,
        "Brier_Score": brier,
        "Expected_Calibration_Error": ece,
        "ROC_AUC": auc
    }])

    summary_path = (
        OUTPUT_FOLDER /
        "uncertainty_evaluation_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    # Display results
    print("\n========== EVALUATION METRICS ==========")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Brier Score: {brier:.4f}")
    print(f"Expected Calibration Error: {ece:.4f}")
    print(f"ROC-AUC: {auc:.4f}")

    print("\n========== RISK-COVERAGE ANALYSIS ==========")

    print(
        risk_coverage.to_string(index=False)
    )

    print("\n========== UNCERTAINTY VS CORRECTNESS ==========")

    print(
        correctness_analysis.to_string(index=False)
    )

    print("\n========== SAVED FILES ==========")

    print(summary_path)
    print(risk_path)
    print(correctness_path)

    print("\nUNCERTAINTY EVALUATION COMPLETED")


if __name__ == "__main__":
    main()
