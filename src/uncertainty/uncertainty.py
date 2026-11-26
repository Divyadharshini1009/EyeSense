
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from scipy.stats import entropy


def calculate_uncertainty(probabilities):
    """
    Calculate prediction confidence, entropy
    and probability margin.
    """

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


def generate_uncertainty_report(model, X, subject_ids):

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

    return results


if __name__ == "__main__":

    print("\n--- UNCERTAINTY ESTIMATION STARTED ---")

    project_root = Path(__file__).resolve().parents[2]

    model_path = (
        project_root / "models" / "saved" /
        "baseline_random_forest.pkl"
    )

    validation_path = (
        project_root / "data" / "processed" /
        "features" / "validation_features.csv"
    )

    output_folder = (
        project_root / "data" / "processed" /
        "uncertainty"
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    model = joblib.load(model_path)

    validation_df = pd.read_csv(validation_path)

    subject_ids = validation_df["SubjectID"]

    X_validation = validation_df.drop(
        columns=["SubjectID", "Group"]
    )

    X_validation = X_validation[model.feature_names_in_]

    results = generate_uncertainty_report(
        model,
        X_validation,
        subject_ids
    )

    output_path = (
        output_folder / "validation_uncertainty.csv"
    )

    results.to_csv(output_path, index=False)

    print("\nParticipants analysed:", len(results))

    print("\nUncertainty distribution:")
    print(results["Uncertainty_Level"].value_counts())

    print("\nAverage prediction confidence:",
          round(results["Prediction_Confidence"].mean(), 4))

    print("Average prediction entropy:",
          round(results["Prediction_Entropy"].mean(), 4))

    print("\nFirst five predictions:")
    print(results.head().to_string(index=False))

    print("\nSaved uncertainty report:")
    print(output_path)

    print("\n--- UNCERTAINTY ESTIMATION COMPLETED ---")
