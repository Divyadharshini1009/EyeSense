
import sys
import numpy as np
import pandas as pd
import joblib

from pathlib import Path

# Locate project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(PROJECT_ROOT))

from src.models.baseline_model import extract_features
from src.uncertainty.uncertainty import (
    calculate_uncertainty,
    assign_uncertainty_level
)


# ==========================================
# 1. LOAD TRAINED MODEL
# ==========================================

MODEL_PATH = (
    PROJECT_ROOT / "models" / "saved" /
    "calibrated_random_forest.pkl"
)

model = joblib.load(MODEL_PATH)

FEATURE_NAMES = list(model.feature_names_in_)


# ==========================================
# 2. PREPARE PARTICIPANT INPUT
# ==========================================

def prepare_input(sequence, age):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.ndim != 2 or sequence.shape[1] != 4:
        raise ValueError(
            "Sequence must have 4 columns: "
            "Word_Number, FIX_X, FIX_Y, FIX_DURATION."
        )

    if len(sequence) == 0:
        raise ValueError("Fixation sequence cannot be empty.")

    if not np.isfinite(sequence).all():
        raise ValueError("Sequence contains invalid values.")

    if not np.isfinite(age) or age <= 0:
        raise ValueError("A valid participant age is required.")

    features = extract_features(sequence)

    features["Age"] = int(age)

    input_df = pd.DataFrame([features])

    # Match the exact training feature order
    input_df = input_df[FEATURE_NAMES]

    return input_df


# ==========================================
# 3. GENERATE PREDICTION
# ==========================================

def predict_participant(sequence, age, subject_id="Unknown"):

    input_df = prepare_input(sequence, age)

    probabilities = model.predict_proba(input_df)

    predicted_group = model.predict(input_df)[0]

    confidence, prediction_entropy, margin = (
        calculate_uncertainty(probabilities)
    )

    group1_index = list(model.classes_).index(1)
    group2_index = list(model.classes_).index(2)

    result = {
        "SubjectID": subject_id,
        "Predicted_Group": int(predicted_group),
        "Group1_Probability": float(
            probabilities[0, group1_index]
        ),
        "Group2_Probability": float(
            probabilities[0, group2_index]
        ),
        "Prediction_Confidence": float(confidence[0]),
        "Prediction_Entropy": float(prediction_entropy[0]),
        "Probability_Margin": float(margin[0]),
        "Uncertainty_Level": assign_uncertainty_level(
            confidence[0],
            prediction_entropy[0]
        )
    }

    return result


# ==========================================
# 4. TEST USING EXISTING DATA
# ==========================================

if __name__ == "__main__":

    print("\n--- EYESENSE AI PREDICTION MODULE ---")

    test_data_path = (
        PROJECT_ROOT / "data" / "processed" /
        "splits" / "test_data.npz"
    )

    test_data = np.load(
        test_data_path,
        allow_pickle=True
    )

    # Use one existing participant for an offline test
    sequence = test_data["sequences"][0]
    age = int(test_data["ages"][0])
    subject_id = str(test_data["subject_ids"][0])

    result = predict_participant(
        sequence,
        age,
        subject_id
    )

    print("\nParticipant:", result["SubjectID"])
    print("Predicted group:", result["Predicted_Group"])

    print(
        "Group 1 probability:",
        round(result["Group1_Probability"], 4)
    )

    print(
        "Group 2 probability:",
        round(result["Group2_Probability"], 4)
    )

    print(
        "Prediction confidence:",
        round(result["Prediction_Confidence"], 4)
    )

    print(
        "Prediction entropy:",
        round(result["Prediction_Entropy"], 4)
    )

    print(
        "Probability margin:",
        round(result["Probability_Margin"], 4)
    )

    print(
        "Uncertainty level:",
        result["Uncertainty_Level"]
    )

    print("\n--- PREDICTION COMPLETED ---")
