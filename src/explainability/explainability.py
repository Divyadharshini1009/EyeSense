
import os
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib.pyplot as plt
import joblib

# -------------------------------
# CONFIGURATION
# -------------------------------

MODEL_PATH = "models/saved/baseline_random_forest.pkl"

DATA_PATH = (
    "data/processed/features/test_features.csv"
)

OUTPUT_DIR = "data/processed/explainability"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# -------------------------------
# LOAD MODEL AND TEST DATA
# -------------------------------

print("\n--- EYESENSE SHAP EXPLAINABILITY ---")

model = joblib.load(MODEL_PATH)

df = pd.read_csv(DATA_PATH)

# Use exactly the features the model was trained on
if not hasattr(model, "feature_names_in_"):
    raise ValueError(
        "Model does not contain feature names. "
        "We need to verify the original training feature order."
    )

feature_names = list(model.feature_names_in_)

missing = [
    feature for feature in feature_names
    if feature not in df.columns
]

if missing:
    raise ValueError(
        f"Missing model features in test data: {missing}"
    )

X = df[feature_names].copy()

# Keep participant information separately
participant_ids = (
    df["SubjectID"]
    if "SubjectID" in df.columns
    else pd.Series(range(len(df)))
)

actual_groups = (
    df["Group"]
    if "Group" in df.columns
    else pd.Series([None] * len(df))
)

print("Test participants:", len(X))
print("Model features:", len(feature_names))


# -------------------------------
# SHAP EXPLAINER
# -------------------------------

explainer = shap.TreeExplainer(model)

# Limit explanation computation to at most 300 participants
X_sample = X.sample(
    n=min(300, len(X)),
    random_state=42
)

print("\nCalculating SHAP values...")

shap_values = explainer.shap_values(X_sample)


# -------------------------------
# EXTRACT GROUP 2 VALUES
# -------------------------------

if isinstance(shap_values, list):

    group2_values = shap_values[1]

else:

    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        group2_values = shap_values[:, :, 1]
    else:
        group2_values = shap_values

print("SHAP calculation completed.")


# -------------------------------
# GLOBAL EXPLAINABILITY
# -------------------------------

importance = np.abs(group2_values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Absolute_SHAP": importance
})

importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

importance_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "global_shap_importance.csv"
    ),
    index=False
)

print("\n--- GLOBAL FEATURE IMPORTANCE ---")
print(importance_df.to_string(index=False))


# Global importance chart
plt.figure(figsize=(10, 7))

plt.barh(
    importance_df["Feature"],
    importance_df["Mean_Absolute_SHAP"]
)

plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Features")
plt.title("EyeSense AI - Global SHAP Feature Importance")
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "global_shap_importance.png"
    ),
    dpi=300
)

plt.close()


# -------------------------------
# INDIVIDUAL EXPLAINABILITY
# -------------------------------

# Map sampled rows back to original participant data
sample_indices = X_sample.index

individual_df = pd.DataFrame(
    group2_values,
    columns=[
        f"SHAP_{feature}"
        for feature in feature_names
    ]
)

individual_df.insert(
    0,
    "SubjectID",
    participant_ids.loc[sample_indices].values
)

individual_df.insert(
    1,
    "Actual_Group",
    actual_groups.loc[sample_indices].values
)

individual_df["Predicted_Group"] = (
    model.predict(X_sample)
)

individual_df["Group2_Probability"] = (
    model.predict_proba(X_sample)[:, 1]
)

individual_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "individual_shap_explanations.csv"
    ),
    index=False
)


# Display the strongest contributors for first participant
first_explanation = pd.DataFrame({
    "Feature": feature_names,
    "SHAP_Contribution": group2_values[0]
})

first_explanation["Absolute_Contribution"] = (
    first_explanation["SHAP_Contribution"].abs()
)

first_explanation = first_explanation.sort_values(
    by="Absolute_Contribution",
    ascending=False
)

print("\n--- INDIVIDUAL PARTICIPANT EXPLANATION ---")

print(
    individual_df.iloc[0][
        ["SubjectID", "Actual_Group",
         "Predicted_Group", "Group2_Probability"]
    ]
)

print("\nTop contributing features:")

print(
    first_explanation[
        ["Feature", "SHAP_Contribution"]
    ].head(5).to_string(index=False)
)


print("\n--- EXPLAINABILITY COMPLETED ---")
print("Output directory:", OUTPUT_DIR)
