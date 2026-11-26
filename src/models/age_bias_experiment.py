
import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# --------------------------------
# CONFIGURATION
# --------------------------------

FEATURE_DIR = "data/processed/features"
MODEL_PATH = "models/saved/baseline_random_forest.pkl"

os.makedirs("data/processed/evaluation", exist_ok=True)

# Load the original model to preserve its exact feature order
original_model = joblib.load(MODEL_PATH)

all_features = list(original_model.feature_names_in_)

if "Age" not in all_features:
    raise ValueError("Age is not a feature in the original model.")

features_without_age = [
    feature for feature in all_features
    if feature != "Age"
]

# --------------------------------
# LOAD EXISTING PARTICIPANT SPLITS
# --------------------------------

train_df = pd.read_csv(
    os.path.join(FEATURE_DIR, "train_features.csv")
)

val_df = pd.read_csv(
    os.path.join(FEATURE_DIR, "validation_features.csv")
)

test_df = pd.read_csv(
    os.path.join(FEATURE_DIR, "test_features.csv")
)

for name, data in [
    ("Training", train_df),
    ("Validation", val_df),
    ("Test", test_df)
]:
    missing = [
        feature for feature in all_features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"{name} data is missing features: {missing}"
        )

print("\n--- AGE BIAS EXPERIMENT ---")
print("Training participants:", len(train_df))
print("Validation participants:", len(val_df))
print("Test participants:", len(test_df))
print("Features with Age:", len(all_features))
print("Features without Age:", len(features_without_age))


# --------------------------------
# MODEL TRAINING AND EVALUATION
# --------------------------------

def run_experiment(feature_list, model_name):

    print(f"\n--- {model_name} ---")

    X_train = train_df[feature_list]
    y_train = train_df["Group"]

    X_val = val_df[feature_list]
    y_val = val_df["Group"]

    X_test = test_df[feature_list]
    y_test = test_df["Group"]

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    results = {}

    for split_name, X, y in [
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test)
    ]:

        predictions = model.predict(X)
        probabilities = model.predict_proba(X)

        group2_index = list(model.classes_).index(2)

        group2_probabilities = probabilities[:, group2_index]

        metrics = {
            "Model": model_name,
            "Split": split_name,
            "Accuracy": accuracy_score(y, predictions),
            "Balanced_Accuracy": balanced_accuracy_score(
                y, predictions
            ),
            "Group2_Precision": precision_score(
                y, predictions, pos_label=2, zero_division=0
            ),
            "Group2_Recall": recall_score(
                y, predictions, pos_label=2, zero_division=0
            ),
            "Group2_F1": f1_score(
                y, predictions, pos_label=2, zero_division=0
            ),
            "ROC_AUC": roc_auc_score(
                y, group2_probabilities
            )
        }

        results[split_name] = metrics

        print(f"\n{split_name} Results:")

        for metric, value in metrics.items():
            if metric not in ["Model", "Split"]:
                print(f"{metric}: {value:.4f}")

    return list(results.values()), model


# --------------------------------
# RUN BOTH EXPERIMENTS
# --------------------------------

results_with_age, model_with_age = run_experiment(
    all_features,
    "Random Forest WITH Age"
)

results_without_age, model_without_age = run_experiment(
    features_without_age,
    "Random Forest WITHOUT Age"
)

# --------------------------------
# SAVE COMPARISON
# --------------------------------

all_results = (
    results_with_age + results_without_age
)

results_df = pd.DataFrame(all_results)

output_path = (
    "data/processed/evaluation/age_bias_comparison.csv"
)

results_df.to_csv(output_path, index=False)

print("\n--- FINAL AGE BIAS COMPARISON ---")

print(
    results_df.round(4).to_string(index=False)
)

# Save experimental models separately
joblib.dump(
    model_with_age,
    "models/saved/rf_experiment_with_age.pkl"
)

joblib.dump(
    model_without_age,
    "models/saved/rf_experiment_without_age.pkl"
)

print("\nComparison saved:", output_path)
print("Experimental models saved separately.")
print("\n--- EXPERIMENT COMPLETED ---")
