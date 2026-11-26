
import numpy as np
import pandas as pd
from pathlib import Path


def extract_features(sequence):
    """
    Extract statistical features from one participant's
    eye-tracking fixation sequence.
    """

    sequence = np.asarray(sequence, dtype=np.float32)

    word_number = sequence[:, 0]
    fix_x = sequence[:, 1]
    fix_y = sequence[:, 2]
    fix_duration = sequence[:, 3]

    features = {
        "Mean_Fixation_Duration": np.mean(fix_duration),
        "Std_Fixation_Duration": np.std(fix_duration),
        "Min_Fixation_Duration": np.min(fix_duration),
        "Max_Fixation_Duration": np.max(fix_duration),

        "Mean_Gaze_X": np.mean(fix_x),
        "Std_Gaze_X": np.std(fix_x),
        "Mean_Gaze_Y": np.mean(fix_y),
        "Std_Gaze_Y": np.std(fix_y),

        "Total_Fixations": len(sequence),
        "Unique_Words": len(np.unique(word_number)),

        "Mean_Word_Number": np.mean(word_number),
        "Std_Word_Number": np.std(word_number)
    }

    return features


def build_feature_dataset(split_data):
    """
    Convert participant sequences into a feature dataset.
    """

    records = []

    for i in range(len(split_data["subject_ids"])):

        sequence = split_data["sequences"][i]

        features = extract_features(sequence)

        features["SubjectID"] = str(split_data["subject_ids"][i])
        features["Age"] = int(split_data["ages"][i])
        features["Group"] = int(split_data["groups"][i])

        records.append(features)

    feature_df = pd.DataFrame(records)

    return feature_df


if __name__ == "__main__":

    print("\n--- BASELINE FEATURE EXTRACTION STARTED ---")

    project_root = Path(__file__).resolve().parents[2]

    split_folder = project_root / "data" / "processed" / "splits"

    output_folder = project_root / "data" / "processed" / "features"
    output_folder.mkdir(parents=True, exist_ok=True)

    for split_name in ["train", "validation", "test"]:

        input_path = split_folder / f"{split_name}_data.npz"

        print(f"\nProcessing {split_name.upper()} data...")

        data = np.load(input_path, allow_pickle=True)

        feature_df = build_feature_dataset(data)

        output_path = output_folder / f"{split_name}_features.csv"

        feature_df.to_csv(output_path, index=False)

        print("Participants:", len(feature_df))
        print("Extracted features:", len(feature_df.columns) - 3)
        print("Saved:", output_path)

    print("\n--- BASELINE FEATURE EXTRACTION COMPLETED ---")
