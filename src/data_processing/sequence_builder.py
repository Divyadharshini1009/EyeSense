
import pandas as pd
import numpy as np
from pathlib import Path


def build_sequences(df):

    print("\n--- SEQUENCE BUILDING STARTED ---")

    # 1. Create a copy of the dataset
    df = df.copy()

    # 2. Check required columns
    required_columns = [
        "SubjectID",
        "Group",
        "Age",
        "Word_Number",
        "FIX_X",
        "FIX_Y",
        "FIX_DURATION",
        "Fixation_Order"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 3. Sort records by participant and original fixation order
    df = df.sort_values(
        ["SubjectID", "Fixation_Order"],
        kind="stable"
    ).reset_index(drop=True)

    # 4. Group fixation records by participant
    sequences = []

    for subject_id, participant_data in df.groupby(
        "SubjectID",
        sort=False
    ):

        participant_data = participant_data.sort_values(
            "Fixation_Order",
            kind="stable"
        )

        # Participant information
        age = participant_data["Age"].iloc[0]
        group = participant_data["Group"].iloc[0]

        # Extract fixation features in reading order
        fixation_sequence = participant_data[
            [
                "Word_Number",
                "FIX_X",
                "FIX_Y",
                "FIX_DURATION"
            ]
        ].to_numpy(dtype=np.float32)

        # Store participant sequence
        sequences.append({
            "SubjectID": subject_id,
            "Age": age,
            "Group": group,
            "Sequence_Length": len(fixation_sequence),
            "Sequence": fixation_sequence
        })

    # 5. Display sequence-building summary
    print("\n--- SEQUENCE SUMMARY ---")

    print("Total participants:", len(sequences))

    lengths = [
        item["Sequence_Length"]
        for item in sequences
    ]

    print("Total fixation records:", sum(lengths))
    print("Shortest sequence:", min(lengths))
    print("Longest sequence:", max(lengths))
    print("Average sequence length:", round(np.mean(lengths), 2))

    print("\nFirst five participant sequences:")

    for item in sequences[:5]:
        print(
            item["SubjectID"],
            "| Age:", item["Age"],
            "| Group:", item["Group"],
            "| Fixations:", item["Sequence_Length"]
        )

    print("\n--- SEQUENCE BUILDING COMPLETED ---")

    return sequences


if __name__ == "__main__":

    # Locate project directory
    project_root = Path(__file__).resolve().parents[2]

    # Load preprocessed dataset
    input_path = (
        project_root
        / "data"
        / "processed"
        / "fixations_preprocessed.csv"
    )

    df = pd.read_csv(input_path)

    # Build participant sequences
    sequences = build_sequences(df)

    # Save sequences in a structured format
    output_folder = (
        project_root
        / "data"
        / "processed"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        output_folder
        / "participant_sequences.npz"
    )

    # Store variable-length sequences and metadata
    np.savez_compressed(
        output_path,
        sequences=np.array(
            [item["Sequence"] for item in sequences],
            dtype=object
        ),
        subject_ids=np.array(
            [item["SubjectID"] for item in sequences]
        ),
        ages=np.array(
            [item["Age"] for item in sequences]
        ),
        groups=np.array(
            [item["Group"] for item in sequences]
        ),
        sequence_lengths=np.array(
            [item["Sequence_Length"] for item in sequences]
        )
    )

    print("\nParticipant sequences saved successfully!")
    print("Saved location:", output_path)
