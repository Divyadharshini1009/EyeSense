
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_data(sequences):

    print("\n--- DATA SPLITTING STARTED ---")

    # 1. Extract participant information
    subject_ids = np.array([
        item["SubjectID"] for item in sequences
    ])

    groups = np.array([
        item["Group"] for item in sequences
    ])

    ages = np.array([
        item["Age"] for item in sequences
    ])

    fixation_sequences = np.array(
        [item["Sequence"] for item in sequences],
        dtype=object
    )

    # 2. Check participant uniqueness
    if len(subject_ids) != len(set(subject_ids)):
        raise ValueError(
            "Duplicate SubjectIDs found. Each participant "
            "must have exactly one sequence."
        )

    # 3. Check group distribution
    unique_groups, group_counts = np.unique(
        groups,
        return_counts=True
    )

    if len(unique_groups) < 2:
        raise ValueError(
            "At least two groups are required for stratified splitting."
        )

    if group_counts.min() < 2:
        raise ValueError(
            "Not enough participants in one of the groups."
        )

    print("Total participants:", len(subject_ids))

    print("\nOriginal group distribution:")
    for group, count in zip(unique_groups, group_counts):
        print(f"Group {group}: {count} participants")

    # 4. Split into training (70%) and temporary (30%)
    all_indices = np.arange(len(sequences))

    train_indices, temp_indices = train_test_split(
        all_indices,
        test_size=0.30,
        random_state=42,
        stratify=groups
    )

    # 5. Split temporary data into validation (15%)
    # and testing (15%)
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.50,
        random_state=42,
        stratify=groups[temp_indices]
    )

    # 6. Create final splits
    splits = {
        "train": train_indices,
        "validation": val_indices,
        "test": test_indices
    }

    # 7. Verify no participant overlap
    train_ids = set(subject_ids[train_indices])
    val_ids = set(subject_ids[val_indices])
    test_ids = set(subject_ids[test_indices])

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)

    print("\n--- SPLIT SUMMARY ---")

    for split_name, indices in splits.items():

        print(f"\n{split_name.upper()} SET")
        print("Participants:", len(indices))

        split_groups, split_counts = np.unique(
            groups[indices],
            return_counts=True
        )

        for group, count in zip(split_groups, split_counts):
            print(f"Group {group}: {count}")

    print("\nParticipant overlap: 0")
    print("No participant appears in multiple splits.")

    print("\n--- DATA SPLITTING COMPLETED ---")

    return {
        name: {
            "sequences": fixation_sequences[indices],
            "subject_ids": subject_ids[indices],
            "ages": ages[indices],
            "groups": groups[indices]
        }
        for name, indices in splits.items()
    }


if __name__ == "__main__":

    # Locate project directory
    project_root = Path(__file__).resolve().parents[2]

    # Load participant sequences
    input_path = (
        project_root
        / "data"
        / "processed"
        / "participant_sequences.npz"
    )

    data = np.load(input_path, allow_pickle=True)

    # Reconstruct participant records
    sequences = []

    for i in range(len(data["subject_ids"])):

        sequences.append({
            "SubjectID": str(data["subject_ids"][i]),
            "Age": int(data["ages"][i]),
            "Group": int(data["groups"][i]),
            "Sequence_Length": int(
                data["sequence_lengths"][i]
            ),
            "Sequence": data["sequences"][i]
        })

    # Perform participant-level splitting
    split_results = split_data(sequences)

    # Create output folder
    output_folder = (
        project_root
        / "data"
        / "processed"
        / "splits"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save each split separately
    for split_name, split_data_result in split_results.items():

        output_path = (
            output_folder / f"{split_name}_data.npz"
        )

        np.savez_compressed(
            output_path,
            sequences=split_data_result["sequences"],
            subject_ids=split_data_result["subject_ids"],
            ages=split_data_result["ages"],
            groups=split_data_result["groups"]
        )

        print(f"\nSaved {split_name} data:")
        print(output_path)

    print("\nAll dataset splits saved successfully!")
