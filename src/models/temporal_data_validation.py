
import numpy as np
from pathlib import Path


def inspect_split(file_path, split_name):

    data = np.load(file_path, allow_pickle=True)

    sequences = data["sequences"]
    groups = data["groups"]
    subject_ids = data["subject_ids"]

    lengths = np.array([len(seq) for seq in sequences])

    print(f"\n--- {split_name.upper()} DATA ---")

    print("Participants:", len(sequences))
    print("Group 1:", np.sum(groups == 1))
    print("Group 2:", np.sum(groups == 2))

    print("Minimum sequence length:", lengths.min())
    print("Maximum sequence length:", lengths.max())
    print("Average sequence length:", round(lengths.mean(), 2))

    all_finite = True
    feature_min = np.full(4, np.inf)
    feature_max = np.full(4, -np.inf)

    invalid_sequences = 0
    unordered_sequences = 0

    for seq in sequences:

        seq = np.asarray(seq, dtype=np.float64)

        if seq.ndim != 2 or seq.shape[1] != 4:
            invalid_sequences += 1
            continue

        if not np.isfinite(seq).all():
            all_finite = False
            invalid_sequences += 1
            continue

        feature_min = np.minimum(
            feature_min, seq.min(axis=0)
        )

        feature_max = np.maximum(
            feature_max, seq.max(axis=0)
        )

        # Check whether Word_Number moves backwards.
        if np.any(np.diff(seq[:, 0]) < 0):
            unordered_sequences += 1

    print("Invalid sequences:", invalid_sequences)
    print("Sequences with decreasing Word_Number:",
          unordered_sequences)
    print("All values finite:", all_finite)

    print("\nFeature ranges:")

    feature_names = [
        "Word_Number",
        "FIX_X",
        "FIX_Y",
        "FIX_DURATION"
    ]

    for i, name in enumerate(feature_names):

        print(
            f"{name}: "
            f"min={feature_min[i]:.2f}, "
            f"max={feature_max[i]:.2f}"
        )

    print("Unique participant IDs:", len(set(subject_ids)))


if __name__ == "__main__":

    print("\n--- EYESENSE TEMPORAL DATA VALIDATION ---")

    root = Path(__file__).resolve().parents[2]

    folder = root / "data" / "processed" / "splits"

    inspect_split(folder / "train_data.npz", "Training")
    inspect_split(folder / "validation_data.npz", "Validation")
    inspect_split(folder / "test_data.npz", "Test")

    print("\n--- VALIDATION COMPLETED ---")
