
import numpy as np
from pathlib import Path


def main():

    print("\n--- TEMPORAL SEQUENCE VALIDATION ---")

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "splits" / "train_data.npz"

    data = np.load(data_path, allow_pickle=True)

    subject_ids = data["subject_ids"]
    sequences = data["sequences"]

    decreasing = []
    non_one_start = []
    missing_word_numbers = []

    for subject_id, sequence in zip(subject_ids, sequences):

        sequence = np.asarray(sequence, dtype=np.float32)
        word_numbers = sequence[:, 0]

        if np.any(np.diff(word_numbers) < 0):
            decreasing.append(subject_id)

        if word_numbers[0] != 1:
            non_one_start.append(subject_id)

        unique_words = np.unique(word_numbers)

        expected_words = np.arange(
            int(unique_words.min()),
            int(unique_words.max()) + 1
        )

        missing = np.setdiff1d(expected_words, unique_words)

        if len(missing) > 0:
            missing_word_numbers.append((subject_id, missing))

    print("\nTotal participants:", len(subject_ids))
    print("Decreasing word-number sequences:", len(decreasing))
    print("Sequences not starting at word 1:", len(non_one_start))
    print("Sequences with missing word numbers:", len(missing_word_numbers))

    print("\n--- EXAMPLES OF SEQUENCES WITH GAPS ---")

    for subject_id, missing in missing_word_numbers[:5]:

        index = np.where(subject_ids == subject_id)[0][0]
        sequence = np.asarray(sequences[index], dtype=np.float32)

        print("\nParticipant:", subject_id)
        print("Sequence length:", len(sequence))
        print("First word:", int(sequence[0, 0]))
        print("Last word:", int(sequence[-1, 0]))
        print("Missing words:", missing.astype(int))

    print("\n--- VALIDATION COMPLETED ---")


if __name__ == "__main__":
    main()
