
import numpy as np
from pathlib import Path


def main():

    print("\n--- TEMPORAL DATA INSPECTION ---")

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "data" / "processed" / "splits" / "train_data.npz"

    data = np.load(data_path, allow_pickle=True)

    subject_ids = data["subject_ids"]
    sequences = data["sequences"]
    ages = data["ages"]
    groups = data["groups"]

    lengths = np.array([len(seq) for seq in sequences])

    print("\nTotal participants:", len(subject_ids))
    print("Total fixation records:", int(np.sum(lengths)))

    print("\n--- SEQUENCE STATISTICS ---")
    print("Minimum length:", int(np.min(lengths)))
    print("Maximum length:", int(np.max(lengths)))
    print("Average length:", round(float(np.mean(lengths)), 2))

    print("\n--- FIRST 5 PARTICIPANTS ---")

    for i in range(min(5, len(subject_ids))):

        sequence = np.asarray(sequences[i], dtype=np.float32)

        print("\n------------------------------")
        print("Participant:", subject_ids[i])
        print("Age:", ages[i])
        print("Group:", groups[i])
        print("Sequence length:", len(sequence))

        print("\nWord_Number | FIX_X | FIX_Y | FIX_DURATION")
        print(sequence[:10])

    print("\n--- INSPECTION COMPLETED ---")


if __name__ == "__main__":
    main()
