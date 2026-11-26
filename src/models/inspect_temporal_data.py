
import numpy as np
from pathlib import Path


if __name__ == "__main__":

    print("\n--- TEMPORAL DATA INSPECTION ---")

    root = Path(__file__).resolve().parents[2]

    split_folder = (
        root / "data" / "processed" / "splits"
    )

    files = [
        "train_data.npz",
        "validation_data.npz",
        "test_data.npz"
    ]

    for filename in files:

        path = split_folder / filename

        print(f"\n{'=' * 50}")
        print("FILE:", filename)

        if not path.exists():
            print("ERROR: File not found!")
            continue

        data = np.load(path, allow_pickle=True)

        print("\nAvailable keys:", data.files)

        for key in data.files:

            array = data[key]

            print(f"\nKey: {key}")
            print("Shape:", array.shape)
            print("Data type:", array.dtype)

            if array.size > 0:

                if array.dtype.kind in "iuf":
                    print("Minimum:", np.min(array))
                    print("Maximum:", np.max(array))

                    if array.ndim == 1:
                        print("First 10 values:", array[:10])

                else:
                    print("First value:", array.flat[0])

    print("\n--- INSPECTION COMPLETED ---")
