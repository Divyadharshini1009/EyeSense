
import pandas as pd
from pathlib import Path


def load_data():

    # Locate the project directory
    project_root = Path(__file__).resolve().parents[2]

    # Locate the raw dataset
    file_path = project_root / "data" / "raw" / "fused_data.csv"

    # Check whether the dataset exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Required columns
    required_columns = [
        "SubjectID",
        "Group",
        "Word_Number",
        "FIX_X",
        "FIX_Y",
        "FIX_DURATION",
        "ID",
        "Age"
    ]

    # Check for missing columns
    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("Dataset loaded successfully!")
    print("Dataset shape:", df.shape)
    print("\nFirst five records:")
    print(df.head())

    return df


if __name__ == "__main__":
    data = load_data()
