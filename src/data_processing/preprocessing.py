
import pandas as pd
import numpy as np
from pathlib import Path


def preprocess_data(df):

    print("\n--- PREPROCESSING STARTED ---")

    # 1. Create a copy to protect the original data
    df = df.copy()

    # 2. Remove ID because it is incomplete and unsuitable
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # 3. Preserve the original fixation order
    df["Fixation_Order"] = np.arange(len(df))

    # 4. Define required columns
    required_columns = [
        "SubjectID",
        "Group",
        "Age",
        "Word_Number",
        "FIX_X",
        "FIX_Y",
        "FIX_DURATION"
    ]

    # 5. Verify required columns
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # 6. Convert numerical features
    numeric_columns = [
        "Group",
        "Age",
        "Word_Number",
        "FIX_X",
        "FIX_Y",
        "FIX_DURATION"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # 7. Remove records with invalid essential values
    before = len(df)

    df = df.dropna(
        subset=required_columns
    ).copy()

    print(
        "Rows removed due to missing values:",
        before - len(df)
    )

    # 8. Remove infinite numerical values
    feature_columns = [
        "Age",
        "Word_Number",
        "FIX_X",
        "FIX_Y",
        "FIX_DURATION"
    ]

    finite_mask = np.isfinite(
        df[feature_columns]
    ).all(axis=1)

    before = len(df)
    df = df.loc[finite_mask].copy()

    print(
        "Rows removed due to infinite values:",
        before - len(df)
    )

    # 9. Remove invalid fixation durations
    before = len(df)

    df = df[df["FIX_DURATION"] > 0].copy()

    print(
        "Rows removed due to invalid duration:",
        before - len(df)
    )

    # 10. Keep records in original fixation order
    df = df.sort_values(
        "Fixation_Order",
        kind="stable"
    ).reset_index(drop=True)

    # 11. Display preprocessing summary
    print("\n--- PREPROCESSING SUMMARY ---")

    print("Remaining fixation records:", len(df))

    print(
        "Unique participants:",
        df["SubjectID"].nunique()
    )

    print(
        "Age range:",
        df["Age"].min(),
        "to",
        df["Age"].max()
    )

    print(
        "Group values:",
        sorted(df["Group"].unique())
    )

    print("\nFeature columns:")
    print(feature_columns)

    print("\nFirst five preprocessed records:")
    print(df.head())

    print("\n--- PREPROCESSING COMPLETED ---")

    return df


if __name__ == "__main__":

    # Locate project directory
    project_root = Path(__file__).resolve().parents[2]

    # Read cleaned dataset
    input_path = (
        project_root
        / "data"
        / "processed"
        / "fixations_cleaned.csv"
    )

    df = pd.read_csv(input_path)

    # Preprocess data
    processed_data = preprocess_data(df)

    # Create output directory
    output_folder = (
        project_root
        / "data"
        / "processed"
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save preprocessed dataset
    output_path = (
        output_folder
        / "fixations_preprocessed.csv"
    )

    processed_data.to_csv(
        output_path,
        index=False
    )

    print("\nPreprocessed dataset saved successfully!")
    print("Saved location:", output_path)
