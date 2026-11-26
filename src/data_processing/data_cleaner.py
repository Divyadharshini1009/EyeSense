
import pandas as pd
from pathlib import Path


def clean_data(df):

    print("\n--- DATA CLEANING STARTED ---")

    # 1. Remove unnecessary exported index columns
    index_columns = [
        col for col in df.columns
        if col.startswith("Unnamed:")
    ]

    df = df.drop(columns=index_columns)

    print("Removed index columns:", index_columns)

    # 2. Clean participant identifiers
    df["SubjectID"] = df["SubjectID"].astype(str).str.strip()

    # 3. Convert numerical columns
    numeric_columns = [
        "Group",
        "Word_Number",
        "FIX_X",
        "FIX_Y",
        "FIX_DURATION",
        "ID",
        "Age"
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 4. Check missing values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # 5. Remove rows missing essential information
    required_columns = [
        "SubjectID",
        "Group",
        "Word_Number",
        "FIX_X",
        "FIX_Y",
        "FIX_DURATION",
        "Age"
    ]

    before = len(df)

    df = df.dropna(subset=required_columns).copy()

    print("\nRows removed:", before - len(df))

    # 6. Dataset summary
    print("\n--- CLEANED DATA SUMMARY ---")

    print("Total fixation records:", len(df))
    print("Unique participants:", df["SubjectID"].nunique())

    # 7. Age distribution
    print("\nAge distribution:")

    age_distribution = (
        df.groupby("SubjectID")["Age"]
        .first()
        .value_counts()
        .sort_index()
    )

    print(age_distribution)

    # 8. Group distribution
    print("\nGroup distribution:")

    group_distribution = (
        df.groupby("SubjectID")["Group"]
        .first()
        .value_counts()
        .sort_index()
    )

    print(group_distribution)

    # 9. Check participant consistency
    inconsistent_groups = (
        df.groupby("SubjectID")["Group"]
        .nunique()
    )

    inconsistent_groups = inconsistent_groups[
        inconsistent_groups > 1
    ]

    inconsistent_ages = (
        df.groupby("SubjectID")["Age"]
        .nunique()
    )

    inconsistent_ages = inconsistent_ages[
        inconsistent_ages > 1
    ]

    print(
        "\nParticipants with inconsistent group labels:",
        len(inconsistent_groups)
    )

    print(
        "Participants with inconsistent ages:",
        len(inconsistent_ages)
    )

    # 10. Display sample participant IDs
    print("\nFirst 20 participant IDs:")

    print(
        df["SubjectID"]
        .drop_duplicates()
        .head(20)
        .tolist()
    )

    # 11. Display missing ID information
    print("\nMissing ID values:", df["ID"].isna().sum())

    print("\n--- DATA CLEANING COMPLETED ---")

    return df


if __name__ == "__main__":

    from data_loader import load_data

    # Load original dataset
    data = load_data()

    # Clean dataset
    cleaned_data = clean_data(data)

    # Locate project directory
    project_root = Path(__file__).resolve().parents[2]

    # Create processed folder if necessary
    output_folder = project_root / "data" / "processed"

    output_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save cleaned dataset
    output_path = output_folder / "fixations_cleaned.csv"

    cleaned_data.to_csv(
        output_path,
        index=False
    )

    print("\nCleaned dataset saved successfully!")
    print("Saved location:", output_path)
