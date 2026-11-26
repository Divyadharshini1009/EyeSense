
import pandas as pd
from pathlib import Path


if __name__ == "__main__":

    print("\n--- AGE DISTRIBUTION ANALYSIS STARTED ---")

    root = Path(__file__).resolve().parents[2]

    input_path = (
        root / "data" / "processed" /
        "fixations_cleaned.csv"
    )

    output_folder = (
        root / "data" / "processed" / "evaluation"
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    # Load cleaned fixation dataset
    df = pd.read_csv(input_path)

    # Keep one record per participant
    participants = df[
        ["SubjectID", "Group", "Age"]
    ].drop_duplicates()

    # Verify that each participant has only one age and group
    duplicates = participants[
        participants.duplicated("SubjectID", keep=False)
    ]

    if not duplicates.empty:
        raise ValueError(
            "Some participants have inconsistent Age or Group values."
        )

    print("\nTotal participants:", len(participants))

    # Display participant counts by group
    print("\nParticipant counts by group:")
    print(participants["Group"].value_counts().sort_index())

    # Age distribution: participant counts
    age_counts = pd.crosstab(
        participants["Age"],
        participants["Group"]
    ).sort_index()

    print("\nAge distribution (participant counts):")
    print(age_counts)

    # Percentage distribution within each group
    age_percentages = (
        pd.crosstab(
            participants["Age"],
            participants["Group"],
            normalize="columns"
        ) * 100
    ).round(2)

    print("\nAge distribution (percentage within each group):")
    print(age_percentages)

    # Summary statistics
    print("\nAge summary by group:")
    print(
        participants.groupby("Group")["Age"].agg(
            ["count", "mean", "std", "min", "max"]
        ).round(2)
    )

    # Save results
    age_counts.to_csv(
        output_folder / "age_distribution_counts.csv"
    )

    age_percentages.to_csv(
        output_folder / "age_distribution_percentages.csv"
    )

    print("\nAnalysis files saved in:")
    print(output_folder)

    print("\n--- AGE DISTRIBUTION ANALYSIS COMPLETED ---")
