
import pandas as pd
from pathlib import Path

def main():

    print("\n--- EYESENSE GROUP ANALYSIS ---")

    project_root = Path(__file__).resolve().parents[2]

    data_path = (
        project_root
        / "data"
        / "processed"
        / "fixations_cleaned.csv"
    )

    df = pd.read_csv(data_path)

    # Participant-level statistics
    participants = df.groupby("SubjectID").agg(
        Group=("Group", "first"),
        Age=("Age", "first"),
        Total_Fixations=("FIX_DURATION", "count"),
        Mean_Fixation_Duration=("FIX_DURATION", "mean"),
        Median_Fixation_Duration=("FIX_DURATION", "median"),
        Std_Fixation_Duration=("FIX_DURATION", "std"),
        Unique_Words=("Word_Number", "nunique"),
        Mean_Gaze_X=("FIX_X", "mean"),
        Mean_Gaze_Y=("FIX_Y", "mean")
    ).reset_index()

    print("\n--- PARTICIPANT COUNTS ---")
    print(participants["Group"].value_counts().sort_index())

    print("\n--- AGE DISTRIBUTION BY GROUP ---")
    print(pd.crosstab(
        participants["Age"],
        participants["Group"]
    ))

    features = [
        "Age",
        "Total_Fixations",
        "Mean_Fixation_Duration",
        "Median_Fixation_Duration",
        "Std_Fixation_Duration",
        "Unique_Words",
        "Mean_Gaze_X",
        "Mean_Gaze_Y"
    ]

    print("\n--- GROUP-WISE MEAN VALUES ---")
    print(
        participants.groupby("Group")[features]
        .mean()
        .round(2)
    )

    print("\n--- GROUP-WISE MEDIAN VALUES ---")
    print(
        participants.groupby("Group")[features]
        .median()
        .round(2)
    )

    print("\n--- ANALYSIS COMPLETED ---")

    output_path = (
        project_root
        / "data"
        / "processed"
        / "features"
        / "group_comparison.csv"
    )

    participants.to_csv(output_path, index=False)

    print("\nParticipant analysis saved to:", output_path)


if __name__ == "__main__":
    main()
