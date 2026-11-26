
import pandas as pd

df = pd.read_csv(
    "data/processed/features/group_comparison.csv"
)

features = [
    "Mean_Fixation_Duration",
    "Std_Fixation_Duration",
    "Total_Fixations",
    "Unique_Words",
    "Mean_Gaze_X",
    "Std_Gaze_X",
    "Mean_Gaze_Y",
    "Std_Gaze_Y"
]

print("\n--- MULTI-FEATURE GROUP ANALYSIS ---")

for feature in features:

    print(f"\nFEATURE: {feature}")

    comparison = df.groupby("Group")[feature].agg(
        ["mean", "median", "std"]
    )

    print(comparison.round(2))

    difference = (
        comparison.loc[2, "mean"]
        - comparison.loc[1, "mean"]
    )

    print("Group 2 - Group 1:", round(difference, 2))

print("\n--- ANALYSIS COMPLETED ---")
