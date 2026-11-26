
import pandas as pd

# Load participant-level data
df = pd.read_csv(
    "data/processed/features/group_comparison.csv"
)

print("\n--- AGE-WISE GROUP COMPARISON ---")

# Compare only ages with sufficient participants in both groups
for age in range(7, 13):

    age_data = df[df["Age"] == age]

    group_1 = age_data[age_data["Group"] == 1]
    group_2 = age_data[age_data["Group"] == 2]

    print(f"\nAGE: {age}")
    print(f"Group 1 participants: {len(group_1)}")
    print(f"Group 2 participants: {len(group_2)}")

    if len(group_1) == 0 or len(group_2) == 0:
        print("Insufficient data for comparison.")
        continue

    print(
        "Group 1 mean fixation duration:",
        round(group_1["Mean_Fixation_Duration"].mean(), 2)
    )

    print(
        "Group 2 mean fixation duration:",
        round(group_2["Mean_Fixation_Duration"].mean(), 2)
    )

    difference = (
        group_2["Mean_Fixation_Duration"].mean()
        - group_1["Mean_Fixation_Duration"].mean()
    )

    print("Difference (Group 2 - Group 1):", round(difference, 2))

print("\n--- ANALYSIS COMPLETED ---")
