
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu

df = pd.read_csv(
    "data/processed/features/group_comparison.csv"
)

results = []

print("\n--- AGE-WISE STATISTICAL TESTING ---")

for age in range(7, 13):

    age_data = df[df["Age"] == age]

    g1 = age_data[
        age_data["Group"] == 1
    ]["Mean_Fixation_Duration"].dropna()

    g2 = age_data[
        age_data["Group"] == 2
    ]["Mean_Fixation_Duration"].dropna()

    if len(g1) < 2 or len(g2) < 2:
        continue

    # Mann-Whitney U test
    statistic, p_value = mannwhitneyu(
        g1, g2, alternative="two-sided"
    )

    # Cohen's d
    pooled_sd = np.sqrt(
        (
            (len(g1) - 1) * g1.var()
            + (len(g2) - 1) * g2.var()
        )
        / (len(g1) + len(g2) - 2)
    )

    cohens_d = (
        (g2.mean() - g1.mean()) / pooled_sd
        if pooled_sd > 0 else np.nan
    )

    results.append({
        "Age": age,
        "Group_1_N": len(g1),
        "Group_2_N": len(g2),
        "Group_1_Mean": g1.mean(),
        "Group_2_Mean": g2.mean(),
        "Mean_Difference": g2.mean() - g1.mean(),
        "Mann_Whitney_P": p_value,
        "Cohens_D": cohens_d
    })

results_df = pd.DataFrame(results)

# Holm correction for six age-wise tests
p_values = results_df["Mann_Whitney_P"].values
order = np.argsort(p_values)
adjusted = np.empty(len(p_values))

running_max = 0

for rank, index in enumerate(order):
    adjusted_value = (len(p_values) - rank) * p_values[index]
    running_max = max(running_max, adjusted_value)
    adjusted[index] = min(running_max, 1.0)

results_df["Holm_Adjusted_P"] = adjusted

results_df["Significant"] = (
    results_df["Holm_Adjusted_P"] < 0.05
)

print(results_df.round(5).to_string(index=False))

output_path = (
    "data/processed/features/"
    "group_significance_results.csv"
)

results_df.to_csv(output_path, index=False)

print("\n--- TESTING COMPLETED ---")
print("Results saved to:", output_path)
