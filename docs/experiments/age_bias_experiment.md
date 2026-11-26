
# Age Bias Experiment

## Objective
To investigate the influence of participant age on
the Random Forest classification performance.

## Experimental Setup
- Model: Random Forest Classifier
- Training participants: 3008
- Validation participants: 645
- Test participants: 645
- Random state: 42
- Trees: 200
- Class weighting: Balanced

Two experiments were conducted:
1. Random Forest with all 13 features, including Age.
2. Random Forest with 12 features, excluding Age.

The same participant splits and hyperparameters were used.

## Test Results

| Metric | With Age | Without Age |
|---|---:|---:|
| Accuracy | 0.9070 | 0.8031 |
| Balanced Accuracy | 0.8930 | 0.7544 |
| Group 2 Precision | 0.8732 | 0.7674 |
| Group 2 Recall | 0.8493 | 0.6027 |
| Group 2 F1 | 0.8611 | 0.6752 |
| ROC-AUC | 0.9623 | 0.8788 |

## Observations
Removing Age reduced classification performance across
all evaluated metrics.

However, the model retained predictive ability using
fixation-related features alone.

Age therefore appears to contribute substantially to
the model's predictions.

## Limitations
- Group 1 and Group 2 meanings remain unverified.
- Dataset provenance requires confirmation.
- The experiment does not establish that Age causes
  the observed performance difference.
- Results cannot currently be interpreted as clinical
  dyslexia detection.

## Conclusion
Age is an influential feature in the current dataset.
Further investigation is required to distinguish
developmental effects from dataset-related patterns.
