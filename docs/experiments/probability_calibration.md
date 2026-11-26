
# Probability Calibration Experiment

## Objective
To improve the reliability of Random Forest prediction
probabilities using sigmoid calibration.

## Method
- Original model: Random Forest
- Calibration method: Sigmoid (Platt scaling)
- Cross-validation folds: 5
- Calibration training data: Training split only
- Validation participants: 645
- Test participants: 645

## Test Results

| Metric | Original | Calibrated |
|---|---:|---:|
| Brier Score | 0.0770 | 0.0706 |
| Log Loss | 0.2641 | 0.2414 |
| ECE | 0.0702 | 0.0391 |

## Validation Results

| Metric | Original | Calibrated |
|---|---:|---:|
| Brier Score | 0.0863 | 0.0841 |
| Log Loss | 0.2815 | 0.2715 |
| ECE | 0.0606 | 0.0551 |

## Observations
All three calibration metrics improved on the held-out
test set after sigmoid calibration.

The calibrated model produced more reliable probability
estimates according to Brier Score, Log Loss and ECE.

## Limitations
- Calibration does not establish clinical reliability.
- Group labels and dataset provenance remain unverified.
- Calibration performance may vary across age groups.
- The experiment does not establish improved classification
  accuracy.

## Conclusion
Sigmoid calibration improved the probability calibration
of the Random Forest on the evaluated dataset.
The calibrated model was saved separately to preserve
the original model.
