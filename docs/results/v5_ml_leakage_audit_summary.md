# V5 Machine-learning Leakage Audit

## Decision

**PASS.** The existing RQ3 machine-learning results do not use held-out test
labels to select the F1 decision threshold.

## Verified procedure

For every outer fold:

1. match events were divided into outer-training and outer-test groups;
2. hyperparameters were selected by PR-AUC from three-fold grouped inner
   out-of-fold predictions inside outer-training data;
3. the F1 threshold was selected from those same inner out-of-fold predictions;
4. the selected model was refitted on the complete outer-training fold;
5. the frozen threshold was applied to outer-test probabilities;
6. outer-test labels were used only to calculate final evaluation metrics.

Scaling for logistic and elastic-net models is contained in a fitted pipeline
inside each training fold. The prediction sample includes only information
available by five minutes after the initiating trade; later reversal fields are
prohibited predictors.

## Artifact checks

- Sample: 5,843 rows, 104 match events.
- Linear/logistic/elastic-net fold models checked: 20.
- Tree fold models checked: 20.
- Random-forest ablation fold models checked: 15.
- Chronological holdout models checked: 4.
- Maximum difference between stored test threshold and corresponding
  training-OOF threshold: 0.
- Duplicate outer held-out rows within a specification: 0.
- Events appearing in more than one outer fold: 0.
- Chronological split: 78 training events, 26 test events, overlap 0.

The chronological holdout likewise tunes hyperparameters and thresholds only
within the earlier 75% training events before evaluating the latest 25%.

## Consequence

No ML rerun is required for threshold leakage. PR-AUC remains the primary metric;
F1 is supporting evidence because it depends on a selected decision threshold.
The models predict thirty-minute full price reversal, not manipulation or
misconduct.

## Reproducibility

- Audit script: `scripts/audit_v5_ml_leakage.py`
- Machine-readable audit: `docs/data_audit/v5/ml_threshold_leakage_audit.json`
- Frozen sample: `model_samples/v4/rq3_ml/rq3_match_pre_full_reversal_30m.csv.gz`

