# Frozen Predictive Decision: 30-second BIS Horizon

## Decision

Before opening the held-out test split, `strict_consensus` is frozen as the primary predictive candidate and `full17_reference` as its reference. The held-out evaluation is limited to these two candidates, GRU and explicit-attention models, and seeds 7, 21, 42, 84, and 123. It must use each run's validation-selected `best_model.pt`; no retraining, checkpoint reselection, or test-set tuning is permitted.

The primary candidate has seven ordered dynamic features:

1. `bis`
2. `bis_sqi`
3. `ppf_rate`
4. `ppf_volume`
5. `ppf_cp`
6. `rftn_volume`
7. `bis_slope`

The reference is the ordered 17-feature state: `bis`, `bis_sqi`, `hr`, `mbp`, `sbp`, `dbp`, `spo2`, `etco2`, `ppf_rate`, `ppf_volume`, `ppf_cp`, `ppf_ce`, `rftn_rate`, `rftn_volume`, `rftn_cp`, `rftn_ce`, and `bis_slope`.

## Validation Evidence

Across five paired seeds, the GRU mean patient-level validation MAE was 3.368612 for `strict_consensus` and 3.393851 for `full17_reference`. The paired strict-minus-full17 mean delta was -0.025239, and strict improved in all five seeds. It was the best observed and simplest non-dominated GRU candidate and remained non-dominated for Attention.

`compact_consensus` had the lowest mean Attention validation MAE (3.362493), but it is frozen as secondary validation-only evidence. It is excluded from held-out test evaluation and cannot replace the primary based on test results.

The source analysis manifest is `outputs/frozen_candidate_retraining_validation_only/analysis/analysis_manifest.json`, SHA256 `1a2e90aa7b96d793aa9286ea802a0d7e5d78dc7b0923f109af8cb80b434ec23a`. That analysis covered 50 validation-only runs and recorded that the held-out test split remained sealed and unread.

## Interpretation Limits

This decision cannot be changed after test results are seen. Small test differences must not be presented as automatically clinically meaningful, and no p-value is an automatic winner rule.

Predictive utility for future BIS does not establish an RL-optimal control state. In particular, `strict_consensus` retains only `rftn_volume` from the remifentanil-related variables. The later control-aware state must preserve all action, drug-history, BIS-target, remifentanil, and exogenous variables required by the professor's external RL baseline.

The planned evaluation uses an internal held-out patient split. It is not pristine external validation.
