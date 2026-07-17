# RL State-Selection Protocol

## Primary claim

The primary experiment tests whether a compact, data-driven state can maintain or
improve control under one fixed downstream PPO implementation. Only the observation
feature set may differ. Simulator dynamics, patient/scenario, synthetic exogenous
remifentanil schedule, action, reward, policy and feature-extractor architecture,
PPO hyperparameters, budget, seeds, and evaluation must remain identical.

The common policy contract is SB3 `MlpPolicy` with `FlattenExtractor` and identical
actor/critic hidden layers. The existing GRU and RL attention extractors are legacy
or secondary architecture experiments and cannot establish state-selection
superiority.

## State manifests

The canonical profiles are `original_reconstructed`, `all_supported`, and
manifest-loaded `selected`. A selected manifest records exact feature order,
selection provenance, split, seeds, patient/feature aggregation, threshold or top-k
rule, timestamp, and canonical commit. A pending manifest cannot execute. Test-split
selection and simulator-unsupported features fail validation.

The final selected set and stability threshold are not yet decided. Prediction
attention must be aggregated across predefined seeds and patients, and its current
cross-seed instability must remain visible. Attention is not a causal effect.

For end-to-end consistency, the final prediction feature universe is restricted to
variables that can also be generated causally by the reconstructed PK-PD control
simulator. The exact shared dynamic universe is BIS, `bis_delta_10s`, fixed-target BIS
error, current propofol/remifentanil rates, causal 60-second recent doses, and causal
cumulative doses. Static context is age, sex, height, and weight. `bis_delta_10s`
always means `BIS[t] - BIS[t-10 seconds]`; the ambiguous historical `bis_slope` name
is prohibited in the main profile.

HR/PLETH_HR, invasive/noninvasive blood pressure, SpO2, ETCO2, respiratory signals,
HRV, PLETH-derived signals, and BIS SQI are ineligible. Recorded Orchestra CP/CE
tracks are not final candidates because they are not reconstructed with the same
repository PK-PD implementation during prediction preprocessing. Prior attention,
ablation, `strict_consensus`, and frozen-candidate outputs are legacy exploratory
evidence and cannot supply the final selected manifest.

## Reconstruction boundary

The PK-PD simulator is reconstructed from published Schnider, Minto, and Yun
equations/settings. The environment currently uses repository-defined nonclinical
action bounds, repository-defined reward, and deterministic synthetic remifentanil
schedules. These choices are shared across profiles but are not claimed to reproduce
unpublished Yun code or clinical practice.

## Leakage and clinical limits

Prediction preprocessing is fit on training patients only, and state selection must
not use test labels. The simulator-compatible build reuses the existing case split and
seals test target summaries. Existing prediction test data have already been inspected
and are not a pristine external holdout. This software is research-only and must not
be used for clinical dosing.
