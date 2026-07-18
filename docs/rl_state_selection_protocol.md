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

The canonical comparison profiles are `original_reconstructed`, `all_supported`,
`prediction_minimal`, `selected_control_core`, and manifest-loaded `selected`.
`prediction_minimal` contains BIS and `bis_delta_10s` plus demographics and is a
minimal stable 30-second prediction candidate without direct drug state; its control
adequacy is unconfirmed. `selected_control_core` adds current propofol/remifentanil
rates and simulator-generated propofol/remifentanil Cp as a control-oriented
candidate. Neither candidate is the declared final selected state.

A selected manifest records exact feature order,
selection provenance, split, seeds, patient/feature aggregation, threshold or top-k
rule, timestamp, and canonical commit. A pending manifest cannot execute. Test-split
selection and simulator-unsupported features fail validation.

The final selected set and stability threshold are not yet decided. Prediction
attention must be aggregated across predefined seeds and patients, and its current
cross-seed instability must remain visible. Attention is not a causal effect.

For end-to-end consistency, the final prediction feature universe is restricted to
variables that can also be generated causally by the reconstructed PK-PD control
simulator. The exact shared dynamic universe has 13 candidates: BIS, `bis_delta_10s`,
fixed-target BIS error, current propofol/remifentanil rates, causal 60-second recent
doses, causal cumulative doses, and Schnider/Minto Cp/Ce reconstructed from rate
history and demographics. Static context is age, sex, height, and weight. `bis_delta_10s`
always means `BIS[t] - BIS[t-10 seconds]`; the ambiguous historical `bis_slope` name
is prohibited in the main profile.

HR/PLETH_HR, invasive/noninvasive blood pressure, SpO2, ETCO2, respiratory signals,
HRV, PLETH-derived signals, and BIS SQI are ineligible. Recorded Orchestra CP/CE are
device-reported TCI estimates used only to audit agreement with the canonical
reconstructions; they are not direct measurements or candidate definitions. CT is
not Cp or Ce. `original_reconstructed` remains Cp/Ce-free, while `all_supported`
contains all four and `selected` may retain or remove them. Prior attention,
ablation, `strict_consensus`, and frozen-candidate outputs are legacy exploratory
evidence and cannot supply the final selected manifest.

Every primary profile uses the same six decision points spanning 60 seconds, the
same temporal validity mask, demographics ordered as age/sex/height/weight, and the
same fixed physical scales. The state-only PPO comparison holds simulator dynamics,
action and reward definitions, common MLP architecture, hyperparameters, budget,
seed, cohort, and evaluation fixed.

VitalDB metadata establishes that Orchestra CP/CE are TCI-pump estimates, but it does
not establish exact equivalence of pump model version, covariates, initialization,
or effect-site implementation. The build therefore quantifies agreement without
using the recorded estimates as canonical inputs.

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
