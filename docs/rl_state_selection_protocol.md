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

## Reconstruction boundary

The PK-PD simulator is reconstructed from published Schnider, Minto, and Yun
equations/settings. The environment currently uses repository-defined nonclinical
action bounds, repository-defined reward, and deterministic synthetic remifentanil
schedules. These choices are shared across profiles but are not claimed to reproduce
unpublished Yun code or clinical practice.

## Leakage and clinical limits

Prediction preprocessing is fit on training patients only, and state selection must
not use test labels. Existing prediction test data have already been inspected and
are not a pristine external holdout. Physiological tracks without validated
action-conditioned simulator transitions remain prediction-only. This software is
research-only and must not be used for clinical dosing.
