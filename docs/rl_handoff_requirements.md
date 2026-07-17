# RL Handoff Requirements

## Legacy Notice

This document originally described the pre-simulator handoff. The repository now has
a reconstructed PK-PD simulator and Gymnasium environment. The physiological-inclusive
predictive state below is retained only to interpret historical artifacts; it is not
the final main state.

## Legacy Frozen Predictive Input

The predictive candidate is an ordered six-step history sampled every 10 seconds over 60 seconds:

`bis`, `bis_sqi`, `ppf_rate`, `ppf_volume`, `ppf_cp`, `rftn_volume`, `bis_slope`

Its future-BIS horizon is 30 seconds. Static covariates and train-fitted preprocessing are defined in `data/modeling/full/dataset_metadata.json` and `data/modeling/full/preprocessing.pkl`. Missing values use the saved train-fitted preprocessing and aligned observation mask. This is a legacy exploratory representation, not an RL-optimal or final selected state. Its prior rankings and `strict_consensus` result must not be reused for the simulator-compatible selection.

## Current Main Handoff

For end-to-end consistency, the final prediction feature universe is restricted to
variables that can also be generated causally by the reconstructed PK-PD control
simulator. The new dataset lives under `data/modeling/simulator_compatible`, uses
`bis_delta_10s` for an exact 10-second BIS change, reuses the frozen patient split,
fits preprocessing on train only, and seals test summaries during selection.

## Control-Aware State Rule

The external baseline state must be read before defining any control state. Do not replace the baseline with the seven predictive features. Preserve every baseline variable needed for actions, propofol exposure and history, BIS target or target error, remifentanil, patient context, and exogenous disturbances. Then compare the baseline, all candidate features, an attention-weighted representation, and optionally the frozen top-k representation under the same environment and 10-second action interval.

`strict_consensus` retains only `rftn_volume` from the remifentanil group. This is insufficient evidence for removing other remifentanil variables from a control state.

## Required External Inputs

Provide all of the following before RL integration:

1. Professor RL repository or attached implementation.
2. Environment class and module path.
3. `reset()` and `step()` API contracts.
4. Ordered baseline state schema and history construction.
5. Action range, unit, clipping, and 10-second hold semantics.
6. Reward definition and all coefficients.
7. Episode termination and truncation rules.
8. Patient-level train, validation, and test split identifiers.
9. RL algorithm and hyperparameters.
10. Baseline policy checkpoint and compatibility metadata.
11. Closed-loop evaluation metrics and protocol.
12. Random seeds and determinism settings.

No RL training should start until these inputs are validated and the patient split is reconciled with the predictive dataset.
