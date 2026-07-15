# RL Handoff Requirements

## Current Status

The repository does not contain the professor's propofol-control implementation. No Gymnasium environment, PK-PD simulator, action contract, reward, episode termination logic, RL agent, or baseline policy checkpoint was found. The `main.py` propofol logic only crops observational records to the propofol administration period.

RL implementation and training are therefore blocked. The missing environment must not be reconstructed from assumptions or from unavailable VitalDB tracks.

## Frozen Predictive Input

The predictive candidate is an ordered six-step history sampled every 10 seconds over 60 seconds:

`bis`, `bis_sqi`, `ppf_rate`, `ppf_volume`, `ppf_cp`, `rftn_volume`, `bis_slope`

Its future-BIS horizon is 30 seconds. Static covariates and train-fitted preprocessing are defined in `data/modeling/full/dataset_metadata.json` and `data/modeling/full/preprocessing.pkl`. Missing values must use the saved train-fitted preprocessing and aligned observation mask. This is a predictive representation, not an RL-optimal state.

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
