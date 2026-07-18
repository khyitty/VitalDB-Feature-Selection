# Fair PPO Attention-State Comparison

> **Legacy architecture protocol.** This frozen v1 protocol changes the feature
> extractor between `all_supported` and `attention_supported`; it therefore cannot
> support the primary scientific claim about state selection. It is retained for
> backward compatibility and optional secondary architecture analysis. The primary
> state-only protocol uses one common SB3 MLP/FlattenExtractor for
> `original_reconstructed`, `all_supported`, candidate profiles
> `prediction_minimal` and `selected_control_core`, and manifest-loaded `selected`.
> The two candidate profiles are not a declared final selected state.

The approved 102,400-step state-only pilot is versioned separately in
`configs/ppo_primary_state_pilot.json`; its audit and execution contract are in
`docs/ppo_primary_state_pilot_protocol.md`. This legacy architecture protocol and
its existing artifacts are not overwritten.

Module 6 compares four observation/encoder conditions around the same Module 5
transition, reward, action bounds, patient split, target, and exogenous
remifentanil scenario.

- `yun_reconstructed`: backward-compatible Module 5 `original_yun` state under
  an honest experiment name. Raw causal BIS and a repository 60-second window
  mean this is not a complete Yun 2023 reproduction.
- `all_supported`: all non-latent simulator-supported control observations with
  a mask-aware GRU encoder.
- `attention_supported`: exactly the same raw observations and order as
  `all_supported`, with explicit feature and temporal attention learned from RL
  reward end-to-end.
- `selected_control_aware`: simulator-supported predictive intersection plus
  protected control variables. It is not the same object as predictive
  `strict_consensus` and is not an attention-selected subset.

The main contrast is `attention_supported - all_supported`. Both use a
64-dimensional policy latent and identical SB3 actor/critic heads. Their total
trainable parameter counts must differ by no more than 10% and are persisted in
the frozen protocol directory.

The policy action is normalized to `[-1,1]` and mapped affinely to one frozen
physical action range. The main `ppo_research_v1` protocol deliberately uses the
repository's narrow `synthetic_nonclinical_v1` range, not the Yun-reported range.
This choice is not a clinical recommendation or a Yun reproduction claim.

Checkpoints are selected only from validation patient/scenario BIS MAE, with
prespecified time-in-range and action-change tie-breakers. Training return and
the test cohort cannot select a checkpoint. The held-out RL test remains sealed
throughout this module.

The virtual cohort resolves demographics from case-level fields embedded in the
modeling metadata, an explicit `--demographics-csv`, or a metadata-named source
under `--project-data-root`, in that order. It never assumes that an untracked
CSV exists inside the repository clone. In Colab, the modeling dataset is
`/content/drive/MyDrive/VitalDB-Feature-Selection/data/modeling/full` and source
discovery is rooted at `/content/drive/MyDrive/VitalDB-Feature-Selection/data`.
If those sources are absent, the training notebook explicitly enables a fallback
to the VitalDB official clinical-information endpoint
`https://api.vitaldb.net/cases`. It downloads no tracks or outcomes, filters the
response to the 98 frozen split case IDs, and atomically caches only `caseid`,
`age`, `sex`, `height`, and `weight` under `data/clinical`. The cache and response
fingerprints are recorded in a provenance sidecar. Library callers do not use
the network unless this fallback is explicitly enabled.
Test case IDs, split membership, and demographics may parameterize the frozen
virtual-patient manifest. Test trajectories, outcomes, policy rollouts, tuning,
and checkpoint selection remain prohibited and are recorded separately in
`cohort_access_manifest.json`.

Full training inventory: four conditions by seeds 7, 21, 42, 84, and 123, for 20
CUDA runs. The exact confirmation is generated from the frozen inventory and is
currently `RUN_20_PPO_CUDA_RUNS`.

Attention weights are model-internal weighting operations, not causal effects.
No predictive attention checkpoint is transferred into the control policy.
