# Fair PPO Attention-State Comparison

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

Full training inventory: four conditions by seeds 7, 21, 42, 84, and 123, for 20
CUDA runs. The exact confirmation is generated from the frozen inventory and is
currently `RUN_20_PPO_CUDA_RUNS`.

Attention weights are model-internal weighting operations, not causal effects.
No predictive attention checkpoint is transferred into the control policy.
