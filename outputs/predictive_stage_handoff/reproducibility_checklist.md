# Predictive Stage Reproducibility Checklist

- [x] Patient-level train/validation/test split established before windows.
- [x] Imputation and normalization fitted on training cases only.
- [x] Sampling interval 10 seconds, history 60 seconds, horizon 30 seconds.
- [x] Primary frozen before held-out test access.
- [x] Primary is `strict_consensus`; reference is `full17_reference`.
- [x] `compact_consensus` excluded from held-out test.
- [ ] All 20 Drive checkpoint SHA256 values recorded by frozen-test preflight.
- [x] Only validation-selected `best_model.pt` is permitted.
- [x] Predictive state is explicitly not an RL-optimality claim.
- [x] No checkpoint, NPZ, or other large artifact copied into this package.
