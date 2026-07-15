# VitalDB Future-BIS Modeling Dataset

This repository separates raw data preparation from modeling-data construction:

- `main.py` downloads VitalDB tracks and performs initial signal-quality filtering,
  limited within-case forward filling, propofol-period cropping, and clinical-data merging.
- `scripts/build_prediction_dataset.py` reads the cleaned CSV, creates patient-level
  splits, resamples each case independently, fits preprocessing on training cases only,
  and constructs future-BIS prediction windows.

The default pilot uses 10-second samples. A 60-second history contains six observations
at `t-50, t-40, t-30, t-20, t-10, t`; the regression target is raw BIS exactly at
`t+30`. Classification labels indicate whether that same future BIS is above 60 or
below 40. Case IDs are split before any modeling windows are generated, so no case is
shared by train, validation, and test.

The static sex input uses the cleaned dataset's existing `sex_male` encoding
(`0` = female, `1` = male). It is imputed consistently if needed but is not standardized.

## Commands

Build the first 10 eligible cases:

```powershell
python scripts/build_prediction_dataset.py --pilot
```

Build all eligible cases (not run as part of the initial pipeline task):

```powershell
python scripts/build_prediction_dataset.py --full
```

Run the synthetic test suite:

```powershell
python -m pytest -q
```

Pilot arrays are saved to `data/modeling/pilot/{train,val,test}.npz`; matching window
metadata, split case lists, feature/preprocessing metadata, and the dataset report are
saved in the same directory.

Audit a completed full build without modifying its modeling arrays:

```powershell
python scripts/audit_prediction_dataset.py --dataset-dir data/modeling/full
```

The audit writes `full_dataset_audit.json` and `case_level_target_summary.csv` in the
full dataset directory.

## Baselines

Evaluate the latest-observed-BIS persistence baseline:

```powershell
python scripts/run_baselines.py persistence --dataset-dir data/modeling/full
```

Run the complete GRU pipeline on a small case subset for at most two epochs:

```powershell
python scripts/run_baselines.py gru --dataset-dir data/modeling/full --smoke --seed 42
```

Run the future full non-attention GRU experiment:

```powershell
python scripts/run_baselines.py gru --dataset-dir data/modeling/full --seed 42 --max-epochs 50 --patience 8 --batch-size 256 --device auto
```

GRU training uses case-balanced sampling by default. Pass
`--uniform-window-sampling` for an ordinary window-uniform comparison. Persistence
outputs are written under `outputs/baselines/persistence`; smoke and full GRU runs use
separate `outputs/baselines/gru/smoke_seed_42` and `outputs/baselines/gru/seed_42`
directories.

After a completed full seed-42 run, create a row-matched persistence comparison with:

```powershell
python scripts/compare_baselines.py --outputs-dir outputs/baselines --dataset-dir data/modeling/full --seed 42 --training-runtime-seconds <measured-seconds>
```

Aggregate completed fixed-seed GRU runs after supplying their measured runtimes:

```powershell
python scripts/aggregate_multiseed_gru.py --outputs-dir outputs/baselines --dataset-dir data/modeling/full --seeds 7,21,42,84,123 --runtime-seconds "7=<seconds>,21=<seconds>,42=<seconds>,84=<seconds>,123=<seconds>"
```

The aggregation command validates all required artifacts and row alignment before it
writes the seed summary, persistence comparison, and patient-by-seed table under
`outputs/baselines/gru`.

Run the explicit factorized feature/temporal-attention GRU smoke pipeline on CPU:

```powershell
python scripts/run_attention.py --dataset-dir data/modeling/full --smoke --seed 42
```

Smoke mode trains on at most four training cases and evaluates at most three validation
cases for no more than two epochs. It writes aligned validation attention arrays under
`outputs/attention/factorized_gru/smoke_seed_42`; these smoke attention values are for
pipeline validation only and are not scientific feature rankings.

Run one complete seed-42 factorized-attention experiment with the fixed baseline
training settings:

```powershell
python scripts/run_attention.py --dataset-dir data/modeling/full --seed 42 --max-epochs 50 --patience 8 --batch-size 256 --device auto
```

After the run, create aligned baseline comparisons and diagnostic attention summaries:

```powershell
python scripts/audit_attention_run.py --run-dir outputs/attention/factorized_gru/seed_42 --dataset-dir data/modeling/full --baselines-dir outputs/baselines --command-wall-seconds <measured-seconds>
```

The audit uses equal case weighting and labels all figures as single-seed diagnostics.
It does not create selected-feature or top-k artifacts.

## BIS Error Redundancy Diagnostic

`bis_error` is constructed in original units as `bis - 50`, so it must not count as
an independently selected predictive feature. The controlled seed-42 diagnostic met
the operational validation criterion for removing it. Future prediction-attention
experiments should therefore load the existing NPZ files and exclude it at runtime:

```powershell
python scripts/run_baselines.py gru --dataset-dir data/modeling/full --output-dir outputs/ablations/no_bis_error/gru/seed_42 --exclude-dynamic-features bis_error --seed 42 --max-epochs 50 --patience 8 --batch-size 256 --device auto
python scripts/run_attention.py --dataset-dir data/modeling/full --output-dir outputs/ablations/no_bis_error/attention/seed_42 --exclude-dynamic-features bis_error --seed 42 --max-epochs 50 --patience 8 --batch-size 256 --device auto
```

The original arrays and their preprocessing statistics remain unchanged. `bis_error`
may later be reconsidered as an explicit control-derived RL policy input, but not as
independent evidence in prediction feature-attention rankings. The complete diagnostic
is written under `outputs/ablations/no_bis_error`.
