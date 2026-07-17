# VitalDB Future-BIS Modeling Dataset

## Scientific objective and scope

The proposed contribution is data-driven RL state construction and selection,
followed by downstream propofol-control validation. PPO is a fixed validation
backbone, not a newly proposed algorithm. A valid primary comparison changes only
the ordered observation features while keeping the reconstructed PK-PD dynamics,
patients, synthetic exogenous remifentanil schedules, action, reward, common MLP
policy architecture, training budget, seeds, and evaluation protocol identical.

The simulator is reconstructed from published equations and available settings;
the unpublished Yun implementation is unavailable. The current main environment
uses repository-defined nonclinical action bounds and reward unless a separately
versioned protocol states otherwise. It is not an exact reproduction, is not
clinically validated, and must not be used for dosing or patient care.

Prediction attention rankings are materially unstable across seeds. Attention is
therefore preserved as an auxiliary association and reproducibility analysis, not
used as the primary selector or described as causal importance. The primary feature
selection method is train-only, patient-level Elastic Net stability selection. For
end-to-end consistency, the final prediction feature universe is restricted to
variables that can also be generated causally by the reconstructed PK-PD control
simulator.

The version-2 `simulator_compatible` universe contains 13 dynamic candidates: BIS,
`bis_delta_10s`, fixed-target BIS error, causal propofol/remifentanil rate and dose
history, and four Schnider/Minto Cp/Ce reconstructions. Age, sex, height, and weight
are static context. HR/PLETH_HR, blood pressure, SpO2, ETCO2, respiratory signals,
HRV, PLETH features, and BIS SQI remain excluded. Orchestra CP/CE are device-reported
TCI estimates, not direct concentration measurements; they are comparison-only and
never share the canonical names. CT target-concentration tracks are not loaded.
Prior physiological-inclusive and 9-feature version-1 datasets, rankings, ablations,
`strict_consensus`, and frozen candidates remain legacy exploratory artifacts.

## Canonical RL state profiles

- `original_reconstructed`: Yun-informed seven-concept dynamic history with
  demographics, reconstructed using causal raw BIS rather than an unspecified
  online LOWESS implementation. It does not expose Cp/Ce.
- `all_supported`: all 13 simulator-compatible dynamic candidates, including
  Schnider/Minto Cp/Ce, plus demographics.
- `selected`: features loaded only from a validated, versioned JSON manifest; Cp/Ce
  may be retained or removed by the later validation-only selection procedure.
- `legacy_control_aware`: former `selected_control_aware` debugging subset. It is
  not the proposed selected state.

`original_yun`, `yun_reconstructed`, and `selected_control_aware` remain temporary
deprecated aliases. The template at
`configs/rl_state_profiles/selected_template.json` is intentionally pending and
cannot start a run. The final selection rule, manifest, seeds, and full PPO budget
remain supervisor decisions.

Run the non-scientific common-policy integration smoke with exactly 1,000 PPO steps:

```bash
python scripts/run_ppo_experiment.py --smoke --state-profile original_reconstructed --smoke-timesteps 1000 --seed 42 --device cpu
```

This command runs Gymnasium validation, a random-action rollout, PPO training,
checkpoint reload, deterministic evaluation, action-clipping diagnostics, and
failure-safe `run_status.json`. Full scientific training is not started.

This repository separates raw data preparation from modeling-data construction:

- `main.py` downloads VitalDB tracks and performs initial signal-quality filtering,
  limited within-case forward filling, propofol-period cropping, and clinical-data merging.
- `scripts/build_prediction_dataset.py` reads the cleaned CSV, reuses the frozen
  patient-level split, resamples each case independently, fits preprocessing on
  training cases only, and constructs future-BIS prediction windows.

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

Build all eligible cases into the new main path without overwriting legacy data:

```powershell
python scripts/build_prediction_dataset.py --full
```

The outputs are written to `data/modeling/simulator_compatible_v2/{pilot,full}`. Test
arrays are serialized for the later locked evaluation, but test target summaries are
sealed and are not emitted by the builder or used during feature selection.

Cp/Ce are reconstructed forward from case-start raw Orchestra rate histories and
patient demographics using the same exact Schnider/Minto implementation as the RL
simulator. `pkpd_reconstruction_audit.json` records initial-state and missing-rate
limitations. Device-reported CP/CE agreement is summarized for train/validation only.
VitalDB documents those tracks as TCI-pump plasma/effect-site estimates but does not
establish the pump's exact model version and covariate configuration as equivalent to
this repository, so recorded values are never substituted for reconstructed values.

The old `data/modeling/full` dataset and all commands/results derived from its
physiological-inclusive 18/17-feature universe are legacy exploratory work. They can
only be rerun through explicit legacy flags and must not determine the final selected
state.

After the full simulator-compatible dataset is built, the final validation-only
five-seed prediction rerun is:

```powershell
$seeds = 7,21,42,84,123
foreach ($seed in $seeds) {
  python scripts/run_simulator_compatible_gru.py --dataset-dir data/modeling/simulator_compatible_v2/full --output-dir "outputs/simulator_compatible_prediction_v2/gru/seed_$seed" --seed $seed --max-epochs 50 --patience 8 --batch-size 256 --device cuda --validation-only
  python scripts/run_simulator_compatible_attention.py --dataset-dir data/modeling/simulator_compatible_v2/full --output-dir "outputs/simulator_compatible_prediction_v2/attention/seed_$seed" --seed $seed --max-epochs 50 --patience 8 --batch-size 256 --device cuda --validation-only
}
```

Colab/Linux equivalent:

```bash
for seed in 7 21 42 84 123; do
  python scripts/run_simulator_compatible_gru.py --dataset-dir data/modeling/simulator_compatible_v2/full --output-dir "outputs/simulator_compatible_prediction_v2/gru/seed_${seed}" --seed "${seed}" --max-epochs 50 --patience 8 --batch-size 256 --device cuda --validation-only
  python scripts/run_simulator_compatible_attention.py --dataset-dir data/modeling/simulator_compatible_v2/full --output-dir "outputs/simulator_compatible_prediction_v2/attention/seed_${seed}" --seed "${seed}" --max-epochs 50 --patience 8 --batch-size 256 --device cuda --validation-only
done
```

This command does not choose the selected subset and does not load the test split.

## Primary feature-selection workflow

The main analysis proceeds in the following order:

1. Evaluate latest-BIS persistence on validation only.
2. Run patient-level Elastic Net stability selection using training cases only.
3. Preserve explicit feature attention as a secondary analysis.
4. Compare frequency-threshold candidate subsets by validation group ablation.
5. Freeze the selected state before downstream PPO validation.

`bis_target_error` is excluded from Elastic Net because it is deterministically
duplicated by standardized BIS under the fixed BIS target of 50. The remaining 12
dynamic features each form one six-lag group. Age, sex, height, and weight are
included as unpenalized covariates in every fit but do not receive selection
frequencies. Hyperparameters are selected with patient-level `GroupKFold`; patient
bootstrap and inverse-window-count weighting prevent long cases from dominating.
Neither validation nor test is used to fit the selector.

Run the validation-only persistence baseline in Windows PowerShell:

```powershell
python -u scripts/run_persistence_baseline.py `
  --dataset-dir data/modeling/simulator_compatible_v2/full `
  --output-dir outputs/simulator_compatible_prediction_v2/persistence `
  --validation-only
```

Run the three-bootstrap integration smoke without reading validation or test:

```powershell
python -u scripts/run_elastic_net_stability.py `
  --dataset-dir data/modeling/simulator_compatible_v2/full `
  --output-dir outputs/simulator_compatible_prediction_v2/elastic_net_stability_smoke `
  --seed 42 `
  --smoke
```

After reviewing the smoke artifacts, run the planned 100 patient bootstraps with:

```powershell
python -u scripts/run_elastic_net_stability.py `
  --dataset-dir data/modeling/simulator_compatible_v2/full `
  --output-dir outputs/simulator_compatible_prediction_v2/elastic_net_stability `
  --seed 42 `
  --bootstrap-count 100 `
  --cv-folds 5 `
  --l1-ratios 0.1,0.5,0.9,1.0 `
  --alpha-min 1e-4 `
  --alpha-max 1 `
  --alpha-count 9 `
  --coefficient-tolerance 1e-6
```

The `0.80`, `0.60`, and `0.40` frequency JSON files are candidate subsets for
validation ablation. They do not automatically become the final PPO state manifest.

Run the synthetic test suite:

```powershell
python -m pytest -q
```

## PK-PD patient simulator

Module 4 provides a research-only Schnider propofol, Minto remifentanil, and Yun
combined-BIS simulator. It uses an exact zero-order-hold matrix exponential at a
default 1-second internal interval while preserving the planned 10-second control
hold. Remifentanil is an exogenous schedule. The module does not generate vital signs,
implement Gymnasium/PPO, train RL agents, or apply actions to actual VitalDB patients.

```bash
python scripts/run_pkpd_simulator_validation.py \
  --patient middle_male \
  --duration-seconds 1800 \
  --internal-dt-seconds 1 \
  --output-dir outputs/pkpd_simulator_validation
```

Equation provenance and unit decisions are documented in
`docs/pkpd_equation_traceability.md` and `docs/pkpd_assumptions_and_units.md`.
The standalone CPU notebook is `notebooks/colab_pkpd_simulator_validation.ipynb`.

This simulator is a research reconstruction of published PK-PD equations. It is not a
medical device and must not be used for clinical dosing.

Pilot arrays are saved to
`data/modeling/simulator_compatible_v2/pilot/{train,val,test}.npz`; matching window
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

Aggregate the paired reduced GRU/attention runs for seeds 7, 21, 42, 84, and 123:

```powershell
python scripts/aggregate_reduced_multiseed.py --root-dir outputs/ablations/no_bis_error --dataset-dir data/modeling/full --output-dir outputs/ablations/no_bis_error/multiseed --seeds 7,21,42,84,123
```

The five-seed diagnostic found effectively preserved prediction performance but
unstable individual and grouped attention rankings. The multiseed outputs must
therefore be treated as reproducibility diagnostics, not causal importance or a basis
for top-k selection.

Run the validation-only unavailable-ablation, within-patient permutation, and
attention-faithfulness audit from the existing five-seed checkpoints:

```powershell
python scripts/audit_attention_faithfulness.py --root-dir outputs/ablations/no_bis_error --dataset-dir data/modeling/full --output-dir outputs/ablations/no_bis_error/faithfulness --seeds 7,21,42,84,123 --batch-size 1024 --permutation-repetitions 10
```

This audit does not read test labels or test importance for development decisions.
Unavailable ablation sets normalized values and observation masks to zero and may
induce distribution shift. Within-patient permutation preserves complete six-step
trajectories and masks but remains an association diagnostic, not causal evidence.

Audit local hardware and benchmark short deterministic CPU training workloads with:

```powershell
python scripts/benchmark_training_runtime.py --dataset-dir data/modeling/full --output-dir outputs/runtime_benchmark --thread-counts 1,2,4,8 --worker-counts 0,2 --measured-batches 20 --warmup-batches 3 --batch-size 256 --seed 42
```

Future training commands may set `--torch-num-threads`,
`--torch-interop-threads`, and `--num-workers`; omitting them preserves historical
defaults. Every future paired comparison must use the same device and backend for all
models and seeds. Completed historical experiments should not be rerun only to adopt
a faster device or thread configuration.

## Google Colab GPU Validation

Open `notebooks/colab_gpu_setup.ipynb` in a GPU-enabled Google Colab runtime. The
notebook mounts Drive, clones or updates this repository, records the active commit,
retains Colab's CUDA-enabled PyTorch, validates modeling artifacts, runs tests, and
performs only validation smoke experiments and a short CUDA benchmark.

The configurable Drive placeholders are:

```python
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/VitalDB-Feature-Selection"
DATASET_DIR = f"{DRIVE_PROJECT_ROOT}/data/modeling/full"
OUTPUT_ROOT = f"{DRIVE_PROJECT_ROOT}/outputs"
```

Do not commit the Drive dataset, credentials, or tokens. GPU training uses only the
saved modeling artifacts and does not require the raw one-second VitalDB CSV. The
Colab dependency installer reads `requirements-colab.txt`, rejects PyTorch packages,
examines pip's dry-run plan, and installs only missing non-PyTorch dependencies.

The notebook executes these guarded commands after CUDA and data validation:

```bash
python -m pytest -q
python scripts/run_baselines.py gru --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_ROOT/colab_smoke/gru/seed_42" --exclude-dynamic-features bis_error --smoke --seed 42 --device cuda
python scripts/run_attention.py --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_ROOT/colab_smoke/attention/seed_42" --exclude-dynamic-features bis_error --smoke --seed 42 --device cuda
python scripts/benchmark_colab_gpu.py --dataset-dir "$DATASET_DIR" --output-dir "$OUTPUT_ROOT/runtime_benchmark" --batch-sizes 256,512,1024,2048 --measured-batches 20 --warmup-batches 3 --seed 42
```

`run_status.json` is written as `running` before training and `complete` only after
checkpoint reload and output serialization. The notebook skips complete smoke runs
and restarts only an incomplete run directory. Smoke mode does not evaluate the test
split. CPU and CUDA outputs need not be bitwise identical, but every paired scientific
comparison must use one backend for every model and seed.

The existing test split has already been inspected during development and is therefore
a development test set, not a pristine final holdout. Future group-retraining candidate
selection must remain validation-only. A final performance claim should use previously
unseen cases or another pre-specified evaluation design.
