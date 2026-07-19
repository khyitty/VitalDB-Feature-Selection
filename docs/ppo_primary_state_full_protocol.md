# Primary-State PPO Full Protocol

## Scope

This is the validation-only full comparison of four simulator-compatible observation
states. It is separate from the legacy encoder experiments and the completed
102,400-step pilot. No pilot model or optimizer state may initialize a full run.

Profiles are `original_reconstructed`, `all_supported`, `prediction_minimal`, and
`selected_control_core`. Seeds are `7, 21, 42, 84, 123`, producing 20 fresh runs.
Each run trains for 1,024,000 decision steps and evaluates all 15 validation patients
every 51,200 steps.

Test IDs and demographics may define the sealed cohort manifest. Test trajectories,
outcomes, policy rollouts, metrics, and checkpoint selection remain forbidden until
the validation-selected state is frozen.

## Frozen Scientific Settings

The committed source is `configs/ppo_primary_state_full.json`. The generated frozen
protocol binds it to one repository commit, cohort fingerprint, SB3 version, and
execution backend.

| Item | Value |
| --- | --- |
| SB3 | 2.9.0 |
| Policy | `MlpPolicy`, `FlattenExtractor` |
| Actor / critic | separate 64,64 hidden layers |
| Activation / optimizer | Tanh / Adam |
| Learning rate | 0.0003 |
| Rollout / batch / epochs | 2,048 / 64 / 10 |
| Gamma / GAE | 0.99 / 0.95 |
| PPO clip / entropy / value | 0.2 / 0.0 / 0.5 |
| Maximum gradient norm | 0.5 |
| Observation normalization | fixed unit-aware physical scaling; no fitted statistics |
| Reward normalization | none |
| Action | raw Gaussian, one `[-1,1]` clip, affine 0-12 mg/min conversion |
| Action interval / internal step | 10 seconds / 1 second |
| Episode | 1,800 seconds, 180 decisions |
| Cohort | 68 train / 15 validation / 15 sealed test |
| Checkpoint selection | BIS MAE, time in BIS 40-60, action-change sum |

The 0-12 mg/min bounds are a synthetic research profile, not a clinical dosing
recommendation.

## Initialization and Resume

Every identity starts at timestep zero from SB3's deterministic seed-controlled
random initialization. Full config records `initialization_source=fresh_random` and
`pilot_checkpoint_used=false`. A directory containing a checkpoint without a
compatible full config is rejected.

After every complete 2,048-step PPO rollout, `resume_model.zip` is atomically
replaced after the optimizer update. Resume requires the same protocol hash,
implementation commit, cohort, backend, SB3 version, state profile, seed, and rollout
alignment. The partial simulator episode is discarded and the train-patient draw
sequence advances. Resume is update-safe but is not claimed bitwise identical to an
uninterrupted trajectory.

## Device Benchmark

The engineering benchmark is separate under `outputs/ppo_device_benchmark`. It uses
`all_supported` and `selected_control_core`, seed 999, 20,480 steps, and three
repeats per profile/device. Each repeat saves and reloads at 10,240 steps. Diagnostic
control metrics are not used to select hardware.

CUDA is recommended only if its median training wall time is at least 25% lower than
CPU and all six CUDA repeats have zero failures, valid resume, and matching metric
schema. Otherwise CPU is recommended. This is an engineering threshold for Colab
and Drive overhead, not a scientific threshold.

## Commands

Local CPU benchmark:

```powershell
python scripts/benchmark_ppo_primary_state_devices.py --device cpu --demographics-csv data/raw/clinical.csv
```

Merge local CPU and Colab CUDA results:

```powershell
python scripts/benchmark_ppo_primary_state_devices.py --analyze outputs/ppo_device_benchmark/benchmark_results_cpu.csv outputs/ppo_device_benchmark/benchmark_results_cuda.csv
```

Freeze without training after backend selection:

```powershell
python scripts/run_ppo_state_full.py --device cpu --backend-decision outputs/ppo_device_benchmark/analysis/backend_decision.json --demographics-csv data/raw/clinical.csv --initialize-only
```

Run or resume all 20 identities:

```powershell
python scripts/run_ppo_state_full.py --device cpu --demographics-csv data/raw/clinical.csv --confirmation RUN_20_PRIMARY_STATE_FULL_RUNS
```

Rebuild validation-only analysis:

```powershell
python scripts/run_ppo_state_full.py --device cpu --demographics-csv data/raw/clinical.csv --analysis-only
```

The Colab entry point is `notebooks/colab_ppo_primary_state_full_training.ipynb`.
It defaults to CUDA benchmark only and keeps full training locked.
Its focused pytest preflight injects `VITALDB_MODELING_DATASET_DIR`,
`VITALDB_DEMOGRAPHICS_CSV`, and `VITALDB_PROJECT_DATA_ROOT` so a clean Colab clone
uses the same Drive-backed inputs as the benchmark and full runner. Data-backed PPO
tests retain repository-local defaults when those variables are unset, fail explicitly
when resolved inputs are missing, and can enforce the notebook's precomputed cohort
fingerprint through `VITALDB_EXPECTED_COHORT_FINGERPRINT`.
