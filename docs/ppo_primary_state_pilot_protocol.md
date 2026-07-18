# Primary-State PPO Pilot Protocol

## Scope

This protocol compares only the observation state. The PK-PD simulator, virtual
patient cohort, exogenous remifentanil scenarios, action and reward definitions,
common MLP architecture, optimizer, training budget, validation trajectories, and
checkpoint rule are fixed across all profiles. The 102,400-step pilot is exploratory;
it cannot select a final winner or establish clinical performance.

The committed source is `configs/ppo_primary_state_pilot.json`. At initialization it
is bound to one exact repository commit, one cohort fingerprint, one execution
backend, and the ordered validation scenario IDs. The resulting
`frozen_primary_state_pilot_protocol.json` is the run-time source of truth.

## Protocol Audit

| Item | Current code-backed value | Source |
| --- | --- | --- |
| Stable-Baselines3 | 2.9.0 | `requirements-rl.txt` |
| Policy | SB3 `MlpPolicy`, separate actor/critic 64,64 MLPs | `training.py`, `policy_registry.py` |
| Feature extractor | SB3 `FlattenExtractor` | `policy_registry.py` |
| Activation | Tanh | SB3 `MlpPolicy` default, frozen explicitly in pilot config |
| Optimizer | Adam | SB3 policy default, frozen explicitly in pilot config |
| Learning rate | 0.0003 | `PPOConfig` |
| Rollout length | 2,048 steps | `PPOConfig.n_steps` |
| Batch size / epochs | 64 / 10 | `PPOConfig` |
| Gamma / GAE lambda | 0.99 / 0.95 | `PPOConfig` |
| PPO clip | 0.2 | `PPOConfig` |
| Entropy / value coefficients | 0.0 / 0.5 | `PPOConfig` |
| Maximum gradient norm | 0.5 | `PPOConfig` |
| Observation normalization | Fixed feature-specific physical scales; no fitted statistics | `ScaledFlattenObservationAdapter` |
| Reward normalization | None | PPO construction and pilot config |
| Agent action | One current propofol infusion rate | `NormalizedPropofolActionWrapper` |
| Raw action transform | Unbounded Gaussian action, one SB3 clip to [-1,1], one affine physical conversion | callback and action wrapper |
| Physical action bounds | 0 to 12 mg/min, nonclinical synthetic profile | `SYNTHETIC_NONCLINICAL_ACTION_BOUNDS` |
| Action interval | 10 seconds | `EnvironmentConfig` |
| Episode duration | 1,800 seconds, 180 actions | `PPOConfig` |
| Train subject sampling | Uniform random train patient per episode | `CohortScenarioWrapper` |
| Train cohort | 68 frozen VitalDB case IDs | modeling split manifest |
| Validation cohort | All 15 frozen IDs in manifest order | modeling split manifest |
| Validation scenario seeds | 100000 plus manifest index | `scenarios_for_split` |
| Test guard | IDs/demographics may define the manifest; trajectories, outcomes, rollouts, and selection are forbidden | cohort wrapper and protocol seal |
| Pilot evaluation interval | 51,200 steps, exactly 25 PPO rollouts | pilot source config |
| Evaluation episodes | All 15 validation patients; count equality is validated | pilot protocol builder |
| Evaluation mode | Deterministic policy and deterministic simulator | evaluator and `PPOConfig` |
| Checkpoint selection | Validation mean BIS MAE, then time in 40-60, then action-change sum | `checkpoint_score` |
| Pilot budget | 102,400 steps per run | pilot source config |
| Full budget | 1,024,000 steps per run | default `PPOConfig` |
| Pilot seeds | 7, 42, 84 | pilot source config |
| Planned full seeds | 7, 21, 42, 84, 123 | legacy/full `EXPERIMENT_SEEDS` |

## Audit Conflicts

1. The existing `ppo_control_comparison_v1` protocol is a legacy four-encoder by
   five-seed experiment. It includes GRU and attention extractor changes, so it is
   not a state-only comparison. It remains untouched.
2. `run_ppo_experiment.py` intentionally permits canonical profiles only in smoke
   mode. The pilot therefore has a separate non-smoke entry point rather than
   weakening that guard.
3. The legacy Colab full runner requires CUDA without a CPU/CUDA throughput
   comparison. The pilot freezes exactly one resolved backend for all 12 runs.
   A machine without CUDA must use CPU; it must not claim a CUDA comparison.
4. `evaluation_episode_count=15` was previously descriptive while the evaluator
   iterated every validation scenario. The pilot builder now fails unless the
   configured count exactly equals the validation cohort size.
5. The legacy runner saved only at evaluation chunks. A mid-chunk interruption could
   lose substantial work and shift a resumed validation boundary. The pilot saves an
optimizer-complete checkpoint every 2,048-step rollout and resumes to the next
original 51,200-step boundary.
6. Legacy training progress omitted several requested SB3 diagnostics. The pilot
   stores policy/value loss, entropy, approximate KL, clip fraction, explained
   variance, learning rate, return, FPS, elapsed time, action diagnostics, protocol,
   cohort, commit, observation dimension, and parameter count.

## State Inventory

- `original_reconstructed`: 53-dimensional flattened observation.
- `all_supported`: 89 dimensions.
- `prediction_minimal`: 23 dimensions.
- `selected_control_core`: 47 dimensions.

Input-layer parameter counts differ with observation dimension. Policy class,
extractor, hidden layers, activation, optimizer, and every PPO hyperparameter remain
identical.

## Checkpoints and Resume

`resume_model.zip` is atomically replaced after each complete PPO rollout. Evaluation
checkpoints are `checkpoint_51200.zip` and `checkpoint_102400.zip`. A matching complete
run is verified and skipped. Partial configuration, protocol hash, cohort fingerprint,
repository commit, SB3 version, device, timestep alignment, and persisted progress are
validated before resume. Failure metadata and traceback are written before the CLI
exits; later inventory items do not continue silently.

The PPO optimizer and timestep are restored, but an in-progress PK-PD episode is not
serialized by SB3. On resume, that partial episode is explicitly discarded. The
train-patient RNG advances past all previously started episode draws and starts the
next deterministic patient scenario, avoiding a silent restart of the patient
sequence. Resume is rollout/update safe but is not claimed to be bitwise trajectory
identical to an uninterrupted run.

## Commands

Freeze and inspect without training:

```powershell
python scripts/run_ppo_state_pilot.py --demographics-csv data/raw/clinical.csv --device cpu --initialize-only
```

Run or resume the requested seed-7 end-to-end subset:

```powershell
python scripts/run_ppo_state_pilot.py --demographics-csv data/raw/clinical.csv --device cpu --seeds 7 --confirmation RUN_12_PRIMARY_STATE_PILOT_RUNS
```

Run or resume every remaining identity under the same frozen hash:

```powershell
python scripts/run_ppo_state_pilot.py --demographics-csv data/raw/clinical.csv --device cpu --confirmation RUN_12_PRIMARY_STATE_PILOT_RUNS
```

Rebuild analysis only from compatible completed artifacts:

```powershell
python scripts/run_ppo_state_pilot.py --demographics-csv data/raw/clinical.csv --device cpu --analysis-only
```

Outputs are under `outputs/ppo_primary_state_pilot` and are ignored by Git. The
analysis includes completion, checkpoint, paired patient, action, learning-curve,
failure, reproducibility, report, and exploratory figure artifacts.
