"""VitalDB split-preserving virtual-patient cohort and paired scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import pandas as pd

from src.pkpd.demographics import PatientDemographics
from src.pkpd.schedules import PiecewiseConstantSchedule, RateSegment
from src.rl_env.cohort import CohortManifest, PatientCohort


@dataclass(frozen=True)
class CohortBundle:
    cohort: PatientCohort
    demographics_source: str
    split_source: str
    fingerprint: str
    patient_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EvaluationScenario:
    scenario_id: str
    split: str
    patient_id: str
    seed: int


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_vitaldb_virtual_cohort(dataset_dir: Path, repo_dir: Path) -> CohortBundle:
    """Reuse case splits and raw demographics without replaying clinical trajectories."""

    metadata = json.loads((dataset_dir / "dataset_metadata.json").read_text(encoding="utf-8"))
    input_path = Path(str(metadata["input_file"]).replace("\\", "/"))
    if not input_path.is_absolute():
        input_path = repo_dir / input_path
    required = ["caseid", "age", "sex_male", "height", "weight"]
    demographics = pd.read_csv(input_path, usecols=required).drop_duplicates("caseid")
    if demographics[required].isna().any().any():
        missing = demographics.loc[demographics[required].isna().any(axis=1), "caseid"].tolist()
        raise ValueError(f"VitalDB demographics are incomplete; no imputation allowed: {missing}")
    if not demographics["sex_male"].isin([0, 1]).all():
        raise ValueError("sex_male must be exactly 0 or 1 for every virtual patient.")

    split_values: dict[str, tuple[str, ...]] = {}
    filenames = {"train": "train_cases.csv", "validation": "val_cases.csv", "test": "test_cases.csv"}
    for split, filename in filenames.items():
        frame = pd.read_csv(dataset_dir / "splits" / filename)
        split_values[split] = tuple(frame["caseid"].astype(str))
    manifest = CohortManifest(
        train_patient_ids=split_values["train"],
        validation_patient_ids=split_values["validation"],
        test_patient_ids=split_values["test"],
    )
    indexed = demographics.set_index(demographics["caseid"].astype(str))
    patients: dict[str, PatientDemographics] = {}
    records: list[dict[str, Any]] = []
    for patient_id in (
        *manifest.train_patient_ids,
        *manifest.validation_patient_ids,
        *manifest.test_patient_ids,
    ):
        if patient_id not in indexed.index:
            raise ValueError(f"Demographics missing for split patient {patient_id}.")
        row = indexed.loc[patient_id]
        patient = PatientDemographics(
            age_years=float(row["age"]),
            sex="male" if int(row["sex_male"]) == 1 else "female",
            height_cm=float(row["height"]),
            weight_kg=float(row["weight"]),
        )
        patients[patient_id] = patient
        records.append(
            {
                "patient_id": patient_id,
                "split": manifest.split_for(patient_id),
                **patient.as_dict(),
            }
        )
    payload = {"splits": {"train": list(manifest.train_patient_ids), "validation": list(manifest.validation_patient_ids), "test": list(manifest.test_patient_ids)}, "patients": records}
    return CohortBundle(
        cohort=PatientCohort(patients=patients, manifest=manifest),
        demographics_source=input_path.relative_to(repo_dir).as_posix(),
        split_source=(dataset_dir / "splits").relative_to(repo_dir).as_posix(),
        fingerprint=_canonical_hash(payload),
        patient_records=tuple(records),
    )


def scenarios_for_split(
    bundle: CohortBundle,
    split: str,
    *,
    base_seed: int,
) -> tuple[EvaluationScenario, ...]:
    if split == "train":
        patient_ids = bundle.cohort.manifest.train_patient_ids
    elif split == "validation":
        patient_ids = bundle.cohort.manifest.validation_patient_ids
    elif split == "test":
        patient_ids = bundle.cohort.manifest.test_patient_ids
    else:
        raise ValueError(f"Unknown cohort split: {split!r}.")
    return tuple(
        EvaluationScenario(
            scenario_id=f"{split}-case-{patient_id}-seed-{base_seed + index}",
            split=split,
            patient_id=patient_id,
            seed=base_seed + index,
        )
        for index, patient_id in enumerate(patient_ids)
    )


def remifentanil_schedule_for_scenario(
    scenario: EvaluationScenario, episode_duration_seconds: float
) -> PiecewiseConstantSchedule:
    digest = hashlib.sha256(scenario.scenario_id.encode()).digest()
    rates = [2.0 + digest[index] / 255.0 * 6.0 for index in range(3)]
    third = episode_duration_seconds / 3.0
    return PiecewiseConstantSchedule(
        [
            RateSegment(0.0, third, rates[0]),
            RateSegment(third, 2.0 * third, rates[1]),
            RateSegment(2.0 * third, episode_duration_seconds + 10.0, rates[2]),
        ]
    )


class CohortScenarioWrapper(gym.Wrapper):
    """Select only an authorized split and expose deterministic scenario IDs."""

    def __init__(
        self,
        env: gym.Env,
        *,
        bundle: CohortBundle,
        split: str,
        base_seed: int,
        episode_duration_seconds: float,
        cycle: bool = False,
    ) -> None:
        if split == "test":
            raise ValueError("Test cohort access is sealed during training/validation workflow.")
        super().__init__(env)
        self.bundle = bundle
        self.split = split
        self.scenarios = scenarios_for_split(bundle, split, base_seed=base_seed)
        self.episode_duration_seconds = episode_duration_seconds
        self.cycle = cycle
        self._rng = np.random.default_rng(base_seed)
        self._index = 0
        self.current_scenario: EvaluationScenario | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "scenario" in options:
            scenario = options["scenario"]
            if not isinstance(scenario, EvaluationScenario) or scenario.split != self.split:
                raise ValueError("Explicit scenario does not belong to the authorized split.")
        elif self.cycle:
            scenario = self.scenarios[self._index % len(self.scenarios)]
            self._index += 1
        else:
            scenario = self.scenarios[int(self._rng.integers(len(self.scenarios)))]
        self.current_scenario = scenario
        merged = dict(options or {})
        merged.pop("scenario", None)
        merged.update(
            {
                "patient_id": scenario.patient_id,
                "remifentanil_schedule": remifentanil_schedule_for_scenario(
                    scenario, self.episode_duration_seconds
                ),
            }
        )
        observation, info = self.env.reset(seed=scenario.seed, options=merged)
        info = dict(info)
        info.update({"scenario_id": scenario.scenario_id, "cohort_split": self.split})
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        assert self.current_scenario is not None
        info.update(
            {"scenario_id": self.current_scenario.scenario_id, "cohort_split": self.split}
        )
        return observation, reward, terminated, truncated, info
