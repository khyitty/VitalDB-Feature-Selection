"""Patient lookup and split-integrity contracts for later RL cohorts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.pkpd.demographics import PatientDemographics


@dataclass(frozen=True)
class CohortManifest:
    train_patient_ids: tuple[str, ...]
    validation_patient_ids: tuple[str, ...]
    test_patient_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        groups = {
            "train": set(self.train_patient_ids),
            "validation": set(self.validation_patient_ids),
            "test": set(self.test_patient_ids),
        }
        if any(not values for values in groups.values()):
            raise ValueError("Each patient split must be non-empty.")
        overlaps = {
            "train_validation": groups["train"] & groups["validation"],
            "train_test": groups["train"] & groups["test"],
            "validation_test": groups["validation"] & groups["test"],
        }
        bad = {name: sorted(values) for name, values in overlaps.items() if values}
        if bad:
            raise ValueError(f"Patient IDs overlap across cohort splits: {bad}")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "CohortManifest":
        return cls(
            train_patient_ids=tuple(map(str, values["train"])),
            validation_patient_ids=tuple(map(str, values["validation"])),
            test_patient_ids=tuple(map(str, values["test"])),
        )

    @classmethod
    def from_json(cls, path: Path) -> "CohortManifest":
        return cls.from_mapping(json.loads(path.read_text(encoding="utf-8")))

    def split_for(self, patient_id: str) -> str:
        matches = [
            name
            for name, values in (
                ("train", self.train_patient_ids),
                ("validation", self.validation_patient_ids),
                ("test", self.test_patient_ids),
            )
            if patient_id in values
        ]
        if len(matches) != 1:
            raise ValueError(f"Patient {patient_id!r} belongs to {len(matches)} splits.")
        return matches[0]


@dataclass(frozen=True)
class PatientCohort:
    patients: Mapping[str, PatientDemographics]
    manifest: CohortManifest

    def __post_init__(self) -> None:
        expected = set(self.manifest.train_patient_ids) | set(
            self.manifest.validation_patient_ids
        ) | set(self.manifest.test_patient_ids)
        missing = sorted(expected - set(self.patients))
        if missing:
            raise ValueError(f"Cohort demographics are missing patient IDs: {missing}")

    def patient(self, patient_id: str) -> PatientDemographics:
        try:
            return self.patients[patient_id]
        except KeyError as exc:
            raise ValueError(f"Unknown cohort patient ID: {patient_id!r}.") from exc
