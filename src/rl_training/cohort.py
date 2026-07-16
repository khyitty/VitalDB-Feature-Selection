"""VitalDB split-preserving virtual-patient cohort and paired scenarios."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any, Literal, Mapping

import gymnasium as gym
import numpy as np
import pandas as pd

from src.pkpd.demographics import PatientDemographics
from src.pkpd.schedules import PiecewiseConstantSchedule, RateSegment
from src.rl_env.cohort import CohortManifest, PatientCohort

from .official_demographics import ensure_official_demographics_cache


@dataclass(frozen=True)
class CohortBundle:
    cohort: PatientCohort
    demographics_source: str
    demographics_source_kind: str
    demographics_source_columns: tuple[str, ...]
    demographics_source_fingerprint: str
    split_source: str
    fingerprint: str
    patient_records: tuple[dict[str, Any], ...]
    missing_demographics: Mapping[str, Mapping[str, int]]
    imputation_statistics: Mapping[str, float]
    access_manifest: Mapping[str, Any]


@dataclass(frozen=True)
class EvaluationScenario:
    scenario_id: str
    split: str
    patient_id: str
    seed: int


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


REQUIRED_DEMOGRAPHIC_COLUMNS = ("caseid", "age", "sex", "height", "weight")
_COLUMN_ALIASES = {
    "caseid": ("caseid", "case_id", "patient_id"),
    "age": ("age", "age_years"),
    "sex": ("sex_male", "sex"),
    "height": ("height", "height_cm"),
    "weight": ("weight", "weight_kg"),
}


class CohortPreflightError(ValueError):
    """Raised when split-safe virtual-patient initialization cannot proceed."""


def _case_id(value: Any) -> str:
    if pd.isna(value):
        raise CohortPreflightError("Demographics contain a missing caseid.")
    text = str(value).strip()
    try:
        numeric = float(text)
    except ValueError:
        return text
    return str(int(numeric)) if numeric.is_integer() else text


def _column_mapping(columns: list[str]) -> dict[str, str]:
    lowered = {str(column).strip().lower(): str(column) for column in columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lowered:
                mapping[canonical] = lowered[alias]
                break
    return mapping


def _read_split_manifest(dataset_dir: Path) -> CohortManifest:
    split_values: dict[str, tuple[str, ...]] = {}
    filenames = {"train": "train_cases.csv", "validation": "val_cases.csv", "test": "test_cases.csv"}
    for split, filename in filenames.items():
        path = dataset_dir / "splits" / filename
        if not path.is_file():
            raise CohortPreflightError(f"Missing patient split file: {path}")
        frame = pd.read_csv(path)
        if "caseid" not in frame.columns:
            raise CohortPreflightError(
                f"Split file {path} lacks required column 'caseid'; found {list(frame.columns)}."
            )
        values = tuple(_case_id(value) for value in frame["caseid"])
        if len(values) != len(set(values)):
            raise CohortPreflightError(f"Duplicate caseid values in split file: {path}")
        split_values[split] = values
    return CohortManifest(
        train_patient_ids=split_values["train"],
        validation_patient_ids=split_values["validation"],
        test_patient_ids=split_values["test"],
    )


def _read_demographic_columns(path: Path) -> list[str]:
    try:
        return [str(column) for column in pd.read_csv(path, nrows=0).columns]
    except Exception as exc:
        raise CohortPreflightError(f"Unable to read demographics header from {path}: {exc}") from exc


def _canonicalize_demographics(frame: pd.DataFrame, mapping: Mapping[str, str]) -> pd.DataFrame:
    canonical = frame.loc[:, list(mapping.values())].rename(
        columns={source: target for target, source in mapping.items()}
    )
    canonical["caseid"] = canonical["caseid"].map(_case_id)
    sex = canonical["sex"]
    numeric = pd.to_numeric(sex, errors="coerce")
    text = sex.astype("string").str.strip().str.lower()
    sex_missing = sex.isna() | text.eq("").fillna(False)
    canonical["sex"] = np.select(
        [
            numeric.eq(1).fillna(False).to_numpy(),
            numeric.eq(0).fillna(False).to_numpy(),
            text.isin(("male", "m")).fillna(False).to_numpy(),
            text.isin(("female", "f")).fillna(False).to_numpy(),
        ],
        [1.0, 0.0, 1.0, 0.0],
        default=np.nan,
    )
    invalid_sex = canonical["sex"].isna() & ~sex_missing
    if invalid_sex.any():
        values = sorted(set(map(str, sex.loc[invalid_sex].tolist())))
        raise CohortPreflightError(
            f"Invalid sex values {values}; accepted values are 0/1 or female/male."
        )
    for column in ("age", "height", "weight"):
        original = canonical[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = converted.isna() & original.notna() & original.astype(str).str.strip().ne("")
        if invalid.any():
            values = sorted(set(map(str, original.loc[invalid].tolist())))
            raise CohortPreflightError(f"Invalid numeric {column} values: {values}.")
        canonical[column] = converted
    return canonical


def _read_demographics_file(path: Path) -> tuple[pd.DataFrame, tuple[str, ...]]:
    columns = _read_demographic_columns(path)
    mapping = _column_mapping(columns)
    missing = sorted(set(REQUIRED_DEMOGRAPHIC_COLUMNS) - set(mapping))
    if missing:
        raise CohortPreflightError(
            f"Demographics source {path} lacks required columns {missing}; found {columns}. "
            f"Accepted aliases: {_COLUMN_ALIASES}."
        )
    chunks = list(
        _canonicalize_demographics(chunk, mapping)
        for chunk in pd.read_csv(path, usecols=list(mapping.values()), chunksize=100_000)
    )
    if not chunks:
        raise CohortPreflightError(f"Demographics source has no data rows: {path}")
    return pd.concat(chunks, ignore_index=True), tuple(columns)


def _embedded_demographics(
    dataset_dir: Path,
) -> tuple[pd.DataFrame, tuple[str, ...], str] | None:
    metadata_path = dataset_dir / "dataset_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CohortPreflightError(f"Invalid dataset metadata JSON: {metadata_path}") from exc
        embedded = metadata.get("case_demographics", metadata.get("demographics"))
        if isinstance(embedded, dict):
            embedded = embedded.get("records")
        if isinstance(embedded, list) and embedded:
            frame = pd.DataFrame(embedded)
            mapping = _column_mapping([str(column) for column in frame.columns])
            if set(mapping) == set(REQUIRED_DEMOGRAPHIC_COLUMNS):
                return (
                    _canonicalize_demographics(frame, mapping),
                    tuple(map(str, frame.columns)),
                    "dataset_metadata_json",
                )
    paths = [
        dataset_dir / "train_metadata.csv",
        dataset_dir / "val_metadata.csv",
        dataset_dir / "test_metadata.csv",
    ]
    if not all(path.is_file() for path in paths):
        return None
    mappings: list[dict[str, str]] = []
    all_columns: list[str] = []
    for path in paths:
        columns = _read_demographic_columns(path)
        all_columns.extend(f"{path.name}:{column}" for column in columns)
        mapping = _column_mapping(columns)
        if set(mapping) != set(REQUIRED_DEMOGRAPHIC_COLUMNS):
            return None
        mappings.append(mapping)
    frames = [
        _canonicalize_demographics(
            pd.read_csv(path, usecols=list(mapping.values())), mapping
        )
        for path, mapping in zip(paths, mappings)
    ]
    return pd.concat(frames, ignore_index=True), tuple(all_columns), "modeling_metadata"


def _metadata_source_names(dataset_dir: Path) -> list[str]:
    names: list[str] = []
    for filename, key_path in (
        ("dataset_metadata.json", ("input_file",)),
        ("full_dataset_audit.json", ("audit_scope", "input_file")),
    ):
        path = dataset_dir / filename
        if not path.is_file():
            continue
        try:
            value: Any = json.loads(path.read_text(encoding="utf-8"))
            for key in key_path:
                value = value[key]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        text = str(value)
        name = PureWindowsPath(text).name or Path(text).name
        if name and name not in names:
            names.append(name)
    metadata_path = dataset_dir / "dataset_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}
        for key in ("demographics_file", "demographics_csv", "clinical_input_file"):
            if key in metadata:
                name = PureWindowsPath(str(metadata[key])).name or Path(str(metadata[key])).name
                if name and name not in names:
                    names.append(name)
    return names


def _project_candidates(dataset_dir: Path, project_data_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for name in _metadata_source_names(dataset_dir):
        candidates.extend(
            (
                project_data_root / "processed" / name,
                project_data_root / "clinical" / name,
                project_data_root / name,
            )
        )
        if project_data_root.is_dir():
            candidates.extend(project_data_root.rglob(name))
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _portable_source_label(path: Path, project_data_root: Path | None) -> str:
    if project_data_root is not None:
        try:
            return f"data/{path.resolve().relative_to(project_data_root.resolve()).as_posix()}"
        except ValueError:
            pass
    return str(path.resolve())


def _resolve_demographics(
    dataset_dir: Path,
    *,
    demographics_csv: Path | None,
    project_data_root: Path | None,
) -> tuple[pd.DataFrame, str, str, tuple[str, ...], tuple[str, ...]]:
    embedded = _embedded_demographics(dataset_dir)
    if embedded is not None:
        frame, columns, kind = embedded
        return frame, "modeling metadata CSV files", kind, columns, ()

    candidates: list[Path] = []
    if demographics_csv is not None:
        candidates.append(demographics_csv)
    if project_data_root is not None:
        candidates.extend(_project_candidates(dataset_dir, project_data_root))
    attempted: list[str] = []
    discovered: dict[str, list[str]] = {}
    errors: list[str] = []
    for candidate in candidates:
        resolved = candidate.expanduser().resolve(strict=False)
        label = str(resolved)
        if label in attempted:
            continue
        attempted.append(label)
        if not resolved.is_file():
            continue
        try:
            frame, columns = _read_demographics_file(resolved)
        except CohortPreflightError as exc:
            try:
                discovered[label] = _read_demographic_columns(resolved)
            except CohortPreflightError:
                discovered[label] = []
            errors.append(str(exc))
            continue
        return (
            frame,
            _portable_source_label(resolved, project_data_root),
            "explicit_csv" if demographics_csv is not None and resolved == demographics_csv.expanduser().resolve(strict=False) else "project_data_discovery",
            columns,
            tuple(attempted),
        )
    metadata_columns = {
        path.name: _read_demographic_columns(path)
        for path in (
            dataset_dir / "train_metadata.csv",
            dataset_dir / "val_metadata.csv",
            dataset_dir / "test_metadata.csv",
        )
        if path.is_file()
    }
    raise CohortPreflightError(
        "Unable to resolve VitalDB demographics without a hidden repository-local dependency. "
        f"Required canonical columns: {list(REQUIRED_DEMOGRAPHIC_COLUMNS)}. "
        f"Modeling metadata columns: {metadata_columns}. Searched candidate paths: {attempted}. "
        f"Readable but incompatible candidates: {discovered}. Errors: {errors}. "
        "Provide --demographics-csv or --project-data-root containing the source CSV."
    )


def _deduplicate_demographics(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(frame)
    unique_rows = frame.drop_duplicates(list(REQUIRED_DEMOGRAPHIC_COLUMNS)).copy()
    inconsistent = unique_rows[unique_rows.duplicated("caseid", keep=False)]["caseid"].unique()
    if len(inconsistent):
        raise CohortPreflightError(
            "Conflicting demographic rows for caseid values: "
            f"{sorted(map(str, inconsistent))}."
        )
    return unique_rows.drop_duplicates("caseid"), before - len(unique_rows)


def _prepare_demographics(
    demographics: pd.DataFrame,
    manifest: CohortManifest,
    *,
    missing_policy: Literal["error", "train_impute"],
) -> tuple[pd.DataFrame, dict[str, dict[str, int]], dict[str, float], dict[str, list[str]]]:
    expected_by_split = {
        "train": manifest.train_patient_ids,
        "validation": manifest.validation_patient_ids,
        "test": manifest.test_patient_ids,
    }
    expected = set().union(*(set(values) for values in expected_by_split.values()))
    available = set(demographics["caseid"])
    absent = sorted(expected - available)
    if absent:
        raise CohortPreflightError(
            f"Demographics are missing split caseid values: {absent}. No patient is silently excluded."
        )
    selected = demographics[demographics["caseid"].isin(expected)].set_index("caseid").copy()
    missing_counts = {
        split: {
            column: int(selected.loc[list(patient_ids), column].isna().sum())
            for column in ("age", "sex", "height", "weight")
        }
        for split, patient_ids in expected_by_split.items()
    }
    if missing_policy not in ("error", "train_impute"):
        raise ValueError(f"Unknown missing demographics policy: {missing_policy!r}.")
    statistics: dict[str, float] = {}
    imputed_fields: dict[str, list[str]] = {}
    if any(count for values in missing_counts.values() for count in values.values()):
        if missing_policy == "error":
            raise CohortPreflightError(
                "Missing age/sex/height/weight values detected; default policy forbids imputation. "
                f"Counts by split: {missing_counts}."
            )
        train = selected.loc[list(manifest.train_patient_ids)]
        missing_columns = {
            column
            for values in missing_counts.values()
            for column, count in values.items()
            if count
        }
        for column in ("age", "height", "weight"):
            if column not in missing_columns:
                continue
            statistic = float(train[column].median(skipna=True))
            if not np.isfinite(statistic):
                raise CohortPreflightError(
                    f"Cannot derive train-only imputation statistic for {column}."
                )
            statistics[column] = statistic
        if "sex" in missing_columns:
            sex_counts = train["sex"].dropna().value_counts()
            if sex_counts.empty or len(sex_counts) > 1 and sex_counts.iloc[0] == sex_counts.iloc[1]:
                raise CohortPreflightError("Cannot derive an unambiguous train-only sex mode.")
            statistics["sex"] = float(sex_counts.index[0])
        for patient_id, row in selected.iterrows():
            fields = [column for column in statistics if pd.isna(row[column])]
            for column in fields:
                selected.at[patient_id, column] = statistics[column]
            if fields:
                imputed_fields[str(patient_id)] = fields
    return selected, missing_counts, statistics, imputed_fields


def load_vitaldb_virtual_cohort(
    dataset_dir: Path,
    *,
    demographics_csv: Path | None = None,
    project_data: Path | None = None,
    project_data_root: Path | None = None,
    allow_test_demographics: bool = True,
    missing_policy: Literal["error", "train_impute"] = "error",
    allow_official_demographics_download: bool = False,
    official_demographics_cache: Path | None = None,
) -> CohortBundle:
    """Resolve split-safe demographics without reading any trajectory arrays."""

    dataset_dir = dataset_dir.expanduser().resolve()
    if project_data is not None and project_data_root is not None:
        if project_data.expanduser().resolve() != project_data_root.expanduser().resolve():
            raise ValueError("project_data and project_data_root refer to different paths.")
    project_root = project_data_root or project_data
    if project_root is not None:
        project_root = project_root.expanduser().resolve()
    if not allow_test_demographics:
        raise CohortPreflightError(
            "This frozen 68/15/15 virtual cohort requires test IDs and demographics in its "
            "manifest; test trajectories and outcomes remain sealed."
        )
    manifest = _read_split_manifest(dataset_dir)
    official_provenance: dict[str, Any] = {}
    try:
        raw, source, source_kind, source_columns, searched = _resolve_demographics(
            dataset_dir,
            demographics_csv=demographics_csv,
            project_data_root=project_root,
        )
    except CohortPreflightError as resolution_error:
        if not allow_official_demographics_download or demographics_csv is not None:
            raise
        cache_path = official_demographics_cache
        if cache_path is None and project_root is not None:
            cache_path = project_root / "clinical/vitaldb_ppo_cohort_demographics.csv"
        if cache_path is None:
            raise CohortPreflightError(
                f"{resolution_error} Official fallback also requires "
                "--official-demographics-cache or --project-data-root."
            ) from resolution_error
        cache_path = cache_path.expanduser().resolve()
        all_case_ids = (
            *manifest.train_patient_ids,
            *manifest.validation_patient_ids,
            *manifest.test_patient_ids,
        )
        try:
            official_provenance = ensure_official_demographics_cache(
                all_case_ids, cache_path
            )
            raw, source, _, source_columns, searched = _resolve_demographics(
                dataset_dir,
                demographics_csv=cache_path,
                project_data_root=project_root,
            )
            source_kind = "official_vitaldb_clinical_api_cache"
        except Exception as official_error:
            raise CohortPreflightError(
                f"{resolution_error} Explicit official clinical-metadata fallback failed: "
                f"{official_error}"
            ) from official_error
    demographics, duplicate_rows_collapsed = _deduplicate_demographics(raw)
    selected, missing_counts, statistics, imputed_fields = _prepare_demographics(
        demographics, manifest, missing_policy=missing_policy
    )
    expected_ids = (
        *manifest.train_patient_ids,
        *manifest.validation_patient_ids,
        *manifest.test_patient_ids,
    )
    source_records = selected.loc[list(expected_ids)].reset_index().to_dict("records")
    source_fingerprint = _canonical_hash(source_records)
    patients: dict[str, PatientDemographics] = {}
    records: list[dict[str, Any]] = []
    for patient_id in expected_ids:
        row = selected.loc[patient_id]
        try:
            patient = PatientDemographics(
                age_years=float(row["age"]),
                sex="male" if int(row["sex"]) == 1 else "female",
                height_cm=float(row["height"]),
                weight_kg=float(row["weight"]),
            )
        except (TypeError, ValueError) as exc:
            raise CohortPreflightError(
                f"Invalid PK-PD demographics for caseid {patient_id}; no silent exclusion: {exc}"
            ) from exc
        patients[patient_id] = patient
        records.append(
            {
                "patient_id": patient_id,
                "split": manifest.split_for(patient_id),
                "imputed_fields": "|".join(imputed_fields.get(patient_id, [])),
                **patient.as_dict(),
            }
        )
    fingerprint_records = [
        {key: value for key, value in record.items() if key != "imputed_fields"}
        for record in records
    ]
    payload = {
        "splits": {
            "train": list(manifest.train_patient_ids),
            "validation": list(manifest.validation_patient_ids),
            "test": list(manifest.test_patient_ids),
        },
        "patients": fingerprint_records,
    }
    access_manifest = {
        "demographics_source": source,
        "demographics_source_kind": source_kind,
        "demographics_source_columns": list(source_columns),
        "demographics_source_fingerprint": source_fingerprint,
        "test_split_membership_loaded": True,
        "test_demographics_loaded": True,
        "test_trajectory_loaded": False,
        "test_outcomes_evaluated": False,
        "test_policy_rollout_performed": False,
        "test_data_used_for_imputation": False,
        "clinical_trajectory_replay": False,
        "split_counts": {
            "train": len(manifest.train_patient_ids),
            "validation": len(manifest.validation_patient_ids),
            "test": len(manifest.test_patient_ids),
        },
        "split_overlaps": {"train_validation": 0, "train_test": 0, "validation_test": 0},
        "extra_demographics_rows_ignored": int(len(demographics) - len(expected_ids)),
        "duplicate_demographics_rows_collapsed": duplicate_rows_collapsed,
        "missing_policy": missing_policy,
        "missing_counts_before_policy": missing_counts,
        "imputation_statistics_source": "train split only" if statistics else "not applicable",
        "imputation_statistics": statistics,
        "imputed_case_fields": imputed_fields,
        "searched_candidate_paths": list(searched),
        "selected_demographics_path": searched[-1] if searched else source,
        "official_clinical_metadata": official_provenance,
    }
    return CohortBundle(
        cohort=PatientCohort(patients=patients, manifest=manifest),
        demographics_source=source,
        demographics_source_kind=source_kind,
        demographics_source_columns=source_columns,
        demographics_source_fingerprint=source_fingerprint,
        split_source=str(dataset_dir / "splits"),
        fingerprint=_canonical_hash(payload),
        patient_records=tuple(records),
        missing_demographics=missing_counts,
        imputation_statistics=statistics,
        access_manifest=access_manifest,
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
