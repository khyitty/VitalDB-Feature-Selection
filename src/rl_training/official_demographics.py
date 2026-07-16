"""Explicitly authorized VitalDB clinical-metadata cache construction."""

from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.request import Request, urlopen

import pandas as pd

from .io import atomic_write_dataframe, atomic_write_json


VITALDB_CASES_ENDPOINT = "https://api.vitaldb.net/cases"
OFFICIAL_DEMOGRAPHIC_COLUMNS = ("caseid", "age", "sex", "height", "weight")


def _canonical_case_ids(case_ids: Iterable[str]) -> tuple[int, ...]:
    values = tuple(sorted({int(value) for value in case_ids}))
    if not values:
        raise ValueError("At least one caseid is required for official demographics download.")
    return values


def _selected_fingerprint(frame: pd.DataFrame) -> str:
    payload = frame.loc[:, OFFICIAL_DEMOGRAPHIC_COLUMNS].to_json(
        orient="records", double_precision=15
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _validate_selected(frame: pd.DataFrame, case_ids: tuple[int, ...]) -> pd.DataFrame:
    missing_columns = sorted(set(OFFICIAL_DEMOGRAPHIC_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"VitalDB clinical metadata lacks columns {missing_columns}; "
            f"found {list(frame.columns)}."
        )
    selected = frame.loc[:, OFFICIAL_DEMOGRAPHIC_COLUMNS].copy()
    selected["caseid"] = pd.to_numeric(selected["caseid"], errors="raise").astype(int)
    selected = selected[selected["caseid"].isin(case_ids)].sort_values("caseid")
    if selected["caseid"].duplicated().any():
        duplicates = selected.loc[selected["caseid"].duplicated(False), "caseid"].tolist()
        raise ValueError(f"Official VitalDB metadata has duplicate caseid values: {duplicates}")
    observed = set(selected["caseid"])
    missing_ids = sorted(set(case_ids) - observed)
    extra_ids = sorted(observed - set(case_ids))
    if missing_ids or extra_ids or len(selected) != len(case_ids):
        raise ValueError(
            "Official VitalDB demographics do not exactly cover the frozen split IDs: "
            f"missing={missing_ids}, extra={extra_ids}."
        )
    return selected.reset_index(drop=True)


def ensure_official_demographics_cache(
    case_ids: Iterable[str],
    output_csv: Path,
    *,
    endpoint: str = VITALDB_CASES_ENDPOINT,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Cache only official clinical demographics for the requested case IDs.

    Existing valid caches are reused without network access. The endpoint contains
    clinical information only; no waveform, numeric-track, outcome, or trajectory
    endpoint is requested.
    """

    canonical_ids = _canonical_case_ids(case_ids)
    provenance_path = output_csv.with_suffix(".provenance.json")
    if output_csv.is_file():
        selected = _validate_selected(pd.read_csv(output_csv), canonical_ids)
        if not provenance_path.is_file():
            raise ValueError(
                f"Official demographics cache lacks provenance sidecar: {provenance_path}"
            )
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        observed_fingerprint = _selected_fingerprint(selected)
        if provenance.get("endpoint") != endpoint:
            raise ValueError(
                "Official demographics cache endpoint differs from the requested endpoint: "
                f"observed={provenance.get('endpoint')!r}, expected={endpoint!r}."
            )
        if provenance.get("selected_demographics_sha256") != observed_fingerprint:
            raise ValueError("Official demographics cache fingerprint does not match its sidecar.")
        return {
            **provenance,
            "cache_path": str(output_csv.resolve()),
            "cache_reused": True,
            "selected_case_count": len(selected),
            "selected_demographics_sha256": observed_fingerprint,
            "trajectory_downloaded": False,
            "outcomes_downloaded": False,
        }

    request = Request(endpoint, headers={"User-Agent": "VitalDB-Feature-Selection/1.0"})
    with urlopen(request, timeout=timeout_seconds) as response:
        compressed = response.read()
    response_hash = hashlib.sha256(compressed).hexdigest()
    try:
        payload = gzip.decompress(compressed) if compressed[:2] == bytes((31, 139)) else compressed
        clinical = pd.read_csv(io.BytesIO(payload))
    except Exception as exc:
        raise RuntimeError(
            f"Unable to decode official VitalDB clinical metadata from {endpoint}: {exc}"
        ) from exc
    selected = _validate_selected(clinical, canonical_ids)
    if selected.loc[:, ("age", "sex", "height", "weight")].isna().any().any():
        missing = selected.loc[
            selected.loc[:, ("age", "sex", "height", "weight")].isna().any(axis=1),
            "caseid",
        ].tolist()
        raise ValueError(
            f"Official VitalDB demographics are incomplete for caseid values: {missing}."
        )
    provenance = {
        "source": "VitalDB official Web API clinical information",
        "endpoint": endpoint,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "response_bytes": len(compressed),
        "response_sha256": response_hash,
        "selected_case_count": len(selected),
        "selected_case_ids": selected["caseid"].tolist(),
        "selected_columns": list(OFFICIAL_DEMOGRAPHIC_COLUMNS),
        "selected_demographics_sha256": _selected_fingerprint(selected),
        "trajectory_downloaded": False,
        "outcomes_downloaded": False,
        "cache_path": str(output_csv.resolve()),
        "cache_reused": False,
    }
    atomic_write_dataframe(output_csv, selected)
    atomic_write_json(provenance_path, provenance)
    return provenance
