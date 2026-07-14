"""PyTorch datasets and sampling utilities for saved BIS prediction arrays."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

EXPECTED_NPZ_KEYS = (
    "X_dynamic",
    "X_static",
    "observation_mask",
    "y_bis",
    "y_high_bis",
    "y_low_bis",
)


class VitalBISDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Load one split of the leakage-safe future-BIS dataset.

    Arrays are decompressed once from the NPZ archive and retained by reference. Item
    tensors use ``torch.from_numpy`` so no additional array copy is made per sample.
    """

    def __init__(self, dataset_dir: Path | str, split: str, validate: bool = True) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        npz_path = self.dataset_dir / f"{split}.npz"
        metadata_path = self.dataset_dir / f"{split}_metadata.csv"
        dataset_metadata_path = self.dataset_dir / "dataset_metadata.json"
        for path in (npz_path, metadata_path, dataset_metadata_path):
            if not path.exists():
                raise FileNotFoundError(f"Expected dataset artifact is missing: {path}")

        with dataset_metadata_path.open("r", encoding="utf-8") as handle:
            self.dataset_metadata = json.load(handle)
        self.dynamic_feature_names = tuple(
            self.dataset_metadata["dynamic_feature_names"]
        )
        self.static_feature_names = tuple(self.dataset_metadata["static_feature_names"])

        with np.load(npz_path, allow_pickle=False) as archive:
            missing_keys = sorted(set(EXPECTED_NPZ_KEYS) - set(archive.files))
            if missing_keys:
                raise ValueError(f"{npz_path} is missing arrays: {missing_keys}")
            self.arrays = {key: archive[key] for key in EXPECTED_NPZ_KEYS}
        self.metadata = pd.read_csv(metadata_path)
        required_metadata = {"case_id", "target_timestamp"}
        missing_columns = sorted(required_metadata - set(self.metadata.columns))
        if missing_columns:
            raise ValueError(f"{metadata_path} is missing columns: {missing_columns}")
        self._case_ids = self.metadata["case_id"].to_numpy(dtype=np.int64, copy=False)
        self._target_timestamps = self.metadata["target_timestamp"].to_numpy(
            dtype=np.int64, copy=False
        )

        if validate:
            self._validate()

    def _validate(self) -> None:
        lengths = {name: len(array) for name, array in self.arrays.items()}
        lengths["metadata"] = len(self.metadata)
        if len(set(lengths.values())) != 1:
            raise ValueError(f"NPZ and metadata lengths do not agree: {lengths}")

        dynamic_shape = self.arrays["X_dynamic"].shape
        static_shape = self.arrays["X_static"].shape
        mask_shape = self.arrays["observation_mask"].shape
        expected_steps = int(self.dataset_metadata["history_steps"])
        if dynamic_shape[1:] != (expected_steps, len(self.dynamic_feature_names)):
            raise ValueError(
                "X_dynamic shape does not match dataset feature metadata: "
                f"{dynamic_shape[1:]}"
            )
        if static_shape[1:] != (len(self.static_feature_names),):
            raise ValueError(
                "X_static shape does not match dataset feature metadata: "
                f"{static_shape[1:]}"
            )
        if mask_shape != dynamic_shape:
            raise ValueError(
                f"observation_mask shape {mask_shape} does not match {dynamic_shape}"
            )
        for name, array in self.arrays.items():
            if not np.isfinite(array).all():
                raise ValueError(f"Array {name!r} contains NaN or infinite values.")

        y_bis = self.arrays["y_bis"]
        expected_high = (y_bis > 60.0).astype(self.arrays["y_high_bis"].dtype)
        expected_low = (y_bis < 40.0).astype(self.arrays["y_low_bis"].dtype)
        if not np.array_equal(expected_high, self.arrays["y_high_bis"]):
            raise ValueError("y_high_bis labels do not agree with y_bis > 60.")
        if not np.array_equal(expected_low, self.arrays["y_low_bis"]):
            raise ValueError("y_low_bis labels do not agree with y_bis < 40.")

    def __len__(self) -> int:
        return len(self.arrays["y_bis"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "X_dynamic": torch.from_numpy(self.arrays["X_dynamic"][index]),
            "X_static": torch.from_numpy(self.arrays["X_static"][index]),
            "observation_mask": torch.from_numpy(
                self.arrays["observation_mask"][index]
            ),
            "y_bis": torch.as_tensor(self.arrays["y_bis"][index], dtype=torch.float32),
            "y_high_bis": torch.as_tensor(
                self.arrays["y_high_bis"][index], dtype=torch.int64
            ),
            "y_low_bis": torch.as_tensor(
                self.arrays["y_low_bis"][index], dtype=torch.int64
            ),
            "case_id": torch.as_tensor(
                self._case_ids[index], dtype=torch.int64
            ),
            "target_timestamp": torch.as_tensor(
                self._target_timestamps[index], dtype=torch.int64
            ),
            "sample_index": torch.as_tensor(index, dtype=torch.int64),
        }

    @property
    def case_ids(self) -> np.ndarray:
        """Return case IDs in sample order without copying when pandas permits."""

        return self._case_ids

    def indices_for_cases(self, case_ids: Sequence[int]) -> np.ndarray:
        """Return sample indices belonging to the requested cases."""

        return np.flatnonzero(np.isin(self.case_ids, np.asarray(case_ids, dtype=int)))


def case_balanced_weights(case_ids: Sequence[int] | np.ndarray) -> torch.Tensor:
    """Give every case equal total expected sampling mass."""

    values = np.asarray(case_ids, dtype=np.int64)
    if values.size == 0:
        raise ValueError("Cannot construct sampling weights for an empty dataset.")
    unique, counts = np.unique(values, return_counts=True)
    inverse_counts = dict(zip(unique.tolist(), (1.0 / counts).tolist(), strict=True))
    return torch.as_tensor(
        [inverse_counts[int(case_id)] for case_id in values], dtype=torch.double
    )


def make_case_balanced_sampler(
    case_ids: Sequence[int] | np.ndarray,
    seed: int,
    num_samples: int | None = None,
) -> WeightedRandomSampler:
    """Create a deterministic replacement sampler balanced across cases."""

    weights = case_balanced_weights(case_ids)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights) if num_samples is None else num_samples,
        replacement=True,
        generator=generator,
    )


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python random state for a DataLoader worker."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
