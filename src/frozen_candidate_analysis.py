"""Validation-only comparison of reused and newly trained frozen candidates."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd

from src.frozen_candidate_retraining import (
    ANCHOR_MAPPING,
    DATASET_FINGERPRINT_FILES,
    FROZEN_CANDIDATES,
    MODELS,
    NEW_CANDIDATES,
    SEEDS,
    dataset_fingerprint,
    dump_json,
    load_frozen_candidates,
    sha256_file,
    validate_run_directory,
)
from src.group_retraining_analysis import hierarchical_paired_bootstrap, load_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

LOGGER = logging.getLogger(__name__)

PRIMARY_METRIC = "validation_patient_level_mae"
BOOTSTRAP_SEED = 20260717
BOOTSTRAP_REPLICATES = 10_000
REFERENCE = "full17_reference"
CONTRASTS = (
    ("no_respiratory_anchor", REFERENCE),
    ("compact11_anchor", REFERENCE),
    ("strict_consensus", REFERENCE),
    ("compact_consensus", REFERENCE),
    ("strict_consensus", "compact_consensus"),
    ("strict_consensus", "compact11_anchor"),
    ("compact_consensus", "no_respiratory_anchor"),
    ("compact_consensus", "compact11_anchor"),
)
PREDICTION_ALIGNMENT_COLUMNS = (
    "sample_index",
    "case_id",
    "target_timestamp",
    "observed_future_bis",
)
PREDICTION_COLUMNS = (*PREDICTION_ALIGNMENT_COLUMNS, "predicted_future_bis")


def load_registry(path: Path) -> list[dict[str, Any]]:
    """Load and validate the exact 5-by-2-by-5 source registry."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read candidate registry {path}: {error}") from error
    if not isinstance(payload, list):
        raise ValueError("candidate_source_registry.json must contain a list.")
    try:
        keys = [
            (row["candidate"], row["model"], int(row["seed"])) for row in payload
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Registry contains an invalid run identity: {error}") from error
    expected = {
        (candidate, model, seed)
        for candidate in FROZEN_CANDIDATES
        for model in MODELS
        for seed in SEEDS
    }
    if len(keys) != 50 or len(set(keys)) != 50 or set(keys) != expected:
        raise ValueError(
            "Registry must contain exactly 50 unique candidate/model/seed combinations."
        )
    if sum(row.get("source_type") == "reused_prior" for row in payload) != 30:
        raise ValueError("Registry must reference exactly 30 reused prior runs.")
    if sum(row.get("source_type") == "newly_trained" for row in payload) != 20:
        raise ValueError("Registry must contain exactly 20 newly trained runs.")
    for row in payload:
        if row["candidate"] in ANCHOR_MAPPING:
            expected_source = "reused_prior"
        elif row["candidate"] in NEW_CANDIDATES:
            expected_source = "newly_trained"
        else:
            raise ValueError(f"Unknown registry candidate: {row['candidate']}")
        if row["source_type"] != expected_source:
            raise ValueError(
                f"Registry source type disagrees with candidate role: {row['candidate']}"
            )
    return payload


def validate_candidate_inventory(
    registry_path: Path, candidate_path: Path, dataset_dir: Path
) -> tuple[pd.DataFrame, dict[tuple[str, str, int], pd.DataFrame]]:
    """Validate all 50 sources, patient splits, and prediction alignment."""

    registry = load_registry(registry_path)
    candidates = load_frozen_candidates(candidate_path)
    fingerprint = dataset_fingerprint(dataset_dir)["combined_sha256"]
    rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, str, int], pd.DataFrame] = {}
    canonical_alignment: pd.DataFrame | None = None
    canonical_split: tuple[tuple[int, ...], tuple[int, ...]] | None = None

    for record in registry:
        key = (record["candidate"], record["model"], int(record["seed"]))
        if record.get("completion_status") != "complete":
            raise ValueError(f"Registry run is not complete: {key}")
        if record.get("test_evaluated") is not False:
            raise ValueError(f"Registry run is not test-sealed: {key}")
        if record.get("dataset_fingerprint") != fingerprint:
            raise ValueError(f"Dataset fingerprint mismatch for {key}.")
        expected_features = candidates.features[key[0]]
        if tuple(record.get("feature_names", ())) != expected_features:
            raise ValueError(f"Registry feature list mismatch for {key}.")
        if int(record.get("feature_count", -1)) != len(expected_features):
            raise ValueError(f"Registry feature count mismatch for {key}.")

        run_dir = Path(record["source_run_directory"])
        validated = validate_run_directory(
            run_dir,
            key[0],
            key[1],
            key[2],
            expected_features,
            dataset_dir,
        )
        config = validated["config"]
        status = validated["status"]
        if record.get("training_commit") != config["git_commit_hash"]:
            raise ValueError(f"Registry training commit mismatch for {key}.")
        split = (
            tuple(int(case) for case in config["selected_training_cases"]),
            tuple(int(case) for case in config["selected_validation_cases"]),
        )
        if canonical_split is None:
            canonical_split = split
        elif split != canonical_split:
            raise ValueError(f"Patient split mismatch for {key}.")

        prediction = (
            pd.read_csv(run_dir / "val_predictions.csv")
            .sort_values("sample_index", kind="stable")
            .reset_index(drop=True)
        )
        missing_columns = sorted(set(PREDICTION_COLUMNS) - set(prediction.columns))
        if missing_columns:
            raise ValueError(
                f"Validation prediction columns are incomplete for {key}: {missing_columns}"
            )
        if prediction["sample_index"].duplicated().any():
            raise ValueError(f"Duplicate validation sample indices for {key}.")
        if not np.isfinite(prediction.loc[:, PREDICTION_COLUMNS].to_numpy(float)).all():
            raise ValueError(f"Non-finite validation predictions for {key}.")
        alignment = prediction.loc[:, PREDICTION_ALIGNMENT_COLUMNS]
        if canonical_alignment is None:
            canonical_alignment = alignment
        elif not alignment.equals(canonical_alignment):
            raise ValueError(f"Validation prediction alignment mismatch for {key}.")

        metrics = load_json(run_dir / "val_metrics.json")
        history = pd.read_csv(run_dir / "training_history.csv")
        pooled = metrics["pooled_window"]["regression"]
        rows.append(
            {
                "candidate": key[0],
                "model": key[1],
                "seed": key[2],
                "source_type": record["source_type"],
                "source_run_directory": str(run_dir),
                "feature_count": len(expected_features),
                "feature_names": json.dumps(expected_features),
                PRIMARY_METRIC: float(metrics["patient_level"]["mae"]["mean"]),
                "validation_pooled_mae": float(pooled["mae"]),
                "validation_pooled_rmse": float(pooled["rmse"]),
                "best_epoch": int(status["best_epoch"]),
                "training_epochs": len(history),
                "training_commit": config["git_commit_hash"],
                "device": config["resolved_device"],
                "test_evaluated": False,
            }
        )
        predictions[key] = prediction

    summary = pd.DataFrame(rows).sort_values(
        ["candidate", "model", "seed"], kind="stable"
    )
    LOGGER.info("Validated %d test-sealed candidate runs.", len(summary))
    return summary.reset_index(drop=True), predictions


def aggregate_candidates(run_summary: pd.DataFrame) -> pd.DataFrame:
    """Aggregate five paired seeds for each candidate and model."""

    rows: list[dict[str, Any]] = []
    for (candidate, model), group in run_summary.groupby(["candidate", "model"]):
        values = group[PRIMARY_METRIC].to_numpy(float)
        rows.append(
            {
                "candidate": candidate,
                "model": model,
                "feature_count": int(group["feature_count"].iloc[0]),
                "seed_count": len(values),
                "mean_validation_patient_mae": float(values.mean()),
                "standard_deviation": float(values.std(ddof=1)),
                "median": float(np.median(values)),
                "min": float(values.min()),
                "max": float(values.max()),
                "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
            }
        )
    return pd.DataFrame(rows).sort_values(["model", "feature_count", "candidate"])


def patient_metrics(
    predictions: Mapping[tuple[str, str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Aggregate window errors to one MAE per candidate/model/seed/patient."""

    rows: list[pd.DataFrame] = []
    for (candidate, model, seed), frame in predictions.items():
        data = frame.assign(
            absolute_error=np.abs(
                frame["predicted_future_bis"] - frame["observed_future_bis"]
            )
        )
        patient = data.groupby("case_id", as_index=False).agg(
            patient_mae=("absolute_error", "mean"),
            window_count=("absolute_error", "size"),
        )
        patient.insert(0, "seed", seed)
        patient.insert(0, "model", model)
        patient.insert(0, "candidate", candidate)
        rows.append(patient)
    return pd.concat(rows, ignore_index=True)


def paired_candidate_statistics(
    run_summary: pd.DataFrame,
    patient: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare prescribed candidates with seed and patient pairing preserved."""

    delta_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for index, (candidate, reference) in enumerate(CONTRASTS):
            left = run_summary.query(
                "candidate == @candidate and model == @model"
            )[["seed", PRIMARY_METRIC]].rename(
                columns={PRIMARY_METRIC: "candidate_mae"}
            )
            right = run_summary.query(
                "candidate == @reference and model == @model"
            )[["seed", PRIMARY_METRIC]].rename(
                columns={PRIMARY_METRIC: "reference_mae"}
            )
            paired = left.merge(right, on="seed", validate="one_to_one")
            paired["delta"] = paired["candidate_mae"] - paired["reference_mae"]
            paired["relative_mae_change_percent"] = (
                100.0 * paired["delta"] / paired["reference_mae"]
            )
            paired.insert(0, "reference", reference)
            paired.insert(0, "candidate", candidate)
            paired.insert(0, "model", model)
            delta_rows.append(paired)

            values = paired["delta"].to_numpy(float)
            summary_rows.append(
                {
                    "model": model,
                    "candidate": candidate,
                    "reference": reference,
                    "mean_delta": float(values.mean()),
                    "delta_standard_deviation": float(values.std(ddof=1)),
                    "median_delta": float(np.median(values)),
                    "min_delta": float(values.min()),
                    "max_delta": float(values.max()),
                    "candidate_better_seed_count": int((values < 0).sum()),
                    "mean_relative_mae_change_percent": float(
                        paired["relative_mae_change_percent"].mean()
                    ),
                    "direction": "negative favors candidate",
                }
            )

            patient_left = patient.query(
                "candidate == @candidate and model == @model"
            )[["seed", "case_id", "patient_mae"]].rename(
                columns={"patient_mae": "candidate_mae"}
            )
            patient_right = patient.query(
                "candidate == @reference and model == @model"
            )[["seed", "case_id", "patient_mae"]].rename(
                columns={"patient_mae": "reference_mae"}
            )
            patient_pairs = patient_left.merge(
                patient_right, on=["seed", "case_id"], validate="one_to_one"
            )
            patient_pairs["paired_delta"] = (
                patient_pairs["candidate_mae"] - patient_pairs["reference_mae"]
            )
            bootstrap = hierarchical_paired_bootstrap(
                patient_pairs[["seed", "case_id", "paired_delta"]],
                replicates=replicates,
                seed=seed + index + 100 * MODELS.index(model),
            )
            bootstrap_rows.append(
                {
                    "model": model,
                    "candidate": candidate,
                    "reference": reference,
                    **bootstrap,
                }
            )
    return (
        pd.concat(delta_rows, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(bootstrap_rows),
    )


def paired_model_statistics(
    run_summary: pd.DataFrame,
    patient: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare Attention minus GRU with paired seeds and patients."""

    delta_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(FROZEN_CANDIDATES):
        gru = run_summary.query("candidate == @candidate and model == 'gru'")[[
            "seed",
            PRIMARY_METRIC,
        ]].rename(columns={PRIMARY_METRIC: "gru_mae"})
        attention = run_summary.query(
            "candidate == @candidate and model == 'attention'"
        )[["seed", PRIMARY_METRIC]].rename(columns={PRIMARY_METRIC: "attention_mae"})
        paired = gru.merge(attention, on="seed", validate="one_to_one")
        paired["delta"] = paired["attention_mae"] - paired["gru_mae"]
        paired.insert(0, "candidate", candidate)
        delta_rows.append(paired)

        values = paired["delta"].to_numpy(float)
        summary_rows.append(
            {
                "candidate": candidate,
                "mean_delta": float(values.mean()),
                "delta_standard_deviation": float(values.std(ddof=1)),
                "median_delta": float(np.median(values)),
                "min_delta": float(values.min()),
                "max_delta": float(values.max()),
                "attention_better_seed_count": int((values < 0).sum()),
                "direction": "negative favors attention",
            }
        )

        patient_gru = patient.query(
            "candidate == @candidate and model == 'gru'"
        )[["seed", "case_id", "patient_mae"]].rename(
            columns={"patient_mae": "gru_mae"}
        )
        patient_attention = patient.query(
            "candidate == @candidate and model == 'attention'"
        )[["seed", "case_id", "patient_mae"]].rename(
            columns={"patient_mae": "attention_mae"}
        )
        patient_pairs = patient_gru.merge(
            patient_attention, on=["seed", "case_id"], validate="one_to_one"
        )
        patient_pairs["paired_delta"] = (
            patient_pairs["attention_mae"] - patient_pairs["gru_mae"]
        )
        bootstrap = hierarchical_paired_bootstrap(
            patient_pairs[["seed", "case_id", "paired_delta"]],
            replicates=replicates,
            seed=seed + 1000 + index,
        )
        bootstrap_rows.append(
            {
                "candidate": candidate,
                "comparison": "attention_minus_gru",
                **bootstrap,
            }
        )
    return (
        pd.concat(delta_rows, ignore_index=True),
        pd.DataFrame(summary_rows),
        pd.DataFrame(bootstrap_rows),
    )


def candidate_pareto(
    aggregate: pd.DataFrame, contrasts: pd.DataFrame
) -> pd.DataFrame:
    """Mark Pareto membership and separate, non-automatic decision aids."""

    rows: list[dict[str, Any]] = []
    for model, group in aggregate.groupby("model"):
        records = group.to_dict("records")
        best = min(
            records,
            key=lambda row: (
                row["mean_validation_patient_mae"],
                row["feature_count"],
                row["candidate"],
            ),
        )["candidate"]
        for row in records:
            dominators = [
                other["candidate"]
                for other in records
                if other["candidate"] != row["candidate"]
                and other["feature_count"] <= row["feature_count"]
                and other["mean_validation_patient_mae"]
                <= row["mean_validation_patient_mae"]
                and (
                    other["feature_count"] < row["feature_count"]
                    or other["mean_validation_patient_mae"]
                    < row["mean_validation_patient_mae"]
                )
            ]
            rows.append(
                {
                    "model": model,
                    "candidate": row["candidate"],
                    "feature_count": row["feature_count"],
                    "mean_validation_patient_mae": row[
                        "mean_validation_patient_mae"
                    ],
                    "dominated": bool(dominators),
                    "pareto_frontier": not dominators,
                    "dominated_by": ",".join(sorted(dominators)),
                    "best_observed_validation_candidate": row["candidate"] == best,
                }
            )

    result = pd.DataFrame(rows)
    result["simplest_non_dominated_candidate"] = False
    for model, frontier in result[result["pareto_frontier"]].groupby("model"):
        simplest = frontier.sort_values(
            ["feature_count", "mean_validation_patient_mae", "candidate"]
        ).iloc[0]["candidate"]
        result.loc[
            (result["model"] == model) & (result["candidate"] == simplest),
            "simplest_non_dominated_candidate",
        ] = True

    against_full = contrasts[contrasts["reference"] == REFERENCE]
    consistent = (
        against_full.sort_values(
            ["model", "candidate_better_seed_count", "mean_delta", "candidate"],
            ascending=[True, False, True, True],
        )
        .groupby("model")
        .head(1)
    )
    consistent_keys = set(zip(consistent["model"], consistent["candidate"], strict=True))
    result["most_seed_consistent_candidate"] = [
        (row.model, row.candidate) in consistent_keys for row in result.itertuples()
    ]
    cautions = {
        "compact11_anchor": "remifentanil group fully removed",
        "strict_consensus": (
            "only rftn_volume retained; not an optimal-control claim"
        ),
    }
    result["control_caution_flags"] = result["candidate"].map(cautions).fillna("")
    return result.sort_values(["model", "feature_count", "candidate"])


def build_evidence_table(
    aggregate: pd.DataFrame, contrasts: pd.DataFrame, pareto: pd.DataFrame
) -> pd.DataFrame:
    """Combine descriptive, paired, and decision-aid evidence without selecting."""

    paired_full = contrasts[contrasts["reference"] == REFERENCE].drop(
        columns=["reference"]
    )
    evidence = aggregate.merge(
        pareto.drop(
            columns=["feature_count", "mean_validation_patient_mae"],
            errors="ignore",
        ),
        on=["candidate", "model"],
        validate="one_to_one",
    )
    return evidence.merge(
        paired_full,
        on=["candidate", "model"],
        how="left",
        validate="one_to_one",
    )


def save_figures(
    run_summary: pd.DataFrame,
    candidate_deltas: pd.DataFrame,
    candidate_bootstrap: pd.DataFrame,
    pareto: pd.DataFrame,
    candidate_features: Mapping[str, Sequence[str]],
    output_dir: Path,
) -> list[Path]:
    """Save the six required validation-only comparison figures."""

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(13, 6))
    groups: list[pd.Series] = []
    labels: list[str] = []
    for model in MODELS:
        for candidate in FROZEN_CANDIDATES:
            groups.append(
                run_summary.query("model == @model and candidate == @candidate")[
                    PRIMARY_METRIC
                ]
            )
            labels.append(f"{candidate}\n{model}")
    ax.boxplot(groups, showfliers=False)
    for index, values in enumerate(groups, start=1):
        ax.scatter(np.full(len(values), index), values, color="#264653", zorder=3)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("Validation-only candidate seed MAE")
    ax.set_ylabel("Patient-level MAE")
    fig.tight_layout()
    paths.append(figures_dir / "candidate_seed_mae.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    full_deltas = candidate_deltas[candidate_deltas["reference"] == REFERENCE]
    comparison_candidates = [name for name in FROZEN_CANDIDATES if name != REFERENCE]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        model_rows = full_deltas[full_deltas["model"] == model]
        for seed in SEEDS:
            seed_rows = model_rows[model_rows["seed"] == seed].set_index("candidate")
            ax.plot(
                range(len(comparison_candidates)),
                seed_rows.loc[comparison_candidates, "delta"],
                marker="o",
                label=str(seed),
            )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xticks(
            range(len(comparison_candidates)),
            comparison_candidates,
            rotation=30,
            ha="right",
        )
        ax.set_title(f"{model}: candidate minus full17")
    axes[0].set_ylabel("Paired patient-level MAE delta")
    axes[1].legend(title="Seed", fontsize=8)
    fig.tight_layout()
    paths.append(figures_dir / "paired_candidate_lines.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    bootstrap = candidate_bootstrap.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    y = np.arange(len(bootstrap))
    means = bootstrap["point_estimate_mean_delta"].to_numpy()
    lower = bootstrap["percentile_95_ci_lower"].to_numpy()
    upper = bootstrap["percentile_95_ci_upper"].to_numpy()
    ax.errorbar(means, y, xerr=np.vstack((means - lower, upper - means)), fmt="o")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_yticks(
        y,
        [
            f"{row.model}: {row.candidate} - {row.reference}"
            for row in bootstrap.itertuples()
        ],
    )
    ax.set_title("Validation-only hierarchical paired bootstrap")
    fig.tight_layout()
    paths.append(figures_dir / "candidate_bootstrap_forest.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, MODELS, strict=True):
        for row in pareto[pareto["model"] == model].itertuples():
            color = "#2a9d8f" if row.pareto_frontier else "#8d99ae"
            ax.scatter(row.feature_count, row.mean_validation_patient_mae, color=color)
            ax.annotate(
                row.candidate,
                (row.feature_count, row.mean_validation_patient_mae),
                fontsize=8,
            )
        ax.set_title(f"{model} validation-only Pareto")
        ax.set_xlabel("Feature count")
    axes[0].set_ylabel("Mean patient-level MAE")
    fig.tight_layout()
    paths.append(figures_dir / "candidate_pareto.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharey=True)
    for ax, candidate in zip(axes.flat, FROZEN_CANDIDATES, strict=False):
        for seed in SEEDS:
            rows = run_summary.query(
                "candidate == @candidate and seed == @seed"
            ).set_index("model")
            ax.plot(
                [0, 1],
                rows.loc[["gru", "attention"], PRIMARY_METRIC],
                marker="o",
            )
        ax.set_xticks([0, 1], ["GRU", "Attention"])
        ax.set_title(candidate)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    paths.append(figures_dir / "gru_attention_pairs.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)

    all_features = list(
        dict.fromkeys(
            feature
            for candidate in FROZEN_CANDIDATES
            for feature in candidate_features[candidate]
        )
    )
    feature_matrix = np.asarray(
        [
            [feature in candidate_features[candidate] for feature in all_features]
            for candidate in FROZEN_CANDIDATES
        ]
    )
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.imshow(feature_matrix, cmap="Greens", aspect="auto")
    ax.set_xticks(range(len(all_features)), all_features, rotation=90)
    ax.set_yticks(range(len(FROZEN_CANDIDATES)), FROZEN_CANDIDATES)
    ax.set_title("Frozen candidate feature matrix")
    fig.tight_layout()
    paths.append(figures_dir / "candidate_feature_matrix.png")
    fig.savefig(paths[-1], dpi=160)
    plt.close(fig)
    return paths


def build_report(
    aggregate: pd.DataFrame, contrasts: pd.DataFrame, evidence: pd.DataFrame
) -> str:
    """Build a validation-only report with the required interpretation warnings."""

    return f"""# Frozen Candidate Validation-Only Comparison

## Scope
This report compares 30 reused anchor runs and 20 newly trained discovery-candidate runs. All 50 runs use train/validation only; the held-out test split remains sealed.

## Adaptive validation warning
Some group anchors were retained after earlier inspection of the same validation set. The same validation cases have been reused across multiple development stages, so their performance can be optimistic. `strict_consensus` and `compact_consensus` were generated by train-only selectors, but their retraining comparison still uses this repeatedly consulted validation set.

## Aggregate evidence
```text
{aggregate.to_string(index=False)}
```

## Paired candidate evidence
Negative deltas favor the first-named candidate. P-values are not automatic winner rules.
```text
{contrasts.to_string(index=False)}
```

## Decision aids
Best observed validation performance, the simplest non-dominated point, seed consistency, and control cautions are reported separately. No final state is selected automatically.
```text
{evidence.to_string(index=False)}
```

## Interpretation limits
- No formal non-inferiority margin was specified; equivalence and non-inferiority are not claimed.
- `compact11_anchor` removes all remifentanil information and must not be adopted directly as a control-aware state.
- `strict_consensus` retains only `rftn_volume`; this is not evidence that it is an optimal control state.
- Predictive utility does not guarantee RL control utility.
- After a final candidate is frozen, perform one pre-specified test evaluation or use separate unseen cases.
"""


def input_fingerprints(
    registry_path: Path, candidate_path: Path, dataset_dir: Path
) -> list[dict[str, Any]]:
    """Fingerprint validation-safe definitions, data metadata, and run inputs."""

    paths = [
        registry_path,
        candidate_path,
        *(dataset_dir / name for name in DATASET_FINGERPRINT_FILES),
    ]
    for record in load_registry(registry_path):
        run_dir = Path(record["source_run_directory"])
        paths.extend(
            run_dir / name
            for name in (
                "config.json",
                "run_status.json",
                "training_history.csv",
                "val_metrics.json",
                "val_predictions.csv",
            )
        )
    return [
        {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]


def run_frozen_candidate_analysis(
    registry_path: Path,
    candidate_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Validate, compare, and report all 50 runs without touching source artifacts."""

    expected_output = registry_path.parent / "analysis"
    if output_dir.resolve() != expected_output.resolve():
        raise ValueError("Output must be <retraining_root>/analysis.")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive.")

    run_summary, predictions = validate_candidate_inventory(
        registry_path, candidate_path, dataset_dir
    )
    aggregate = aggregate_candidates(run_summary)
    patients = patient_metrics(predictions)
    candidate_deltas, candidate_contrasts, candidate_bootstrap = (
        paired_candidate_statistics(
            run_summary,
            patients,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
    )
    model_deltas, model_contrasts, model_bootstrap = paired_model_statistics(
        run_summary,
        patients,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    pareto = candidate_pareto(aggregate, candidate_contrasts)
    evidence = build_evidence_table(aggregate, candidate_contrasts, pareto)

    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "validated_candidate_run_inventory.csv": run_summary[
            [
                "candidate",
                "model",
                "seed",
                "source_type",
                "source_run_directory",
                "feature_count",
                "training_commit",
                "device",
                "test_evaluated",
            ]
        ],
        "candidate_validation_run_summary.csv": run_summary,
        "candidate_validation_aggregate.csv": aggregate,
        "paired_candidate_seed_deltas.csv": candidate_deltas,
        "paired_candidate_contrasts.csv": candidate_contrasts,
        "paired_model_seed_deltas.csv": model_deltas,
        "paired_model_contrasts.csv": model_contrasts,
        "hierarchical_bootstrap_candidate_contrasts.csv": candidate_bootstrap,
        "hierarchical_bootstrap_model_contrasts.csv": model_bootstrap,
        "candidate_pareto.csv": pareto,
        "candidate_evidence_table.csv": evidence,
    }
    for name, frame in tables.items():
        frame.to_csv(output_dir / name, index=False)

    candidates = load_frozen_candidates(candidate_path)
    figures = save_figures(
        run_summary,
        candidate_deltas,
        candidate_bootstrap,
        pareto,
        candidates.features,
        output_dir,
    )
    report_path = output_dir / "frozen_candidate_validation_report.md"
    report_path.write_text(
        build_report(aggregate, candidate_contrasts, evidence), encoding="utf-8"
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    manifest = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_git_commit": commit,
        "run_count": 50,
        "reused_prior_count": 30,
        "newly_trained_count": 20,
        "candidates": list(FROZEN_CANDIDATES),
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "primary_metric": PRIMARY_METRIC,
        "primary_metric_direction": "lower is better",
        "test_split_sealed": True,
        "test_split_read_by_analysis": False,
        "adaptive_validation_warning": True,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_resampling_units": "paired seed and validation patient",
        "input_fingerprints": input_fingerprints(
            registry_path, candidate_path, dataset_dir
        ),
        "generated_outputs": sorted(
            [
                *tables,
                "analysis_manifest.json",
                report_path.name,
                *(str(path.relative_to(output_dir)) for path in figures),
            ]
        ),
    }
    dump_json(manifest, output_dir / "analysis_manifest.json")
    LOGGER.info("Wrote validation-only analysis to %s.", output_dir)
    return {
        "output_dir": str(output_dir),
        "run_count": 50,
        "test_split_sealed": True,
    }
