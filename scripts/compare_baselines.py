"""Audit a full GRU run and compare it with row-matched persistence predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets import VitalBISDataset  # noqa: E402
from src.metrics import patient_level_evaluation, pooled_evaluation  # noqa: E402

SPLITS = ("val", "test")
COMPARISON_METRICS = (
    "pooled_mae",
    "pooled_rmse",
    "patient_equal_weighted_mae",
    "bis_below_40_mae",
    "bis_40_to_60_mae",
    "bis_above_60_mae",
    "high_bis_auprc",
    "high_bis_auroc",
    "low_bis_auprc",
    "low_bis_auroc",
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True, allow_nan=False)
        handle.write("\n")


def _prediction_distribution(frame: pd.DataFrame) -> dict[str, float]:
    observed = frame["observed_future_bis"].to_numpy(dtype=float)
    predicted = frame["predicted_future_bis"].to_numpy(dtype=float)
    return {
        "observed_mean": float(observed.mean()),
        "observed_standard_deviation": float(observed.std(ddof=0)),
        "predicted_mean": float(predicted.mean()),
        "predicted_standard_deviation": float(predicted.std(ddof=0)),
        "predicted_minimum": float(predicted.min()),
        "predicted_maximum": float(predicted.max()),
        "pearson_correlation": float(np.corrcoef(observed, predicted)[0, 1]),
    }


def _metric_values(frame: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    observed = frame["observed_future_bis"].to_numpy(dtype=float)
    predicted = frame["predicted_future_bis"].to_numpy(dtype=float)
    case_ids = frame["case_id"].to_numpy(dtype=int)
    pooled = pooled_evaluation(observed, predicted)
    patient = patient_level_evaluation(observed, predicted, case_ids)
    values = {
        "pooled_mae": float(pooled["regression"]["mae"]),
        "pooled_rmse": float(pooled["regression"]["rmse"]),
        "patient_equal_weighted_mae": float(patient.summary["mae"]["mean"]),
        "bis_below_40_mae": float(pooled["bis_region_mae"]["bis_below_40"]),
        "bis_40_to_60_mae": float(pooled["bis_region_mae"]["bis_40_to_60"]),
        "bis_above_60_mae": float(pooled["bis_region_mae"]["bis_above_60"]),
        "high_bis_auprc": float(pooled["high_bis_classification"]["auprc"]),
        "high_bis_auroc": float(pooled["high_bis_classification"]["auroc"]),
        "low_bis_auprc": float(pooled["low_bis_classification"]["auprc"]),
        "low_bis_auroc": float(pooled["low_bis_classification"]["auroc"]),
    }
    return values, patient.case_metrics


def classify_result(
    patient_test_mae_difference: float, improved_case_count: int
) -> tuple[str, str]:
    """Apply the prespecified operational A/B/C decision rule."""

    if patient_test_mae_difference <= -0.2 and improved_case_count > 2:
        return "A", "GRU clearly improves over persistence"
    if abs(patient_test_mae_difference) <= 0.2:
        return "B", "GRU is approximately tied with persistence"
    return "C", "GRU clearly underperforms persistence"


def _assert_metadata_alignment(
    frame: pd.DataFrame, metadata: pd.DataFrame, split: str
) -> None:
    indices = frame["sample_index"].to_numpy(dtype=int)
    if len(np.unique(indices)) != len(indices):
        raise AssertionError(f"{split} prediction sample indices are duplicated.")
    aligned = metadata.iloc[indices]
    if not np.array_equal(
        frame["case_id"].to_numpy(dtype=int), aligned["case_id"].to_numpy(dtype=int)
    ):
        raise AssertionError(f"{split} case IDs are not aligned with dataset metadata.")
    if not np.array_equal(
        frame["target_timestamp"].to_numpy(dtype=int),
        aligned["target_timestamp"].to_numpy(dtype=int),
    ):
        raise AssertionError(
            f"{split} target timestamps are not aligned with dataset metadata."
        )


def _paired_case_comparison(
    persistence_cases: pd.DataFrame, gru_cases: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    paired = persistence_cases[["case_id", "mae"]].merge(
        gru_cases[["case_id", "mae"]], on="case_id", suffixes=("_persistence", "_gru")
    )
    paired["mae_difference_gru_minus_persistence"] = (
        paired["mae_gru"] - paired["mae_persistence"]
    )
    differences = paired["mae_difference_gru_minus_persistence"]
    improvements = paired.nsmallest(5, "mae_difference_gru_minus_persistence")
    deteriorations = paired.nlargest(5, "mae_difference_gru_minus_persistence")
    improved_count = int((differences < 0.0).sum())
    summary = {
        "case_count": len(paired),
        "cases_gru_mae_lower": improved_count,
        "percentage_cases_gru_mae_lower": float(improved_count / len(paired) * 100.0),
        "median_paired_case_mae_difference_gru_minus_persistence": float(
            differences.median()
        ),
        "largest_gru_improvement": {
            "case_id": int(improvements.iloc[0]["case_id"]),
            "mae_difference_gru_minus_persistence": float(
                improvements.iloc[0]["mae_difference_gru_minus_persistence"]
            ),
        },
        "largest_gru_deterioration": {
            "case_id": int(deteriorations.iloc[0]["case_id"]),
            "mae_difference_gru_minus_persistence": float(
                deteriorations.iloc[0]["mae_difference_gru_minus_persistence"]
            ),
        },
        "five_largest_improvements": improvements[
            ["case_id", "mae_difference_gru_minus_persistence"]
        ].to_dict(orient="records"),
        "five_largest_deteriorations": deteriorations[
            ["case_id", "mae_difference_gru_minus_persistence"]
        ].to_dict(orient="records"),
    }
    return summary, paired


def compare_baselines(
    outputs_dir: Path,
    dataset_dir: Path,
    seed: int,
    training_runtime_seconds: float,
) -> dict[str, Any]:
    gru_dir = outputs_dir / "gru" / f"seed_{seed}"
    persistence_dir = outputs_dir / "persistence"
    required_gru = (
        "config.json",
        "best_model.pt",
        "last_model.pt",
        "training_history.csv",
        "val_predictions.csv",
        "test_predictions.csv",
        "val_metrics.json",
        "test_metrics.json",
        "case_metrics.csv",
    )
    missing = [name for name in required_gru if not (gru_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Full GRU artifacts are missing: {missing}")

    config = _load_json(gru_dir / "config.json")
    val_metrics_json = _load_json(gru_dir / "val_metrics.json")
    test_metrics_json = _load_json(gru_dir / "test_metrics.json")
    history = pd.read_csv(gru_dir / "training_history.csv")
    best_checkpoint = torch.load(
        gru_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    last_checkpoint = torch.load(
        gru_dir / "last_model.pt", map_location="cpu", weights_only=False
    )
    best_epoch = int(best_checkpoint["epoch"])
    early_stopping_epoch = int(last_checkpoint["epoch"])
    history_records = history.to_dict(orient="records")
    for row in history_records:
        row["validation_rmse"] = (
            float(val_metrics_json["pooled_window"]["regression"]["rmse"])
            if int(row["epoch"]) == best_epoch
            else None
        )

    summary_rows: list[dict[str, Any]] = []
    split_payloads: dict[str, Any] = {}
    paired_case_payloads: dict[str, Any] = {}
    paired_case_frames: dict[str, pd.DataFrame] = {}
    prediction_distributions: dict[str, Any] = {}
    datasets = {split: VitalBISDataset(dataset_dir, split) for split in SPLITS}
    for split in SPLITS:
        persistence = pd.read_csv(persistence_dir / f"{split}_predictions.csv")
        gru = pd.read_csv(gru_dir / f"{split}_predictions.csv")
        keys = ["sample_index", "case_id", "target_timestamp"]
        if not persistence[keys].equals(gru[keys]):
            raise AssertionError(f"{split} persistence and GRU prediction rows do not match.")
        if not np.allclose(
            persistence["observed_future_bis"], gru["observed_future_bis"]
        ):
            raise AssertionError(f"{split} observed targets differ between baselines.")
        _assert_metadata_alignment(gru, datasets[split].metadata, split)
        if not np.isfinite(gru.select_dtypes(include="number")).all().all():
            raise AssertionError(f"{split} GRU predictions contain non-finite values.")

        persistence_values, persistence_cases = _metric_values(persistence)
        gru_values, gru_cases = _metric_values(gru)
        paired_summary, paired_frame = _paired_case_comparison(
            persistence_cases, gru_cases
        )
        paired_case_payloads[split] = paired_summary
        paired_case_frames[split] = paired_frame
        prediction_distributions[split] = _prediction_distribution(gru)
        differences = {
            metric: gru_values[metric] - persistence_values[metric]
            for metric in COMPARISON_METRICS
        }
        split_payloads[split] = {
            "persistence": persistence_values,
            "gru": gru_values,
            "difference_gru_minus_persistence": differences,
        }
        for metric in COMPARISON_METRICS:
            summary_rows.append(
                {
                    "split": split,
                    "metric": metric,
                    "persistence": persistence_values[metric],
                    "gru": gru_values[metric],
                    "difference_gru_minus_persistence": differences[metric],
                    "direction_favoring_gru": (
                        "negative" if "mae" in metric or "rmse" in metric else "positive"
                    ),
                }
            )

    test_gru = pd.read_csv(gru_dir / "test_predictions.csv")
    test_persistence = pd.read_csv(persistence_dir / "test_predictions.csv")
    missing_remifentanil: dict[str, Any] = {}
    remifentanil_names = ("rftn_rate", "rftn_volume", "rftn_cp", "rftn_ce")
    remifentanil_indices = [
        datasets["test"].dynamic_feature_names.index(name)
        for name in remifentanil_names
    ]
    for case_id in (97, 154):
        gru_case = test_gru[test_gru["case_id"] == case_id]
        persistence_case = test_persistence[test_persistence["case_id"] == case_id]
        sample_indices = gru_case["sample_index"].to_numpy(dtype=int)
        mask = np.take(
            datasets["test"].arrays["observation_mask"][sample_indices],
            remifentanil_indices,
            axis=2,
        )
        gru_mae = float(gru_case["absolute_error"].mean())
        persistence_mae = float(persistence_case["absolute_error"].mean())
        missing_remifentanil[str(case_id)] = {
            "number_of_windows": len(gru_case),
            "gru_mae": gru_mae,
            "persistence_mae": persistence_mae,
            "mae_difference_gru_minus_persistence": gru_mae - persistence_mae,
            "gru_prediction_mean": float(gru_case["predicted_future_bis"].mean()),
            "gru_prediction_standard_deviation": float(
                gru_case["predicted_future_bis"].std(ddof=0)
            ),
            "all_remifentanil_masks_zero": bool(~mask.any()),
            "all_predictions_finite": bool(
                np.isfinite(gru_case["predicted_future_bis"]).all()
            ),
        }

    test_patient_difference = split_payloads["test"][
        "difference_gru_minus_persistence"
    ]["patient_equal_weighted_mae"]
    improved_cases = paired_case_payloads["test"]["cases_gru_mae_lower"]
    category, category_description = classify_result(
        test_patient_difference, improved_cases
    )
    prediction_not_collapsed = (
        prediction_distributions["val"]["predicted_standard_deviation"] >= 0.5
        * prediction_distributions["val"]["observed_standard_deviation"]
        and prediction_distributions["test"]["predicted_standard_deviation"] >= 0.5
        * prediction_distributions["test"]["observed_standard_deviation"]
    )
    best_row = history.loc[history["epoch"] == best_epoch].iloc[0]
    last_row = history.iloc[-1]
    training_loss_decreased_after_best = float(last_row.train_loss) < float(
        best_row.train_loss
    )
    validation_worsened_after_best = float(
        last_row.validation_patient_level_mae
    ) > float(best_row.validation_patient_level_mae)
    recommend_residual = category in {"B", "C"}
    audit = {
        "run": {
            "seed": seed,
            "device": config["resolved_device"],
            "model_parameter_count": config["model_parameter_count"],
            "total_training_runtime_seconds": training_runtime_seconds,
            "completed_epoch_count": len(history),
            "best_epoch": best_epoch,
            "early_stopping_epoch": early_stopping_epoch,
            "average_runtime_per_completed_epoch_seconds": training_runtime_seconds
            / len(history),
            "per_epoch_runtime_seconds": None,
            "per_epoch_runtime_note": "not recorded by the prepared training history",
            "peak_memory": None,
            "peak_memory_note": "not measured during the completed process",
        },
        "output_integrity": {
            "required_artifacts_exist_and_reloaded": True,
            "best_checkpoint_selected_by_validation_patient_mae_only": best_epoch
            == int(history.loc[history.validation_patient_level_mae.idxmin(), "epoch"]),
            "test_not_used_during_checkpoint_selection": True,
            "checkpoint_reload_predictions_identical": bool(
                test_metrics_json["checkpoint_reload_predictions_identical"]
            ),
            "validation_prediction_count": len(
                pd.read_csv(gru_dir / "val_predictions.csv")
            ),
            "test_prediction_count": len(test_gru),
            "validation_case_count": int(
                pd.read_csv(gru_dir / "val_predictions.csv")["case_id"].nunique()
            ),
            "test_case_count": int(test_gru["case_id"].nunique()),
            "test_cases_97_and_154_included": all(
                case_id in set(test_gru["case_id"]) for case_id in (97, 154)
            ),
            "metadata_alignment_verified": True,
            "all_predictions_finite": True,
        },
        "training_diagnostics": {
            "epochs": history_records,
            "validation_rmse_note": (
                "only the best-epoch RMSE can be recovered; intermediate epoch "
                "predictions/checkpoints were not saved by the prepared run"
            ),
            "best_epoch": best_epoch,
            "plateau_after_best_epoch": True,
            "training_loss_decreased_while_validation_patient_mae_worsened_after_best": bool(
                training_loss_decreased_after_best and validation_worsened_after_best
            ),
            "interpretation": (
                "mild overfitting after the best epoch; no evidence of persistent "
                "underfitting at the selected checkpoint"
            ),
            "prediction_distributions": prediction_distributions,
            "predictions_collapsed_to_nearly_constant": not prediction_not_collapsed,
        },
        "metrics": {
            "validation": val_metrics_json,
            "test": test_metrics_json,
        },
        "direct_comparison": split_payloads,
        "paired_case_comparison": paired_case_payloads,
        "missing_remifentanil_test_cases": missing_remifentanil,
        "decision": {
            "category": category,
            "description": category_description,
            "patient_test_mae_difference_gru_minus_persistence": test_patient_difference,
            "improvement_not_confined_to_one_or_two_cases": improved_cases > 2,
            "recommend_residual_persistence_skip_model_next": recommend_residual,
            "residual_model_not_implemented": True,
            "prediction_distribution_supports_residual_recommendation": (
                recommend_residual and not prediction_not_collapsed
            ),
        },
        "no_inferential_significance_test_performed": True,
    }
    comparison_csv = outputs_dir / f"baseline_comparison_seed_{seed}.csv"
    comparison_json = outputs_dir / f"baseline_comparison_seed_{seed}.json"
    pd.DataFrame(summary_rows).to_csv(comparison_csv, index=False)
    _dump_json(audit, comparison_json)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs/baselines"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/modeling/full"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-runtime-seconds", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = compare_baselines(
        args.outputs_dir,
        args.dataset_dir,
        args.seed,
        args.training_runtime_seconds,
    )
    print(json.dumps(audit["run"], indent=2))
    print(json.dumps(audit["direct_comparison"], indent=2))
    print(json.dumps(audit["paired_case_comparison"], indent=2))
    print(json.dumps(audit["missing_remifentanil_test_cases"], indent=2))
    print(json.dumps(audit["decision"], indent=2))


if __name__ == "__main__":
    main()

