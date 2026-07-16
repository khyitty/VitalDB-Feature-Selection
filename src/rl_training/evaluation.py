"""Validation-only paired scenario evaluation for PPO checkpoints."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .attention_logging import save_attention_artifact
from .cohort import CohortBundle, EvaluationScenario
from .config import PPOConfig, PolicyCondition
from .environment_factory import make_cohort_environment
from .feature_extractors import FactorizedAttentionControlExtractor


def evaluate_scenarios(
    model: Any,
    *,
    condition: PolicyCondition,
    config: PPOConfig,
    cohort: CohortBundle,
    scenarios: Iterable[EvaluationScenario],
    training_seed: int,
    checkpoint_path: Path | None = None,
    attention_output_path: Path | None = None,
) -> pd.DataFrame:
    """Evaluate deterministic policy actions; reject held-out test scenarios."""

    scenario_list = tuple(scenarios)
    if not scenario_list:
        raise ValueError("At least one validation scenario is required.")
    if any(scenario.split != "validation" for scenario in scenario_list):
        raise ValueError("Module 6 evaluation is validation-only; test cohort remains sealed.")
    env = make_cohort_environment(
        condition=condition,
        ppo=config,
        cohort=cohort,
        split="validation",
        seed=training_seed,
        cycle=True,
    )
    rows: list[dict[str, Any]] = []
    feature_rows: list[np.ndarray] = []
    temporal_rows: list[np.ndarray] = []
    mask_rows: list[np.ndarray] = []
    attention_scenarios: list[str] = []
    attention_bis: list[float] = []
    inference_times: list[float] = []
    extractor = model.policy.features_extractor
    for scenario in scenario_list:
        observation, _ = env.reset(options={"scenario": scenario})
        done = False
        final_info: dict[str, Any] = {}
        total_return = 0.0
        normalized_clips = 0
        while not done:
            attention_mask = np.asarray(observation["history_mask"], dtype=bool).copy()
            attention_state_bis = float(observation["history"][-1, 0])
            started = time.perf_counter()
            action, _ = model.predict(observation, deterministic=True)
            inference_times.append(time.perf_counter() - started)
            observation, reward, terminated, truncated, final_info = env.step(action)
            total_return += float(reward)
            normalized_clips += int(final_info["normalized_clipping_applied"])
            done = terminated or truncated
            if condition == "attention_supported":
                if not isinstance(extractor, FactorizedAttentionControlExtractor):
                    raise TypeError("Attention condition does not use the explicit extractor.")
                attention = extractor.last_attention
                if attention is None:
                    raise RuntimeError("Attention extractor did not retain forward weights.")
                feature_rows.append(attention.feature_attention.cpu().numpy()[0])
                temporal_rows.append(attention.temporal_attention.cpu().numpy()[0])
                mask_rows.append(attention_mask)
                attention_scenarios.append(scenario.scenario_id)
                attention_bis.append(attention_state_bis)
        metrics = dict(final_info["episode_metrics"])
        rows.append(
            {
                "condition": condition,
                "training_seed": training_seed,
                "scenario_id": scenario.scenario_id,
                "patient_id": scenario.patient_id,
                "cohort_split": scenario.split,
                "scenario_seed": scenario.seed,
                "return": total_return,
                "normalized_clipping_count": normalized_clips,
                **metrics,
            }
        )
    env.close()
    frame = pd.DataFrame(rows)
    frame["mean_inference_seconds_per_action"] = float(np.mean(inference_times))
    if condition == "attention_supported" and attention_output_path is not None:
        if checkpoint_path is None:
            raise ValueError("Attention artifacts require an exact checkpoint path.")
        contract = model.policy.features_extractor
        assert isinstance(contract, FactorizedAttentionControlExtractor)
        save_attention_artifact(
            attention_output_path,
            feature_attention=np.stack(feature_rows),
            temporal_attention=np.stack(temporal_rows),
            history_mask=np.stack(mask_rows),
            feature_names=contract.feature_names,
            scenario_ids=np.asarray(attention_scenarios),
            bis=np.asarray(attention_bis),
            checkpoint_path=checkpoint_path,
        )
    return frame


def checkpoint_score(frame: pd.DataFrame) -> tuple[float, float, float]:
    """Prespecified lexicographic validation selection score."""

    if set(frame["cohort_split"]) != {"validation"}:
        raise ValueError("Checkpoint selection can use only validation scenarios.")
    return (
        float(frame["bis_target_mae"].mean()),
        -float(frame["fraction_time_in_bis_40_60"].mean()),
        float(frame["absolute_action_change_sum"].mean()),
    )
