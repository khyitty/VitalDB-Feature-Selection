"""Published additive propofol-remifentanil BIS response."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


BASELINE_BIS = 98.0
PROPOFOL_HALF_EFFECT_MG_PER_L = 4.47
REMIFENTANIL_HALF_EFFECT_MICROGRAMS_PER_L = 19.3
BIS_EXPONENT = 1.43
STOCHASTIC_NOISE_MEAN = 10.0
STOCHASTIC_NOISE_VARIANCE = 0.4


@dataclass(frozen=True)
class BISResponse:
    raw_noiseless_bis: float
    noiseless_bis: float
    noise_bis_units: float
    raw_observed_bis: float
    observed_bis: float
    clipping_applied: bool


def combined_bis_response(
    propofol_ce_mg_per_l: float,
    remifentanil_ce_micrograms_per_l: float,
) -> float:
    """Evaluate Yun's equation (32)/(6) before output bounding or noise."""

    values = (propofol_ce_mg_per_l, remifentanil_ce_micrograms_per_l)
    if not all(math.isfinite(value) and value >= 0.0 for value in values):
        raise ValueError("Effect-site concentrations must be finite and non-negative.")
    normalized = (
        1.0
        + propofol_ce_mg_per_l / PROPOFOL_HALF_EFFECT_MG_PER_L
        + remifentanil_ce_micrograms_per_l
        / REMIFENTANIL_HALF_EFFECT_MICROGRAMS_PER_L
    )
    return float(BASELINE_BIS * normalized ** (-BIS_EXPONENT))


def sample_bis_noise(rng: np.random.Generator) -> float:
    """Sample the experimental Yun 2024 additive perturbation."""

    return float(rng.normal(STOCHASTIC_NOISE_MEAN, math.sqrt(STOCHASTIC_NOISE_VARIANCE)))


def evaluate_bis(
    propofol_ce_mg_per_l: float,
    remifentanil_ce_micrograms_per_l: float,
    *,
    deterministic: bool,
    rng: np.random.Generator,
) -> BISResponse:
    raw_noiseless = combined_bis_response(
        propofol_ce_mg_per_l, remifentanil_ce_micrograms_per_l
    )
    noiseless = float(np.clip(raw_noiseless, 0.0, 100.0))
    noise = 0.0 if deterministic else sample_bis_noise(rng)
    raw_observed = raw_noiseless + noise
    if not math.isfinite(raw_observed):
        raise FloatingPointError("BIS response became non-finite.")
    observed = float(np.clip(raw_observed, 0.0, 100.0))
    return BISResponse(
        raw_noiseless_bis=raw_noiseless,
        noiseless_bis=noiseless,
        noise_bis_units=noise,
        raw_observed_bis=raw_observed,
        observed_bis=observed,
        clipping_applied=(noiseless != raw_noiseless or observed != raw_observed),
    )
