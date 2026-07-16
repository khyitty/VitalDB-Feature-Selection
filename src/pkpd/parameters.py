"""Published Schnider propofol and Minto remifentanil parameters."""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Mapping

from .demographics import PatientDemographics


SCHNIDER_CONSTANTS: Mapping[str, float] = MappingProxyType(
    {
        "h1": 4.27,
        "h2": 18.9,
        "h3": 0.391,
        "h4": 53.0,
        "h5": 238.0,
        "h6": 1.89,
        "h7": 0.0456,
        "h8": 77.0,
        "h9": 0.0681,
        "h10": 59.0,
        "h11": 0.0264,
        "h12": 177.0,
        "h13": 1.29,
        "h14": 0.024,
        "h15": 53.0,
        "h16": 0.836,
        "h17": 0.456,
    }
)

MINTO_CONSTANTS: Mapping[str, float] = MappingProxyType(
    {
        "f1": 5.1,
        "f2": 0.0201,
        "f3": 0.072,
        "f4": 9.82,
        "f5": 0.0811,
        "f6": 0.108,
        "f7": 5.42,
        "f8": 2.6,
        "f9": 0.0162,
        "f10": 0.0191,
        "f11": 2.05,
        "f12": 0.030,
        "f13": 0.076,
        "f14": 0.00113,
        "f15": 0.595,
        "f16": 0.007,
        "f17": 40.0,
        "f18": 55.0,
    }
)


@dataclass(frozen=True)
class DrugPKParameters:
    """Three-compartment and effect-site parameters with explicit units."""

    drug_name: str
    source_model: str
    v1_l: float
    v2_l: float
    v3_l: float
    cl1_l_per_min: float
    cl2_l_per_min: float
    cl3_l_per_min: float
    ke0_per_min: float
    amount_unit: str
    concentration_unit: str

    def __post_init__(self) -> None:
        numeric = {
            "v1_l": self.v1_l,
            "v2_l": self.v2_l,
            "v3_l": self.v3_l,
            "cl1_l_per_min": self.cl1_l_per_min,
            "cl2_l_per_min": self.cl2_l_per_min,
            "cl3_l_per_min": self.cl3_l_per_min,
            "ke0_per_min": self.ke0_per_min,
        }
        invalid = {
            name: value
            for name, value in numeric.items()
            if not math.isfinite(value) or value <= 0.0
        }
        if invalid:
            raise ValueError(
                f"{self.source_model} produced non-positive/non-finite parameters: {invalid}"
            )

    @property
    def volumes_l(self) -> tuple[float, float, float]:
        return self.v1_l, self.v2_l, self.v3_l

    @property
    def clearances_l_per_min(self) -> tuple[float, float, float]:
        return self.cl1_l_per_min, self.cl2_l_per_min, self.cl3_l_per_min

    @property
    def k10_per_min(self) -> float:
        return self.cl1_l_per_min / self.v1_l

    @property
    def k12_per_min(self) -> float:
        return self.cl2_l_per_min / self.v1_l

    @property
    def k13_per_min(self) -> float:
        return self.cl3_l_per_min / self.v1_l

    @property
    def k21_per_min(self) -> float:
        return self.cl2_l_per_min / self.v2_l

    @property
    def k31_per_min(self) -> float:
        return self.cl3_l_per_min / self.v3_l

    @property
    def micro_rate_constants_per_min(self) -> dict[str, float]:
        return {
            "k10": self.k10_per_min,
            "k12": self.k12_per_min,
            "k13": self.k13_per_min,
            "k21": self.k21_per_min,
            "k31": self.k31_per_min,
            "ke0": self.ke0_per_min,
        }

    @property
    def unit_metadata(self) -> dict[str, str]:
        return {
            "volume": "L",
            "clearance": "L/min",
            "micro_rate": "1/min",
            "amount": self.amount_unit,
            "concentration": self.concentration_unit,
            "infusion_rate": f"{self.amount_unit}/min",
        }


def schnider_propofol_parameters(patient: PatientDemographics) -> DrugPKParameters:
    """Calculate Schnider propofol parameters from Yun equations (6)-(17)."""

    h = SCHNIDER_CONSTANTS
    age = patient.age_years
    weight = patient.weight_kg
    height = patient.height_cm
    lbm = patient.lean_body_mass_kg
    return DrugPKParameters(
        drug_name="propofol",
        source_model="Schnider",
        v1_l=h["h1"],
        v2_l=h["h2"] - h["h3"] * (age - h["h4"]),
        v3_l=h["h5"],
        cl1_l_per_min=(
            h["h6"]
            + h["h7"] * (weight - h["h8"])
            - h["h9"] * (lbm - h["h10"])
            + h["h11"] * (height - h["h12"])
        ),
        cl2_l_per_min=h["h13"] - h["h14"] * (age - h["h15"]),
        cl3_l_per_min=h["h16"],
        ke0_per_min=h["h17"],
        amount_unit="mg",
        concentration_unit="mg/L",
    )


def minto_remifentanil_parameters(patient: PatientDemographics) -> DrugPKParameters:
    """Calculate Minto remifentanil parameters from Yun equations (18)-(29)."""

    f = MINTO_CONSTANTS
    age_delta = patient.age_years - f["f17"]
    lbm_delta = patient.lean_body_mass_kg - f["f18"]
    return DrugPKParameters(
        drug_name="remifentanil",
        source_model="Minto",
        v1_l=f["f1"] - f["f2"] * age_delta + f["f3"] * lbm_delta,
        v2_l=f["f4"] - f["f5"] * age_delta + f["f6"] * lbm_delta,
        v3_l=f["f7"],
        cl1_l_per_min=f["f8"] - f["f9"] * age_delta + f["f10"] * lbm_delta,
        cl2_l_per_min=f["f11"] - f["f12"] * age_delta,
        cl3_l_per_min=f["f13"] - f["f14"] * age_delta,
        ke0_per_min=f["f15"] - f["f16"] * age_delta,
        amount_unit="microgram",
        concentration_unit="microgram/L (ng/mL)",
    )
