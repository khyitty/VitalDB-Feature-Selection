"""Patient demographic inputs for the published Schnider/Minto models."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Literal, Mapping
import warnings


Sex = Literal["male", "female"]

HARD_AGE_RANGE_YEARS = (18.0, 90.0)
HARD_HEIGHT_RANGE_CM = (120.0, 220.0)
HARD_WEIGHT_RANGE_KG = (35.0, 200.0)
SOURCE_AGE_RANGE_YEARS = (25.0, 81.0)
SOURCE_HEIGHT_RANGE_CM = (155.0, 196.0)
SOURCE_WEIGHT_RANGE_KG = (44.0, 123.0)


def calculate_lean_body_mass_kg(
    *, sex: Sex, height_cm: float, weight_kg: float
) -> float:
    """Calculate James lean body mass used by Schnider and Minto.

    The Yun papers omit the square in typesetting. The squared ratio is the
    dimensionally coherent source-model definition; see the traceability document.
    """

    ratio_squared = (float(weight_kg) / float(height_cm)) ** 2
    if sex == "male":
        return 1.10 * float(weight_kg) - 128.0 * ratio_squared
    if sex == "female":
        return 1.07 * float(weight_kg) - 148.0 * ratio_squared
    raise ValueError("sex must be exactly 'male' or 'female'.")


def _validate_bounded(name: str, value: float, bounds: tuple[float, float]) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite; received {value!r}.")
    if not bounds[0] <= value <= bounds[1]:
        raise ValueError(f"{name} must be within {bounds}; received {value}.")


@dataclass(frozen=True)
class PatientDemographics:
    """Validated adult covariates used to parameterize both drug models."""

    age_years: float
    sex: Sex
    height_cm: float
    weight_kg: float
    lean_body_mass_kg: float = field(init=False)

    def __post_init__(self) -> None:
        age = float(self.age_years)
        height = float(self.height_cm)
        weight = float(self.weight_kg)
        if self.sex not in ("male", "female"):
            raise ValueError("sex must be exactly 'male' or 'female'.")
        _validate_bounded("age_years", age, HARD_AGE_RANGE_YEARS)
        _validate_bounded("height_cm", height, HARD_HEIGHT_RANGE_CM)
        _validate_bounded("weight_kg", weight, HARD_WEIGHT_RANGE_KG)
        lbm = calculate_lean_body_mass_kg(
            sex=self.sex, height_cm=height, weight_kg=weight
        )
        if not math.isfinite(lbm) or lbm <= 0.0 or lbm > weight:
            raise ValueError(
                "Demographics produce an invalid James lean body mass: "
                f"LBM={lbm}, weight={weight}."
            )
        object.__setattr__(self, "age_years", age)
        object.__setattr__(self, "height_cm", height)
        object.__setattr__(self, "weight_kg", weight)
        object.__setattr__(self, "lean_body_mass_kg", lbm)

        source_values = (
            ("age_years", age, SOURCE_AGE_RANGE_YEARS),
            ("height_cm", height, SOURCE_HEIGHT_RANGE_CM),
            ("weight_kg", weight, SOURCE_WEIGHT_RANGE_KG),
        )
        outside = [name for name, value, bounds in source_values if not bounds[0] <= value <= bounds[1]]
        if outside:
            warnings.warn(
                "Demographics are within simulator hard bounds but outside the "
                f"combined source-study envelope for: {outside}.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PatientDemographics":
        """Prepare for a later VitalDB-demographics adapter without loading a cohort."""

        return cls(
            age_years=float(values["age_years"]),
            sex=str(values["sex"]).lower(),  # type: ignore[arg-type]
            height_cm=float(values["height_cm"]),
            weight_kg=float(values["weight_kg"]),
        )

    def as_dict(self) -> dict[str, float | str]:
        return {
            "age_years": self.age_years,
            "sex": self.sex,
            "height_cm": self.height_cm,
            "weight_kg": self.weight_kg,
            "lean_body_mass_kg": self.lean_body_mass_kg,
        }
