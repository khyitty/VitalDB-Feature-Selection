# PK-PD assumptions and units

## Scope

This module is a research reconstruction of published propofol/remifentanil PK-PD
equations. It is not a medical device and must not be used for clinical dosing. It is
not externally validated.

The simulator generates drug compartment states and BIS only. It does **not** generate
HR, MBP, SBP, DBP, SpO2, ETCO2, or HRV. Those signals remain only in legacy
physiological-inclusive prediction artifacts because no source-supported transition
equations were found. The final prediction and RL candidate universe uses only the
shared simulator-compatible contract. Offline prediction Cp/Ce are reconstructed
causally from infusion histories and demographics with these same Schnider/Minto
equations. Orchestra CP/CE are device-reported TCI estimates used only for agreement
auditing, not direct measurements or canonical model inputs. CT target concentration
is never treated as Cp or Ce.

## Unit system

| Quantity | Public/internal unit | Notes |
|---|---|---|
| Time | seconds at API; minutes in PK matrices | `seconds / 60` is explicit |
| Propofol amount | mg | Central/peripheral state `x1-x3` |
| Propofol infusion | mg/min | Constant over an action-hold interval |
| Propofol Cp/Ce | mg/L | Numerically equal to microgram/mL |
| Remifentanil amount | microgram | Central/peripheral state `x1-x3` |
| Remifentanil infusion | microgram/min | Exogenous, never an RL action in this module |
| Remifentanil Cp/Ce | microgram/L | Numerically equal to ng/mL |
| Volume | L | Both drugs |
| Clearance | L/min | Both drugs |
| Micro-rate and ke0 | 1/min | Both drugs |
| BIS | unitless 0-100 scale | Raw and bounded values are both retained |

Different drug units are represented by explicit argument and field names. The code
does not assume that propofol mg and remifentanil micrograms are interchangeable.

## Timing and integration

- Default internal integration interval: 1 second.
- Planned control/action interval: 10 seconds.
- Propofol rate is held constant over `advance`.
- Remifentanil is an exogenous constant, piecewise-constant, callable, or CSV schedule.
- Production integration uses a float64 matrix-exponential exact zero-order hold for
  each constant-rate subinterval.
- Independent validation uses SciPy `solve_ivp` with tight tolerances.
- Values below `-1e-10` are treated as a numerical/physical failure. Values in
  `[-1e-10, 0)` are set to zero as round-off only; larger negatives are never hidden.

## Demographic policy

- Supported sex labels are `male` and `female`; this encoding selects the published
  sex-specific James LBM equation and is not a gender-identity model.
- Hard input bounds: age 18-90 years, height 120-220 cm, weight 35-200 kg.
- Warnings are emitted outside the combined source-study envelope: age 25-81 years,
  height 155-196 cm, or weight 44-123 kg.
- The code rejects non-finite values, non-positive LBM, LBM above total weight, or any
  demographic combination producing non-positive PK parameters.
- Synthetic validation patients are not claimed to represent the VitalDB population.

## BIS and stochastic policy

The deterministic response is Yun's additive interaction equation:

`BIS_raw = 98 * (1 + Ce_propofol/4.47 + Ce_remifentanil/19.3)^(-1.43)`.

`BIS_raw` is retained. The public noiseless BIS is bounded to `[0, 100]`; clipping is
reported and never used to conceal non-finite values.

Deterministic mode adds no noise and is seed-independent. Experimental stochastic mode
adds the Yun 2024 perturbation `epsilon ~ N(10, 0.4)`, interpreting the second normal
parameter as variance (`standard deviation = sqrt(0.4)`). Both the raw noisy value and
the bounded observed BIS are retained. This interpretation is source-ambiguous and is
not a clinical measurement-noise model. The paper's separate approximately 10%
effect-site drug-drop statement lacks a complete distribution/update equation and is
not implemented.

## Unsupported claims

- No actual VitalDB patient is dosed or simulated in this module.
- No paper figure is claimed to be reproduced exactly.
- No clinical safety, dosing, or efficacy claim is made.
- No Gymnasium API, PPO, actor, critic, or RL training is included.
