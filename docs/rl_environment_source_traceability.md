# RL Environment Source Traceability

This document separates source-confirmed definitions from repository design
choices for the research-only propofol control environment.

## Primary source

W. J. Yun et al., "Deep reinforcement learning-based propofol infusion control
for anesthesia," *Computers in Biology and Medicine*, 156 (2023), 106739.
The page numbers below are PDF page numbers in the local source copy.

| Item | Source location | Source-confirmed definition | Repository implementation | Status / ambiguity |
|---|---|---|---|---|
| BIS target and range | PDF p.4, after Eq. (32) | Typical target 50 and range 40--60 | Config defaults are 50 and 40--60 | Confirmed |
| Dynamic state | PDF p.5, Eqs. (36)--(39) | BIS history, BIS slope/error, propofol history/recent cumulative dose, remifentanil history/recent cumulative dose | `original_reconstructed` exposes the same seven ordered concepts | Reconstructed concepts; unpublished source code and online LOWESS details are unavailable |
| Demographics | PDF p.5, State Definition | Age, gender, weight, height are separate covariates | Static vector order is age, male indicator, height, weight | Confirmed concepts; sex encoding is repository metadata |
| BIS smoothing | PDF p.5, State Definition | LOWESS-smoothed BIS is used | Online environment uses raw causal BIS; it never applies offline/noncausal LOWESS | LOWESS span, edge handling, and online causal procedure are not reported |
| BIS slope | PDF p.5, Eq. (36) | Current smoothed BIS minus previous smoothed BIS | Current raw BIS minus previous raw BIS per 10-second decision | Formula confirmed; raw causal substitution is a repository choice |
| BIS error | PDF p.5, Eq. (37) | Current smoothed BIS minus target BIS | Current raw BIS minus current target | Formula confirmed; raw causal substitution is a repository choice |
| Recent dose window | PDF p.5, Eqs. (38)--(39) | Sum infusion history over window `W` | Cumulative delivered dose over the configured 60-second causal history | `W` and exact sampling/unit convention are not numerically reported |
| Action timing | PDF p.5, Action Definition; PDF p.8 Table 3 | A continuous propofol action is selected every 10 seconds | One action is held for exactly 10 seconds; simulator integrates internally at 1 second | Confirmed |
| Action bound | PDF p.5, Action Definition | `0 <= a_t <= 27.7`; surrounding text describes milligrams delivered over 10 seconds | API unit is mg/min, so the source bound is converted to `0 <= rate <= 166.2 mg/min` | Conversion is `27.7 mg / (10/60 min)`; paper alternates "rate" and 10-second dose wording |
| Remifentanil | PDF p.4 and p.6 Algorithm 1 | Exogenous history is sampled/replayed and is not the agent action | A fixed schedule supplies remifentanil identically across state profiles | Confirmed role; repository schedule selection is explicit |
| Reward | PDF p.5, Eq. (40) | `1 / (abs(target - BIS(t+1)) + alpha)` | Available only as `paper_yun2023_parameterized` with an explicit positive alpha | The paper does not report alpha, so this is not labeled an exact reproduction |
| Default reward | Repository design | Not applicable | `transparent_tracking_v1` uses normalized absolute target error, explicit outside-range penalties, and disabled-by-default action/concentration terms | Repository design, fixed across state profiles |
| Episode duration | Repository design | No single training episode duration is specified; long-horizon examples are reported | Fixed duration must be a multiple of 10 seconds and ends with `truncated=True` | Repository design |
| Initial history | Repository design | Fig. 6 notes 150 seconds of initial input but does not define a simulator reset rule | Repeat the initial snapshot and expose a validity mask with only the newest row valid | Repository design; no hidden warm-up or fabricated past |

## Observable state boundary

The environment exposes only BIS, infusion/dose history, Cp/Ce, target-derived
quantities, and demographics produced by the simulator or causal history. It does
not fabricate HR, blood pressure, SpO2, ETCO2, HRV, BIS SQI, or infusion volume.
The internal compartment amounts `x1`, `x2`, and `x3` remain simulator internals
and are not part of any ordinary observation profile.

## Safety statement

This environment is a research-only reconstruction around a published PK-PD
simulator. It is not a medical device and must not be used for clinical dosing or
patient care.
