# PK-PD equation traceability

This document records the equations used by the research-only anesthesia patient
simulator. PDF page numbers below are one-based PDF pages in the local files. Printed
page numbers are included where they differ.

## Sources inspected

1. W. J. Yun et al., "Deep reinforcement learning-based propofol infusion control
   for anesthesia: A feasibility study with a 3000-subject dataset," *Computers in
   Biology and Medicine* 156 (2023), 106739, DOI 10.1016/j.compbiomed.2023.106739.
   Local PDF inspected directly, including equations and Table A.4.
2. W. J. Yun et al., "Hierarchical Deep Reinforcement Learning-Based Propofol
   Infusion Assistant Framework in Anesthesia," *IEEE Transactions on Neural
   Networks and Learning Systems* 35(2) (2024), 2510-2520,
   DOI 10.1109/TNNLS.2022.3185345. Local PDF inspected directly.
3. T. W. Schnider et al., "The influence of method of administration and covariates
   on the pharmacokinetics of propofol in adult volunteers," *Anesthesiology* 88(5)
   (1998), 1170-1182, DOI 10.1097/00000542-199805000-00006.
4. C. F. Minto et al., "Influence of age and gender on the pharmacokinetics and
   pharmacodynamics of remifentanil. I. Model development," *Anesthesiology* 86(1)
   (1997), 10-23, DOI 10.1097/00000542-199701000-00004.

The two Yun PDFs are the directly inspected implementation sources. The Schnider and
Minto publications are the models cited by Yun; their bibliographic records and model
parameterization were cross-checked against PubMed and published model summaries.

## Equation mapping

| Quantity | Source location | Source units | Code units | Implementation |
|---|---|---|---|---|
| Lean body mass | Yun 2023 PDF p.3, eqs. (4)-(5); Yun 2024 PDF p.10, eq. (21); Schnider/Minto James LBM definition | kg, cm | kg, cm | `src/pkpd/demographics.py:calculate_lean_body_mass_kg` |
| Three-compartment mass balance | Yun 2023 PDF p.3, eqs. (1)-(3); Yun 2024 PDF pp.2-3, eqs. (1)-(3) | amount, min | drug-specific amount, min | `src/pkpd/compartment.py:system_matrix` |
| Propofol V1-V3 and Cl1-Cl3 | Yun 2023 PDF pp.3-4, eqs. (6)-(11); Table A.4, PDF p.9 | L, L/min | L, L/min | `src/pkpd/parameters.py:schnider_propofol_parameters` |
| Propofol micro-rates and ke0 | Yun 2023 PDF pp.3-4, eqs. (12)-(17) | 1/min | 1/min | `src/pkpd/parameters.py:DrugPKParameters` |
| Remifentanil V1-V3 and Cl1-Cl3 | Yun 2023 PDF p.4, eqs. (18)-(23); Table A.4, PDF p.9 | L, L/min | L, L/min | `src/pkpd/parameters.py:minto_remifentanil_parameters` |
| Remifentanil micro-rates and ke0 | Yun 2023 PDF p.4, eqs. (24)-(29) | 1/min | 1/min | `src/pkpd/parameters.py:DrugPKParameters` |
| Plasma concentration | Yun 2023 PDF p.4 following eq. (29), `Cp=x1/Vol1` | amount/L | propofol mg/L; remifentanil microgram/L | `src/pkpd/compartment.py:state_from_vector` |
| Effect-site dynamics | Yun 2023 PDF p.4, eq. (31); Yun 2024 PDF p.4 (printed p.2513), eq. (5) | concentration/min | concentration/min | fourth row of `src/pkpd/compartment.py:system_matrix` |
| Combined BIS | Yun 2023 PDF p.4, eq. (32); Yun 2024 PDF p.4 (printed p.2513), eq. (6) | propofol 4.47 mg/L; remifentanil 19.3 microgram/L | propofol mg/L; remifentanil microgram/L | `src/pkpd/bis_response.py:combined_bis_response` |
| Gaussian BIS perturbation | Yun 2024 PDF p.4 (printed p.2513), text following eq. (6) | `epsilon ~ N(10, 0.4)`; second parameter not explicitly named | BIS units | `src/pkpd/bis_response.py:sample_bis_noise` |
| Infusion conversion | Yun 2023 PDF p.3 states `u(t)` in mg/min; Minto concentration scale requires microgram/min for remifentanil | mg/min or microgram/min | amount/min internally, seconds at public API boundary | `src/pkpd/units.py` |

## Parameter constants

The implemented values are transcribed from Yun 2023 Table A.4 (PDF p.9) and Yun
2024 Table IV (PDF p.10, printed p.2519).

- Schnider propofol: `h1=4.27`, `h2=18.9`, `h3=0.391`, `h4=53`,
  `h5=238`, `h6=1.89`, `h7=0.0456`, `h8=77`, `h9=0.0681`,
  `h10=59`, `h11=0.0264`, `h12=177`, `h13=1.29`, `h14=0.024`,
  `h15=53`, `h16=0.836`, `h17=0.456`.
- Minto remifentanil: `f1=5.1`, `f2=0.0201`, `f3=0.072`, `f4=9.82`,
  `f5=0.0811`, `f6=0.108`, `f7=5.42`, `f8=2.6`, `f9=0.0162`,
  `f10=0.0191`, `f11=2.05`, `f12=0.030`, `f13=0.076`,
  `f14=0.00113`, `f15=0.595`, `f16=0.007`, `f17=40`, `f18=55`.

## Source defects and resolutions

1. Both Yun PDFs print the James LBM term as `weight/height` without the square.
   This is dimensionally inconsistent and produces an implausible reference LBM far
   from the centering constants 59 kg and 55 kg. The code uses the source-model James
   equation `(weight/height)^2`, and the registry records the Yun typography conflict.
2. Both Yun PDFs print `h18` in the remifentanil `Cl1` LBM term, but no `h18` exists.
   The surrounding Minto equations, Table A.4/Table IV, and the Minto model identify
   this as `f18=55`; the code uses `f18` and records the correction.
3. Yun 2023 calls all compartment quantities mg and both inputs mg/min. That cannot
   be combined numerically with the remifentanil BIS half-effect scale of 19.3
   microgram/L (ng/mL). The code therefore uses mg for propofol and micrograms for
   remifentanil. The ODE is unchanged; only the coherent amount unit differs by drug.
4. Yun 2024 states both additive Gaussian noise and an approximately 10% random
   effect-site drug drop but does not provide a complete distribution or update rule
   for the drop. The experimental stochastic mode implements only the stated additive
   BIS noise. Effect-site drop is not implemented.
5. Yun Table A.4/Table IV report `f12=0.030`, while common transcriptions of the
   original Minto parameter use `0.0301`. The reconstruction uses the directly
   inspected Yun table value `0.030`; the difference is retained as source ambiguity
   rather than silently mixing parameter sets.

## Reproduction claim

The simulator reconstructs the published equations and verifies numerical and
qualitative behavior. The papers do not disclose all initial conditions, sampled
patient demographics, remifentanil trajectories, or random draws used in their
figures. Consequently, the validation report must not claim exact paper-figure
reproduction or external/clinical validation.
