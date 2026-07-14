# VitalDB Future-BIS Modeling Dataset

This repository separates raw data preparation from modeling-data construction:

- `main.py` downloads VitalDB tracks and performs initial signal-quality filtering,
  limited within-case forward filling, propofol-period cropping, and clinical-data merging.
- `scripts/build_prediction_dataset.py` reads the cleaned CSV, creates patient-level
  splits, resamples each case independently, fits preprocessing on training cases only,
  and constructs future-BIS prediction windows.

The default pilot uses 10-second samples. A 60-second history contains six observations
at `t-50, t-40, t-30, t-20, t-10, t`; the regression target is raw BIS exactly at
`t+30`. Classification labels indicate whether that same future BIS is above 60 or
below 40. Case IDs are split before any modeling windows are generated, so no case is
shared by train, validation, and test.

The static sex input uses the cleaned dataset's existing `sex_male` encoding
(`0` = female, `1` = male). It is imputed consistently if needed but is not standardized.

## Commands

Build the first 10 eligible cases:

```powershell
python scripts/build_prediction_dataset.py --pilot
```

Build all eligible cases (not run as part of the initial pipeline task):

```powershell
python scripts/build_prediction_dataset.py --full
```

Run the synthetic test suite:

```powershell
python -m pytest -q
```

Pilot arrays are saved to `data/modeling/pilot/{train,val,test}.npz`; matching window
metadata, split case lists, feature/preprocessing metadata, and the dataset report are
saved in the same directory.
