# Dayside Bz Prediction at 6 RE Using Gradient Boosting

Predicts the dayside magnetospheric Bz field at 6 Earth radii (RE) from solar wind observations at L1 (OMNI data base), using SWMF MHD simulation output as the training target. Separate models are trained for the current timestep (t0), 1-hour (t+60 min), and 2-hour (t+120 min) forecast horizons.

**Author:** TsigeA  
**Dataset:** 23 geomagnetic storm events, 2015–2019

---

## Repository layout

```
.
├── Generate_Tabulardata_OMNI_SWMF.py   # Step 1: build feature/target CSV per storm
├── gbm_swmf_bz_prediction.py           # Step 2: train and evaluate XGBoost models
├── OMNIfile_formater.py                # Helper: reformat raw OMNI downloads
├── plot_model_results.py               # Plots for trained model outputs
├── plot_stormevents_omniSWMF.py        # Plots for individual storm events
├── StormList_23events.csv              # Storm phase timestamps (23 events)
├── Storm_csvs/                         # Output of Step 1, one CSV per storm
├── omni_5min_*.txt                     # Raw OMNI 5-min solar wind data
└── z=0_var_2_e*Bz.txt                 # Raw SWMF Bz output (z=0 plane)
```

---

## Workflow

### Step 1 — Build training tables

```bash
python Generate_Tabulardata_OMNI_SWMF.py
```

For each storm event the script:
1. Reads OMNI 5-min data, replaces fill values, and linearly interpolates gaps.
2. Computes derived solar wind quantities (see Features below).
3. Reads SWMF Bz on the z=0 plane and extracts the mean dayside Bz at r = 6 ± 0.25 RE.
4. Restricts the time window to the overlap between OMNI and SWMF data.
5. Builds one row per SWMF timestamp containing 2 hours of lagged OMNI features and future Bz targets at t+60 min and t+120 min.
6. Saves the result to `Storm_csvs/<storm_id>.csv`.

Key settings (top of script):

| Setting | Default | Description |
|---|---|---|
| `HISTORY_HOURS` | 2 | Length of OMNI lag window |
| `OMNI_STEP_MIN` | 15 | Lag step size (matches SWMF cadence) |
| `HORIZONS_MIN` | [60, 120] | Forecast horizons in minutes |
| `TARGET_RADIUS_RE` | 6.0 | Target radial distance |
| `R_TOL` | 0.25 | Tolerance on radius selection |
| `DAYSIDE_ONLY` | True | Restrict to X > 0 (dayside) |
| `USE_GSM` | True | Use GSM coordinates for By/Bz |

### Step 2 — Train and evaluate

```bash
python gbm_swmf_bz_prediction.py
```

Loads all storm CSVs, performs a storm-wise train/val/test split, optionally tunes hyperparameters with Optuna, trains one XGBoost model per forecast horizon, and saves results to a timestamped `model_results_*/` folder.

Key settings (top of script):

| Setting | Default | Description |
|---|---|---|
| `TRAIN_FRAC / VAL_FRAC / TEST_FRAC` | 0.80 / 0.10 / 0.10 | Storm-wise split fractions |
| `TARGETS_TO_RUN` | ["tplus_60m", "tplus_120m"] | Which horizons to train |
| `USE_OPTUNA` | True | Bayesian hyperparameter search |
| `RUN_MANUAL_GRID_SEARCH` | False | Exhaustive grid search (slow) |

---

## Features

Each row contains **seasonal features** and **lagged OMNI variables** spanning the past 2 hours at 15-minute steps (lag_0m = current, lag_15m = 15 min ago, …, lag_120m = 2 hr ago).

| Feature | Description |
|---|---|
| `sin_doy`, `cos_doy` | Day-of-year encoded as sine/cosine |
| `Bmag_lag_*` | IMF total field magnitude (nT) |
| `BX_lag_*` | IMF Bx in GSM (nT) |
| `BY_used_lag_*` | IMF By in GSM (nT) |
| `BZ_used_lag_*` | IMF Bz in GSM (nT) |
| `V_lag_*` | Solar wind speed (km/s) |
| `Pdyn_lag_*` | Dynamic pressure (nPa) |
| `SYMH_lag_*` | SYM-H index (nT) |
| `Ey_lag_*` | Dawn-dusk electric field: −V × Bz × 10⁻³ (mV/m) |
| `Es_lag_*` | Dayside reconnection proxy: max(0, Ey) |
| `theta_lag_*` | IMF clock angle: arctan2(By, Bz) |
| `Newell_Coupling_lag_*` | Newell coupling: V^(4/3) B^(2/3) sin^(8/3)(θ/2) |

---

## Targets

| Column | Description |
|---|---|
| `target_Bz_dayside_6RE_t0` | SWMF dayside Bz at 6 RE, current time |
| `target_Bz_dayside_6RE_tplus_60m` | Same, 60 min ahead |
| `target_Bz_dayside_6RE_tplus_120m` | Same, 120 min ahead |

---

## Data splitting

Splitting is done **at the storm level** to prevent temporal leakage between training and evaluation sets. All rows from a given storm event belong to exactly one partition.

```
23 storms  →  ~18 train  |  ~2 val  |  ~3 test
```

The validation set is used for early stopping and (when `USE_OPTUNA=True`) hyperparameter tuning. The test set is only used for final performance reporting.

---

## Input file formats

**OMNI** (`omni_5min_*.txt`): space-separated, one row per 5-minute interval.
```
YYYY  DOY  HH  MM  Bmag  BX  BY_GSE  BZ_GSE  BY_GSM  BZ_GSM  V  Pdyn  SYMH
```
Fill values (99999.9, 9999.99, 999.99, 99.99) are replaced with NaN and interpolated.

**SWMF** (`z=0_var_2_e*Bz.txt`): space-separated.
```
YYYYMMDD  HHMMSS  X_RE  Y_RE  Bz
```

**Storm list** (`StormList_23events.csv`): one row per event with columns `prestorm_Stime`, `SSC_Stime`, `Main_Stime`, `Recov_Stime`, `Recov_endtime`.

---

## Outputs (`model_results_<timestamp>/`)

| File | Description |
|---|---|
| `model_<target>_<ts>.pkl` | Trained XGBoost model (joblib) |
| `test_predictions_<target>_<ts>.csv` | Per-row predictions on the test set |
| `validation_predictions_<target>_<ts>.csv` | Per-row predictions on the validation set |
| `gbm_multi_storm_summary<ts>.json` | Metrics, best hyperparameters, top features |
| `feature_importance_comparison_<ts>.png` | Top-20 feature gain comparison across targets |
| `grouped_feature_importance_comparison_<ts>.png` | Gain grouped by feature family |
| `*_testresults_<target>_<ts>.png` | Time-series prediction plots per test storm |

---

## Dependencies

```
numpy
pandas
xgboost
scikit-learn
optuna
matplotlib
joblib
```

Install with:
```bash
pip install numpy pandas xgboost scikit-learn optuna matplotlib joblib
```

---

## References

- Iong et al. (2022): *New Findings From Explainable SYM-H Forecasting Using Gradient Boosting Machines*, Space Weather. — Motivation for lagged SYM-H features and GBM architecture.
- Besliu-Ionescu & Mierla (2021): *Geoeffectiveness Prediction of CMEs* — Context for storm selection and geoeffectiveness criteria.
