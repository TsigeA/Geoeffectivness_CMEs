# varx_swmf_dayside_bz.py
'''
Aim: to predict the storm time magnetospheric dynamics using Vector Autoregressive
models with exogenous input (VARX) https://github.com/lcparra/varx/tree/main

The model is fit jointly on all 4 inner-magnetosphere regions
(dayside dawn/dusk, nightside dawn/dusk) as a single 4-dimensional VARX system:

    Y(t) = A*Y(t-1) + B*X(t) + e(t)

where Y(t) is the [dayside_dawn, dayside_dusk, nightside_dawn, nightside_dusk] Bz-like
state vector and X(t) are the solar wind driver parameters from OMNI. This lets the
model capture interactions between the 4 regions (through A) in addition to the solar
wind driving (through B), rather than fitting 4 independent AR models.

Target variable: Bz_dayside_dawn, Bz_dayside_dusk, Bz_nightside_dawn, Bz_nightside_dusk,
1 hr and 2 hr ahead.
Training and testing data: SWMF simulation results for 4 regions (dawn, dusk, nightside
dawn, nightside dusk) and Solar wind data from OMNI database.
    - The data is split storm-wise into training, validation, and test sets.
    - exogenous inputs are the OMNI solar wind parameters at their current (lag_0m)
      sample; varx() builds the required lag history of these internally via nb, so
      the pre-lagged feature columns used by other scripts in this repo are not needed
      here.

Since varx() is a one-step-ahead transition model (native cadence here is 15 min,
confirmed from the data), 1h/2h-ahead forecasts are produced by recursively iterating
the fitted model forward, feeding back its own predictions for the AR part while using
the (assumed known ahead-of-time) solar wind exogenous inputs for the MA part. Forecast
uncertainty is quantified with a Monte Carlo simulation: Gaussian noise drawn from the
model's per-channel residual variance is injected at every recursion step and propagated
through the fitted A/B dynamics, giving a distribution of forecast paths per horizon.

Granger analysis is performed on the fitted model to quantify the significance of each lag block or exogenous input, 
producing p-values for significance testing.

L2 regularization (ridge shrinkage) is applied to the VARX coefficients to reduce overfitting, controlled by the GAMMA parameter.


@date: Jul 5, 2026
'''
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import varxModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ----------------------------
# User settings
# ----------------------------
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "Storm_csv_4regions2"
FILE_GLOB = "*.csv"

DATETIME_COL = "datetime"

# The 4 inner-magnetosphere regions modeled jointly as the VARX output vector Y(t).
# target_Bz_{region}_t0 is the current-time value (identical to the *_lag_0m columns
# used elsewhere in this repo), used here as the endogenous state.
# REGIONS = ["dayside_dawn", "dayside_dusk", "nightside_dawn", "nightside_dusk"]
REGIONS = ["dayside", "duskside", "dawnside", "nightside"]
ENDOG_COLS = [f"target_Bz_{r}_t0" for r in REGIONS]
TARGETS_1H = [f"target_Bz_{r}_tplus_60m" for r in REGIONS]
TARGETS_2H = [f"target_Bz_{r}_tplus_120m" for r in REGIONS]

# OMNI solar wind drivers used as the exogenous input X(t). varx() builds the lag
# history internally (see MA_ORDER below), so only the current sample is needed here.
SW_PARAMS = ["Bmag", "BX", "BY_used", "BZ_used", "V", "Pdyn", "SYMH", "Ey", "Es",
             "theta", "Newell_Coupling"]
# EXOG_COLS = [f"{p}_lag_0m" for p in SW_PARAMS]
EXOG_COLS = ["Bmag","SYMH","Pdyn","Ey","Es"]
# storms are sampled every 15 minutes, confirmed from the data
FORECAST_STEPS = {
    "1h": 4,   # 4 x 15 min = 60 min
    "2h": 8,   # 8 x 15 min = 120 min
}

AR_ORDER = 1     # na: lags of Y (1 hour of AR history across the 4 regions)
MA_ORDER = 1     # nb: lags of X (2 hours of solar-wind history)
GAMMA = 1.0      # ridge shrinkage passed to varx() (docstring caps this at ~1)

N_SIMS = 300     # Monte Carlo forecast paths used to quantify uncertainty
CI_PCT = (5, 95)  # percentile band plotted around the point forecast

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10
RANDOM_SEED = 42

RNG = np.random.default_rng(RANDOM_SEED)


# ----------------------------
# Load data
# ----------------------------
def load_all_storms(data_dir):
    paths = sorted(Path(data_dir).glob(FILE_GLOB))
    if len(paths) == 0:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    all_dfs = []

    for path in paths:
        df = pd.read_csv(path)
        df[DATETIME_COL] = pd.to_datetime(df[DATETIME_COL])
        df = df.sort_values(DATETIME_COL).reset_index(drop=True)
        df["storm_id"] = path.stem
        df["source_file"] = path.name
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# ----------------------------
# Storm-wise split
# ----------------------------
def split_storms(df):
    storm_ids = sorted(df["storm_id"].unique())

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(storm_ids)

    n = len(storm_ids)
    n_train = int(round(TRAIN_FRAC * n))
    n_val = max(1, int(round(VAL_FRAC * n)))
    n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        n_train -= 1

    train_ids = storm_ids[:n_train]
    val_ids = storm_ids[n_train:n_train + n_val]
    test_ids = storm_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def subset(df, ids):
    return df[df["storm_id"].isin(ids)].copy().reset_index(drop=True)


# ----------------------------
# Metrics
# ----------------------------
def metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ----------------------------
# Per-storm array preparation
# ----------------------------
def storm_arrays(storm_df):
    '''Time-ordered endogenous/exogenous arrays for one storm. NaNs are kept as-is;
    varx()/myxcorr() exclude them internally rather than us dropping rows, since
    dropping rows would break the fixed 15-min lag structure the model relies on.'''
    d = storm_df.sort_values(DATETIME_COL).reset_index(drop=True)
    Y = d[ENDOG_COLS].to_numpy(dtype=float)
    X = d[EXOG_COLS].to_numpy(dtype=float)
    t = d[DATETIME_COL].to_numpy()
    return d, Y, X, t


def fit_standardizer(arrays):
    stacked = np.concatenate(arrays, axis=0)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize(arr, mean, std):
    return (arr - mean) / std


def destandardize(arr, mean, std):
    return arr * std + mean


# ----------------------------
# Recursive multi-step forecast with Monte Carlo uncertainty
# ----------------------------
def varx_forecast(model, Y_std, X_std, anchor, steps, n_sims, rng):
    '''
    Recursively forecast `steps` samples past `anchor` using a fitted VARX model:
        Y(t) = sum_k A[k] @ Y(t-1-k) + sum_j B[j] @ X(t-j) + e(t)
    Y_std, X_std: full standardized storm-length arrays (time x dim).
    anchor: index of the last observed sample.
    Returns sims, shape (n_sims, steps, ydim); sims[0] is the noise-free point forecast,
    the remaining paths inject Gaussian noise (from the model's residual variance) at
    every step and propagate it through the fitted dynamics.
    '''
    A, B = model["A"], model["B"]
    na, ydim, _ = A.shape
    nb = B.shape[0]
    sigma = np.sqrt(np.maximum(model["s2"], 0))

    # y_hist[k] holds Y(anchor+h-k) for every simulated path, shape (n_sims, ydim)
    y_hist = [np.tile(Y_std[anchor - k], (n_sims, 1)) for k in range(na)]

    noise = rng.normal(0.0, sigma, size=(n_sims, steps, ydim))
    noise[0] = 0.0  # path 0 stays noise-free -> the point forecast

    sims = np.empty((n_sims, steps, ydim))
    for h in range(steps):
        y_hat = np.zeros((n_sims, ydim))
        for k in range(na):
            y_hat += y_hist[k] @ A[k].T
        for j in range(nb):
            y_hat += X_std[anchor + h + 1 - j] @ B[j].T
        y_hat = y_hat + noise[:, h, :]
        sims[:, h, :] = y_hat
        y_hist = [y_hat] + y_hist[:-1]

    return sims


def rolling_forecast_storm(model, Y_std, X_std, t, na, nb, n_sims, rng):
    '''Slide a forecast anchor across one storm, producing Monte Carlo forecast
    ensembles at every horizon in FORECAST_STEPS. Anchors whose required history/
    future window contains NaNs are skipped.'''
    n = len(Y_std)
    max_steps = max(FORECAST_STEPS.values())
    records = {label: [] for label in FORECAST_STEPS}

    for anchor in range(na - 1, n - max_steps):
        lo_y = anchor - na + 1
        lo_x = anchor - nb + 2
        hi_x = anchor + max_steps
        if lo_x < 0 or hi_x >= n:
            continue
        if not np.isfinite(Y_std[lo_y:anchor + 1]).all():
            continue
        if not np.isfinite(X_std[lo_x:hi_x + 1]).all():
            continue

        sims = varx_forecast(model, Y_std, X_std, anchor, max_steps, n_sims, rng)

        for label, steps in FORECAST_STEPS.items():
            true_idx = anchor + steps
            if not np.isfinite(Y_std[true_idx]).all():
                continue
            records[label].append({
                "time": t[true_idx],
                "y_true_std": Y_std[true_idx],
                "sims_std": sims[:, steps - 1, :],
            })

    return records


# ----------------------------
# Plotting
# ----------------------------
def plot_storm_forecast(pred_df, storm_id, label, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, region in zip(axes.ravel(), REGIONS):
        d = pred_df[pred_df["region"] == region].sort_values("time")
        # if d.empty:
        #     continue
        # ax.fill_between(d["time"], d["ci_lo"], d["ci_hi"], color="tab:orange",
        #                  alpha=0.25, label=f"{CI_PCT[1] - CI_PCT[0]}% CI")
        ax.plot(d["time"], d["y_pred"], color="tab:orange", label="predicted")
        ax.plot(d["time"], d["y_true"], color="tab:blue", label="true")
        ax.set_title(region)
        ax.tick_params(axis="x", rotation=30)

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"{storm_id} - {label} ahead forecast")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"forecast_{storm_id}_{label}_{current_time}.png"), dpi=150)
    plt.close(fig)


def plot_scatter_summary(all_preds, label, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, region in zip(axes.ravel(), REGIONS):
        d = all_preds[all_preds["region"] == region]
        if d.empty:
            continue
        yerr = np.vstack([d["y_pred"] - d["ci_lo"], d["ci_hi"] - d["y_pred"]])
        ax.errorbar(d["y_true"], d["y_pred"], yerr=yerr, fmt="o", ms=3,
                     alpha=0.4, ecolor="tab:orange", elinewidth=0.5, color="tab:blue")
        lims = [min(d["y_true"].min(), d["y_pred"].min()),
                max(d["y_true"].max(), d["y_pred"].max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_title(region)
        ax.set_xlabel("true")
        ax.set_ylabel("predicted")

    fig.suptitle(f"{label} ahead forecast (test storms), error bars = {CI_PCT[1] - CI_PCT[0]}% CI")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"scatter_summary_{label}_{current_time}.png"), dpi=150)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    df = load_all_storms(DATA_DIR)
    output_dir = os.path.join("VARX_results", f"VARX_results_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    print("Number of storms:", df["storm_id"].nunique())
    print("Regions modeled jointly:", REGIONS)
    print("AR order (na):", AR_ORDER, "  MA order (nb):", MA_ORDER)
    print("Exogenous features:", EXOG_COLS)

    train_ids, val_ids, test_ids = split_storms(df)
    print("\nStorm split:")
    print("Train:", train_ids)
    print("Val  :", val_ids)
    print("Test :", test_ids)

    train_df = subset(df, train_ids)
    test_df = subset(df, test_ids)

    # Build per-storm arrays (list of records, one per storm, as varx() expects)
    train_storms = [storm_arrays(g) for _, g in train_df.groupby("storm_id")]
    test_storms = {sid: storm_arrays(g) for sid, g in test_df.groupby("storm_id")}

    Y_mean, Y_std_ = fit_standardizer([Y for _, Y, _, _ in train_storms])
    X_mean, X_std_ = fit_standardizer([X for _, _, X, _ in train_storms])

    Y_train_list = [standardize(Y, Y_mean, Y_std_) for _, Y, _, _ in train_storms]
    X_train_list = [standardize(X, X_mean, X_std_) for _, _, X, _ in train_storms]

    print("\nFitting VARX model on", len(Y_train_list), "training storms...")
    model = varxModel.varx(Y_train_list, AR_ORDER, X=X_train_list, nb=MA_ORDER, gamma=GAMMA)
    print("Residual variance (s2) per region (standardized units):", model["s2"])
    print("A p-values (per output lag-block):\n", model["A_pval"])
    print("B p-values (per exogenous lag-block):\n", model["B_pval"])

    all_outputs = {}
    for label in FORECAST_STEPS:
        all_preds = []

        for storm_id, (_, Y, X, t) in test_storms.items():
            Y_s = standardize(Y, Y_mean, Y_std_)
            X_s = standardize(X, X_mean, X_std_)

            records = rolling_forecast_storm(model, Y_s, X_s, t, AR_ORDER, MA_ORDER,
                                              N_SIMS, RNG)

            rows = []
            for rec in records[label]:
                y_true = destandardize(rec["y_true_std"], Y_mean, Y_std_)
                sims = destandardize(rec["sims_std"], Y_mean, Y_std_)  # (n_sims, ydim)
                y_pred = sims[0]
                lo, hi = np.percentile(sims, CI_PCT, axis=0)
                for i, region in enumerate(REGIONS):
                    rows.append({
                        "time": rec["time"], "storm_id": storm_id, "region": region,
                        "y_true": y_true[i], "y_pred": y_pred[i],
                        "ci_lo": lo[i], "ci_hi": hi[i],
                    })

            pred_df = pd.DataFrame(rows)
            if pred_df.empty:
                continue
            plot_storm_forecast(pred_df, storm_id, label, output_dir)
            all_preds.append(pred_df)

        all_preds = pd.concat(all_preds, ignore_index=True)
        plot_scatter_summary(all_preds, label, output_dir)

        outname = f"varx_predictions_{label}_{current_time}.csv"
        all_preds.to_csv(os.path.join(output_dir, outname), index=False)
        print("Saved:", os.path.join(output_dir, outname))

        region_scores = {
            region: metrics(all_preds.loc[all_preds["region"] == region, "y_true"],
                             all_preds.loc[all_preds["region"] == region, "y_pred"])
            for region in REGIONS
        }
        print(f"\n{label} forecast metrics:")
        for region, score in region_scores.items():
            print(f"  {region}: " + ", ".join(f"{k}={v:.4f}" for k, v in score.items()))

        all_outputs[label] = {"metrics": region_scores, "output_file": str(os.path.join(output_dir, outname))}

    summary = {
        "train_storms": list(train_ids),
        "validation_storms": list(val_ids),
        "test_storms": list(test_ids),
        "regions": REGIONS,
        "exogenous_features": EXOG_COLS,
        "forecast_results": all_outputs,
        "model": f"varx(na={AR_ORDER}, nb={MA_ORDER}, gamma={GAMMA})",
    }
    pd.Series(summary).to_json(os.path.join(output_dir, f"varx_summary_{current_time}.json"), indent=2)
    print("\nSaved summary: varx_summary.json")


if __name__ == "__main__":
    main()
