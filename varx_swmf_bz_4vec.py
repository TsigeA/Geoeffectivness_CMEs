# varx_swmf_bz_4vec.py
'''
Aim: to predict the storm time magnetospheric dynamics at 6 RE dayside, nightside, duskside, dawnside
jointly using a Vector Autoregressive model with eXogenous input (VARX), via statsmodels VAR:
https://www.statsmodels.org/devel/generated/statsmodels.tsa.vector_ar.var_model.VAR.html
class statsmodels.tsa.vector_ar.var_model.VAR(endog, exog=None, dates=None, freq=None, missing='none')

The 4 regions are modeled jointly as a single 4-dimensional endogenous vector Y(t) =
[Bz_dayside, Bz_duskside, Bz_dawnside, Bz_nightside], with OMNI solar wind parameters as the
contemporaneous exogenous input X(t):

    Y(t) = c + sum_{k=1..p} A_k Y(t-k) + B X(t) + e(t)

This mirrors the methodology used in AR_swmf_dayside_bz.py for the single-target ARX case:
 - The data is split storm-wise into a train+val (CV) pool and a held-out test pool.
 - All exogenous features are standardized using training-set statistics.
 - The VAR lag order p is selected via storm-wise k-fold cross-validation, minimizing the
   rolling multi-step forecast RMSE (pooled across the 4 regions) on held-out validation storms.
 - The final model is refit on the full train+val pool at the selected order.
 - Rolling forecast on the test storms applies the fitted model's coefficients to each storm
   history window via VARResults.forecast(y=..., exog_future=...), which -- unlike AutoReg --
   does not require re-instantiating/re-fitting a model per window.

@author: TsigeA
@date: Jul 17, 2026
'''
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


# ----------------------------
# User settings
# ----------------------------
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "Storm_csvs_4regions2"
FILE_GLOB = "*.csv"

DATETIME_COL = "datetime"

# The 4 inner-magnetosphere regions modeled jointly as the VARX output vector Y(t).
REGIONS = ["nightside","duskside"] # take out  "nightside" , "duskside" to see how the model performs
ENDOG_COLS = [f"target_Bz_{r}_t0" for r in REGIONS]
TARGETS_1H = [f"target_Bz_{r}_tplus_60m" for r in REGIONS]
TARGETS_2H = [f"target_Bz_{r}_tplus_120m" for r in REGIONS]

# OMNI solar wind drivers used as the exogenous input X(t). statsmodels VAR's exog is
# contemporaneous only (no automatic lagging), same convention as AutoReg's exog in
# AR_swmf_dayside_bz.py.
SW_PARAMS = ["Bmag", "BX", "BY_used", "BZ_used", "V", "Pdyn", "SYMH", "Ey", "Es",
             "theta", "Newell_Coupling"]
EXOG_COLS = [ "Bmag","Pdyn","SYMH", "Es","Ey"] 

# storms are sampled every 15 minutes, confirmed from the data
FORECAST_STEPS = {
    "1h": 4,   # 4 x 15 min = 60 min
    "2h": 8,   # 8 x 15 min = 120 min
}

VAR_ORDER_CANDIDATES = list(range(1, 13))  # candidate lag orders p; best one chosen via k-fold CV rolling RMSE

TEST_FRAC = 0.10      # storms held out for final evaluation only; remainder used for k-fold CV / final training
N_SPLITS = 5           # number of storm-wise k-folds used to select VAR_ORDER
RANDOM_SEED = 42


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
    """Split storms into a trainval pool (used for k-fold CV + final fit) and
    a held-out test pool (used only for final evaluation)."""
    storm_ids = sorted(df["storm_id"].unique())

    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(storm_ids)

    n = len(storm_ids)
    n_test = max(1, int(round(TEST_FRAC * n)))
    n_trainval = n - n_test

    if n_trainval < N_SPLITS:
        raise ValueError(
            f"Not enough storms ({n_trainval}) for N_SPLITS={N_SPLITS}; "
            "reduce N_SPLITS or TEST_FRAC."
        )

    trainval_ids = storm_ids[:n_trainval]
    test_ids = storm_ids[n_trainval:]

    return trainval_ids, test_ids


def kfold_storm_splits(storm_ids, n_splits, seed):
    """Yield (train_ids, val_ids) storm-wise folds via sklearn KFold."""
    storm_ids = np.array(storm_ids)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    for train_idx, val_idx in kf.split(storm_ids):
        folds.append((list(storm_ids[train_idx]), list(storm_ids[val_idx])))
    return folds


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
# Prepare VAR data
# ----------------------------
def prepare_var_data(df, endog_cols, exog_cols):
    needed = endog_cols + exog_cols
    data = df.dropna(subset=needed).copy()

    endog = data[endog_cols].copy()   # DataFrame, 4 columns (one per region)
    exog = data[exog_cols].copy()

    return data, endog, exog


def standardize_train_test(train_exog, test_exog):
    mean = train_exog.mean()
    std = train_exog.std().replace(0, 1)

    train_scaled = (train_exog - mean) / std
    test_scaled = (test_exog - mean) / std

    return train_scaled, test_scaled, mean, std


# ----------------------------
# Rolling forecast by storm
# ----------------------------
def rolling_forecast_one_storm(
    fitted_results,
    storm_df,
    endog_cols,
    exog_cols,
    exog_mean,
    exog_std,
    steps_ahead,
    var_order,
):
    """Slide a forecast anchor across one storm. At each anchor, feed the fitted
    model's coefficients (via VARResults.forecast) the last `var_order` rows of
    the raw endogenous vector plus the standardized future exog, and compare the
    `steps_ahead`-step-ahead forecast against the true value. Returns a long-format
    DataFrame with one row per (timestamp, region)."""
    storm_df = storm_df.dropna(subset=endog_cols + exog_cols).copy()
    storm_df = storm_df.sort_values(DATETIME_COL).reset_index(drop=True)

    n = len(storm_df)
    rows = []

    for i in range(var_order - 1, n - steps_ahead):
        history = storm_df.iloc[i - var_order + 1 : i + 1]
        future = storm_df.iloc[i + 1 : i + 1 + steps_ahead]

        endog_hist = history[endog_cols].to_numpy()
        exog_future = ((future[exog_cols] - exog_mean) / exog_std).to_numpy()

        try:
            fcst = fitted_results.forecast(y=endog_hist, steps=steps_ahead, exog_future=exog_future)
            pred = fcst[-1]  # last forecast step, shape (n_regions,)
            true = storm_df.loc[i + steps_ahead, endog_cols].to_numpy()
            pred_time = storm_df.loc[i + steps_ahead, DATETIME_COL]

            for j, region in enumerate(REGIONS):
                rows.append(
                    {
                        DATETIME_COL: pred_time,
                        "region": region,
                        "y_true": true[j],
                        "y_pred": pred[j],
                        "residual": true[j] - pred[j],
                    }
                )
        except Exception:
            continue

    return pd.DataFrame(rows)


# ----------------------------
# VAR order diagnostic (VAR.select_order)
# ----------------------------
def diagnostic_var_select_order(endog, exog, maxlag, output_dir):
    """Quick sanity check only: VAR.select_order picks a lag order by minimizing
    in-sample AIC/BIC/HQIC/FPE on a single fit over the pooled trainval series.
    This is NOT the criterion used to choose VAR_ORDER below (that's out-of-sample
    rolling multi-step RMSE via storm-wise k-fold CV) -- it's just a fast,
    independent comparison point."""
    model = VAR(endog, exog=exog)
    try:
        sel = model.select_order(maxlags=maxlag)
    except np.linalg.LinAlgError as exc:
        print(f"  VAR.select_order failed ({exc}); skipping this diagnostic.")
        return pd.DataFrame(columns=["ic", "selected_order"])

    rows = []
    for ic in ("aic", "bic", "hqic", "fpe"):
        order = int(getattr(sel, ic))
        rows.append({"ic": ic, "selected_order": order})
        print(f"  VAR.select_order ({ic}): order={order}")

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(output_dir, f"var_select_order_diagnostic_{current_time}.csv"), index=False)
    return results


# ----------------------------
# VAR order selection (storm-wise k-fold CV)
# ----------------------------
def _fit_and_score_fold(train_df, val_df, endog_cols, exog_cols, order):
    """Fit a VAR(order) on train_df's storms and score it by rolling forecast
    RMSE (pooled across regions, averaged across FORECAST_STEPS) on val_df's storms."""
    _, train_endog, train_exog = prepare_var_data(train_df, endog_cols, exog_cols)
    val_ids = sorted(val_df["storm_id"].unique())

    exog_mean = train_exog.mean()
    exog_std = train_exog.std().replace(0, 1)
    train_exog_s = (train_exog - exog_mean) / exog_std

    fitted = VAR(train_endog, exog=train_exog_s).fit(maxlags=order, trend="c")

    horizon_rmse = {}
    for label, steps in FORECAST_STEPS.items():
        preds = []
        for storm_id in val_ids:
            storm = val_df[val_df["storm_id"] == storm_id].copy()
            preds.append(
                rolling_forecast_one_storm(
                    fitted_results=fitted,
                    storm_df=storm,
                    endog_cols=endog_cols,
                    exog_cols=exog_cols,
                    exog_mean=exog_mean,
                    exog_std=exog_std,
                    steps_ahead=steps,
                    var_order=order,
                )
            )
        fold_pred = pd.concat(preds, ignore_index=True)
        horizon_rmse[label] = np.sqrt(mean_squared_error(fold_pred["y_true"], fold_pred["y_pred"]))

    return horizon_rmse


def select_var_order_kfold(trainval_df, endog_cols, exog_cols, candidates, n_splits, seed):
    """Select VAR_ORDER via storm-wise k-fold cross-validation: for each fold,
    fit on the training storms and score by rolling forecast RMSE (pooled across
    regions, averaged across FORECAST_STEPS) on the held-out fold storms, then
    average across folds."""
    storm_ids = sorted(trainval_df["storm_id"].unique())
    folds = kfold_storm_splits(storm_ids, n_splits=n_splits, seed=seed)

    print(f"  Using {len(folds)} storm-wise folds for CV (storms per fold: "
          f"{[len(v) for _, v in folds]})")

    rows = []
    fold_rows = []
    for order in candidates:
        fold_scores = []
        for fold_i, (fold_train_ids, fold_val_ids) in enumerate(folds):
            fold_train_df = subset(trainval_df, fold_train_ids)
            fold_val_df = subset(trainval_df, fold_val_ids)

            horizon_rmse = _fit_and_score_fold(fold_train_df, fold_val_df, endog_cols, exog_cols, order)
            fold_avg = float(np.mean(list(horizon_rmse.values())))
            fold_scores.append(fold_avg)

            fold_rows.append({
                "VAR_ORDER": order,
                "fold": fold_i,
                **{f"val_RMSE_{k}": v for k, v in horizon_rmse.items()},
                "val_RMSE_avg": fold_avg,
            })

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        rows.append({
            "VAR_ORDER": order,
            "cv_RMSE_mean": mean_score,
            "cv_RMSE_std": std_score,
        })
        print(f"  VAR_ORDER={order}: cv_RMSE_mean={mean_score:.3f}, cv_RMSE_std={std_score:.3f}")

    results = pd.DataFrame(rows).sort_values("cv_RMSE_mean").reset_index(drop=True)
    fold_results = pd.DataFrame(fold_rows)
    best_order = int(results.loc[0, "VAR_ORDER"])
    return best_order, results, fold_results


# ----------------------------
# Main
# ----------------------------
def main():
    df = load_all_storms(DATA_DIR)
    endog_cols = ENDOG_COLS
    exog_cols = EXOG_COLS
    output_dir = os.path.join("VARX_results", f"VARX_results_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    print("Number of storms:", df["storm_id"].nunique())
    print("Regions modeled jointly:", REGIONS)
    print("Endogenous columns:", endog_cols)
    print("VAR order candidates (p):", VAR_ORDER_CANDIDATES)
    print("Exogenous features:", exog_cols)

    trainval_ids, test_ids = split_storms(df)

    print("\nStorm split:")
    print(f"Train+Val (CV pool, {N_SPLITS}-fold):", trainval_ids)
    print("Test :", test_ids)

    trainval_df = subset(df, trainval_ids)
    test_df = subset(df, test_ids)

    trainval_data, trainval_endog, trainval_exog = prepare_var_data(trainval_df, endog_cols, exog_cols)
    test_data, test_endog, test_exog = prepare_var_data(test_df, endog_cols, exog_cols)

    trainval_exog_s, test_exog_s, exog_mean, exog_std = standardize_train_test(
        trainval_exog,
        test_exog,
    )

    print(f"\n[Diagnostic] VAR.select_order (in-sample AIC/BIC/HQIC/FPE, maxlag={max(VAR_ORDER_CANDIDATES)})...")
    diagnostic_var_select_order(
        trainval_endog,
        trainval_exog_s,
        maxlag=max(VAR_ORDER_CANDIDATES),
        output_dir=output_dir,
    )

    print(f"\nSelecting VAR_ORDER via {N_SPLITS}-fold storm-wise cross-validation rolling forecast RMSE...")
    VAR_ORDER, order_results, fold_results = select_var_order_kfold(
        trainval_df=trainval_df,
        endog_cols=endog_cols,
        exog_cols=exog_cols,
        candidates=VAR_ORDER_CANDIDATES,
        n_splits=N_SPLITS,
        seed=RANDOM_SEED,
    )
    print(f"Selected VAR_ORDER={VAR_ORDER}")
    order_results.to_csv(os.path.join(output_dir, f"var_order_selection_{current_time}.csv"), index=False)
    fold_results.to_csv(os.path.join(output_dir, f"var_order_selection_folds_{current_time}.csv"), index=False)

    model = VAR(trainval_endog, exog=trainval_exog_s)

    print("\nFitting final VAR model on full train+val pool...")
    result = model.fit(maxlags=VAR_ORDER, trend="c")
    print(result.summary())

    all_outputs = {}

    for label, steps in FORECAST_STEPS.items():
        print(f"\nRunning rolling {label} forecast...")

        outputs = []
        storm_metrics = {}

        for storm_id in test_ids:
            storm = test_df[test_df["storm_id"] == storm_id].copy()

            pred_df = rolling_forecast_one_storm(
                fitted_results=result,
                storm_df=storm,
                endog_cols=endog_cols,
                exog_cols=exog_cols,
                exog_mean=exog_mean,
                exog_std=exog_std,
                steps_ahead=steps,
                var_order=VAR_ORDER,
            )

            pred_df["storm_id"] = storm_id
            outputs.append(pred_df)

            region_scores = {}
            for region in REGIONS:
                sub = pred_df[pred_df["region"] == region]
                region_scores[region] = metrics(sub["y_true"], sub["y_pred"])
            storm_metrics[storm_id] = region_scores

        final_pred = pd.concat(outputs, ignore_index=True)

        overall_scores = {
            region: metrics(
                final_pred.loc[final_pred["region"] == region, "y_true"],
                final_pred.loc[final_pred["region"] == region, "y_pred"],
            )
            for region in REGIONS
        }

        print(f"\n{label} forecast metrics (overall, by region):")
        for region, score in overall_scores.items():
            print(f"  {region}: " + ", ".join(f"{k}={v:.4f}" for k, v in score.items()))

        print(f"{label} forecast metrics (per storm):")
        for storm_id, region_scores in storm_metrics.items():
            for region, score in region_scores.items():
                print(f"  {storm_id} [{region}]: " + ", ".join(f"{k}={v:.4f}" for k, v in score.items()))

        outname = f"varx_4region_Bz_predictions_{label}_{current_time}.csv"
        final_pred.to_csv(os.path.join(output_dir, outname), index=False)
        print("Saved:", os.path.join(output_dir, outname))

        storm_metrics_rows = []
        for storm_id, region_scores in storm_metrics.items():
            for region, score in region_scores.items():
                storm_metrics_rows.append({"storm_id": storm_id, "region": region, **score})
        storm_metrics_df = pd.DataFrame(storm_metrics_rows)
        storm_metrics_outname = f"varx_storm_metrics_{label}_{current_time}.csv"
        storm_metrics_df.to_csv(os.path.join(output_dir, storm_metrics_outname), index=False)
        print("Saved:", os.path.join(output_dir, storm_metrics_outname))

        all_outputs[label] = {
            "metrics": overall_scores,
            "storm_metrics": storm_metrics,
            "output_file": str(os.path.join(output_dir, outname)),
        }

    summary = {
        "trainval_storms": list(trainval_ids),
        "n_cv_splits": N_SPLITS,
        "test_storms": list(test_ids),
        "regions": REGIONS,
        "endogenous_columns": endog_cols,
        "exogenous_features": exog_cols,
        "forecast_results": all_outputs,
        "var_order_candidates": VAR_ORDER_CANDIDATES,
        "var_order_selected": VAR_ORDER,
        "var_order_cv_rmse": order_results.to_dict(orient="records"),
        "var_order_cv_rmse_by_fold": fold_results.to_dict(orient="records"),
        "model": f"VAR(lags={VAR_ORDER}, trend='c') with exog",
    }

    pd.Series(summary).to_json(os.path.join(output_dir, f"varx_summary_{current_time}.json"), indent=2)
    print("\nSaved summary: varx_summary.json")

    # --- per-storm prediction plots (2x2 grid: one panel per region) ---
    for label in FORECAST_STEPS.keys():
        pred_file = all_outputs[label]["output_file"]
        pred_df = pd.read_csv(pred_file)
        pred_df[DATETIME_COL] = pd.to_datetime(pred_df[DATETIME_COL])

        for storm_id in test_ids:
            storm_pred = pred_df[pred_df["storm_id"] == storm_id]
            if storm_pred.empty:
                continue
            # subplot size based on the number of regions 
            if len(REGIONS)==4:
                fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
            elif len(REGIONS)==3:
                fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
            elif len(REGIONS)==2:
                fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
            elif len((REGIONS))==1:
                fig, axes = plt.subplots(1, 1, figsize=(10, 6))
            # fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
            for ax, region in zip(axes.ravel(), REGIONS):
                d = storm_pred[storm_pred["region"] == region].sort_values(DATETIME_COL)
                ax2 = ax.twinx()
                ax.plot(d[DATETIME_COL], d["y_true"], label="True", color="black")
                ax.plot(d[DATETIME_COL], d["y_pred"], label="Predicted", color="blue")
                ax.set_ylabel('Bz (nT)',fontsize=14)
                ax2.plot(d[DATETIME_COL], d["residual"], label="Residual", color="red", linewidth=0.5, alpha=0.8)
                ax2.grid(alpha=0.3, which="both")
                ax2.set_ylim(-100, 100)
                ax2.tick_params(axis="y", labelcolor="red")
                ax2.set_ylabel('Residual',fontsize=14)
                ax.legend(loc="best", fontsize=12)
                region_score = summary["forecast_results"][label]["storm_metrics"][storm_id][region]
                ax.set_title(
                    f"{region}  RMSE={region_score['RMSE']:.2f} MAE={region_score['MAE']:.2f} R2={region_score['R2']:.2f}",
                    fontsize=14,
                )
                ax.tick_params(axis="x", rotation=30)

            # axes[0, 0].legend(loc="best", fontsize=8)
            fig.suptitle(f"Storm {storm_id} - {label} VARX Bz Forecast ({int(len(REGIONS))} regions)",fontsize=16.0)
            fig.tight_layout()
            plot_name = f"varx_{storm_id}_{label}_forecast_{current_time}.png"
            fig.savefig(os.path.join(output_dir, plot_name))
            plt.close(fig)
            print(f"Saved plot: {plot_name}")


if __name__ == "__main__":
    main()
