# AR_swmf_Nightside_bz.py
'''
Aim: to predict the storm time magnetospheric dynamics using an AutoReg model
with Exogenous inputs (ARX) for a single target variable: Bz_Nightside_6RE.
 - This model predicts Bz_nightside_6RE at 1 hour and 2 hours ahead.
 - Implemented via statsmodels AutoReg with lags=p and exog.
 - The exogenous features are selected solar wind parameters.
 - The data is split storm-wise into training, validation, and test sets.
 - All exogenous features are standardized using training-set statistics.
 - Rolling forecast applies the training-fitted parameters to each storm history window
   without re-running optimization (AutoReg.predict with params=fitted_model.params).
@author: TsigeA
@date: Jul 21, 2026
@updates:
'''
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


# ----------------------------
# User settings
# ----------------------------
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "Storm_csv_nolag_nightside"
FILE_GLOB = "*.csv"

DATETIME_COL = "datetime"

TARGET_COL = "target_Bz_nightside_6RE_t0"

TARGET_1H = "target_Bz_nightside_6RE_tplus_60m"
TARGET_2H = "target_Bz_nightside_6RE_tplus_120m"

FORECAST_STEPS = {
    "1h": 4,   # 4 x 15 min = 60 min
    "2h": 8,   # 8 x 15 min = 120 min
}

AR_ORDER_CANDIDATES = list(range(1, 13))  # candidate lag orders p; best one is chosen via k-fold CV rolling RMSE

TEST_FRAC = 0.10      # storms held out for final evaluation only; remainder used for k-fold CV / final training
N_SPLITS = 5           # number of storm-wise k-folds used to select AR_ORDER
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


def get_exog_columns(df):
    exclude = {
        DATETIME_COL,
        "storm_id",
        "source_file",
        "phase",
        "n_points_target",
        TARGET_COL,
        TARGET_1H,
        TARGET_2H,
    }
    use_cols = {"BY_used", "BZ_used"} #"Bmag", "BX", "BY_used", "BZ_used", "V", "Pdyn", "SYMH", "Ey", "Es", "theta", "Newell_Coupling"

    exog_cols = []

    for col in df.columns:
        if col in exclude:
            continue
        if "_lag_" in col or "lag_" in col:
            continue
        # if pd.api.types.is_numeric_dtype(df[col]):
        #     exog_cols.append(col)
        if col in use_cols:
            exog_cols.append(col)

    return exog_cols


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
# Prepare AR data
# ----------------------------
def prepare_ar_data(df, exog_cols):
    needed = [TARGET_COL] + exog_cols
    data = df.dropna(subset=needed).copy()

    endog = data[TARGET_COL]          # 1-D Series
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
    fitted_model,
    storm_df,
    exog_cols,
    exog_mean,
    exog_std,
    steps_ahead,
    ar_order,
):
    storm_df = storm_df.dropna(subset=[TARGET_COL] + exog_cols).copy()
    storm_df = storm_df.sort_values(DATETIME_COL).reset_index(drop=True)

    y_true = []
    y_pred = []
    pred_time = []

    n = len(storm_df)
    # start after initial lags and a few extra points to ensure enough history. It also depends on the number of exog features used. 
    # scales with ar_order itself (ar_order + len(exog_cols) + 1)
    # for i in range(ar_order + 5, n - steps_ahead): 
    for i in range(ar_order + len(fitted_model.params) - 1, n - steps_ahead):
        history = storm_df.iloc[: i + 1]
        future = storm_df.iloc[i + 1 : i + 1 + steps_ahead]

        endog_hist = history[TARGET_COL]                          # Series
        exog_hist = (history[exog_cols] - exog_mean) / exog_std
        exog_future = (future[exog_cols] - exog_mean) / exog_std

        try: # this will swallow the error caused by insufficient history length for the given AR order, and continue to the next iteration
            window_model = AutoReg(
                endog=endog_hist,
                lags=ar_order,
                exog=exog_hist,
                trend="c",
            )

            n_hist = len(endog_hist)
            fcst = window_model.predict(
                params=fitted_model.params,
                start=n_hist,
                end=n_hist + steps_ahead - 1,
                exog_oos=exog_future.values,
            )

            pred_value = fcst.iloc[-1]
            # # visualize the prediction, confidence interval and true value for debugging
            # fig, ax = plt.subplots(figsize=(8, 4))
            # fcst.plot_predict(start=n_hist, end=n_hist + steps_ahead - 1, ax=ax)
            # ax.axhline(y=pred_value, color="blue", linestyle="--", label="Predicted")
            # ax.axhline(y=storm_df.loc[i + steps_ahead, TARGET_COL], color="red", linestyle="--", label="True")
            # ax.set_title(f"Rolling Forecast - Storm {storm_df.loc[i, 'storm_id']} - Step {steps_ahead} - Time {storm_df.loc[i + steps_ahead, DATETIME_COL]}")
            # ax.legend()
            # plt.pause(0.1)  # pause to allow the plot to render
            # plt.close(fig)  # close the figure to avoid displaying it during the loop
            true_value = storm_df.loc[i + steps_ahead, TARGET_COL]

            y_pred.append(pred_value)
            y_true.append(true_value)
            pred_time.append(storm_df.loc[i + steps_ahead, DATETIME_COL])

        except Exception:
            continue

    return pd.DataFrame(
        {
            DATETIME_COL: pred_time,
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": np.array(y_true) - np.array(y_pred),
        }
    )


# ----------------------------
# AR order diagnostic (ar_select_order)
# ----------------------------
def diagnostic_ar_select_order(endog, exog, maxlag, output_dir):
    """Quick sanity check only: ar_select_order picks a lag order by minimizing
    in-sample AIC/BIC/HQIC on a single one-step-ahead fit over the pooled
    trainval series. This is NOT the criterion used to choose AR_ORDER below
    (that's out-of-sample rolling multi-step RMSE via storm-wise k-fold CV) --
    it's just a fast, independent comparison point."""
    rows = []
    for ic in ("aic", "bic", "hqic"): # criterion to minimize
        sel = ar_select_order(endog, maxlag=maxlag, ic=ic, trend="c", exog=exog, old_names=False)
        order = len(sel.ar_lags)
        rows.append({"ic": ic, "selected_lags": sel.ar_lags, "selected_order": order})
        print(f"  ar_select_order ({ic}): lags={sel.ar_lags} -> order={order}")

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(output_dir, f"ar_select_order_diagnostic_{current_time}.csv"), index=False)
    return results


# ----------------------------
# AR order selection (storm-wise k-fold CV)
# ----------------------------
def _fit_and_score_fold(train_df, val_df, exog_cols, order):
    """Fit an AutoReg(order) on train_df's storms and score it by rolling
    forecast RMSE (averaged across FORECAST_STEPS) on val_df's storms."""
    _, train_endog, train_exog = prepare_ar_data(train_df, exog_cols)
    val_ids = sorted(val_df["storm_id"].unique())

    exog_mean = train_exog.mean()
    exog_std = train_exog.std().replace(0, 1)
    train_exog_s = (train_exog - exog_mean) / exog_std

    fitted = AutoReg(endog=train_endog, lags=order, exog=train_exog_s, trend="c").fit()

    horizon_rmse = {}
    for label, steps in FORECAST_STEPS.items():
        preds = []
        for storm_id in val_ids:
            storm = val_df[val_df["storm_id"] == storm_id].copy()
            preds.append(
                rolling_forecast_one_storm(
                    fitted_model=fitted,
                    storm_df=storm,
                    exog_cols=exog_cols,
                    exog_mean=exog_mean,
                    exog_std=exog_std,
                    steps_ahead=steps,
                    ar_order=order,
                )
            )
        fold_pred = pd.concat(preds, ignore_index=True)
        horizon_rmse[label] = np.sqrt(mean_squared_error(fold_pred["y_true"], fold_pred["y_pred"]))

    return horizon_rmse


def select_ar_order_kfold(trainval_df, exog_cols, candidates, n_splits, seed):
    """Select AR_ORDER via storm-wise k-fold cross-validation: for each fold,
    fit on the training storms and score by rolling forecast RMSE (averaged
    across FORECAST_STEPS) on the held-out fold storms, then average across
    folds."""
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

            horizon_rmse = _fit_and_score_fold(fold_train_df, fold_val_df, exog_cols, order)
            fold_avg = float(np.mean(list(horizon_rmse.values())))
            fold_scores.append(fold_avg)

            fold_rows.append({
                "AR_ORDER": order,
                "fold": fold_i,
                **{f"val_RMSE_{k}": v for k, v in horizon_rmse.items()},
                "val_RMSE_avg": fold_avg,
            })

        mean_score = float(np.mean(fold_scores))
        std_score = float(np.std(fold_scores))
        rows.append({
            "AR_ORDER": order,
            "cv_RMSE_mean": mean_score,
            "cv_RMSE_std": std_score,
        })
        print(f"  AR_ORDER={order}: cv_RMSE_mean={mean_score:.3f}, cv_RMSE_std={std_score:.3f}")

    results = pd.DataFrame(rows).sort_values("cv_RMSE_mean").reset_index(drop=True)
    fold_results = pd.DataFrame(fold_rows)
    best_order = int(results.loc[0, "AR_ORDER"])
    return best_order, results, fold_results


# ----------------------------
# Main
# ----------------------------
def main():
    df = load_all_storms(DATA_DIR)
    exog_cols = get_exog_columns(df)
    output_dir = os.path.join("AR_results_Nightside", f"AR_results_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    print("Number of storms:", df["storm_id"].nunique())
    print("Target variable:", TARGET_COL)
    print("AR order candidates (p):", AR_ORDER_CANDIDATES)
    print("Number of exogenous features:", len(exog_cols))
    print("Exogenous features:")
    for c in exog_cols:
        print("  ", c)

    trainval_ids, test_ids = split_storms(df)

    print("\nStorm split:")
    print(f"Train+Val (CV pool, {N_SPLITS}-fold):", trainval_ids)
    print("Test :", test_ids)

    trainval_df = subset(df, trainval_ids)
    test_df = subset(df, test_ids)

    trainval_data, trainval_endog, trainval_exog = prepare_ar_data(trainval_df, exog_cols)
    test_data, test_endog, test_exog = prepare_ar_data(test_df, exog_cols)

    trainval_exog_s, test_exog_s, exog_mean, exog_std = standardize_train_test(
        trainval_exog,
        test_exog,
    )

    print(f"\n[Diagnostic] ar_select_order (in-sample AIC/BIC/HQIC, maxlag={max(AR_ORDER_CANDIDATES)})...")
    diagnostic_ar_select_order(
        trainval_endog,
        trainval_exog_s,
        maxlag=max(AR_ORDER_CANDIDATES),
        output_dir=output_dir,
    )

    print(f"\nSelecting AR_ORDER via {N_SPLITS}-fold storm-wise cross-validation rolling forecast RMSE...")
    AR_ORDER, order_results, fold_results = select_ar_order_kfold(
        trainval_df=trainval_df,
        exog_cols=exog_cols,
        candidates=AR_ORDER_CANDIDATES,
        n_splits=N_SPLITS,
        seed=RANDOM_SEED,
    )
    print(f"Selected AR_ORDER={AR_ORDER}")
    order_results.to_csv(os.path.join(output_dir, f"ar_order_selection_{current_time}.csv"), index=False)
    fold_results.to_csv(os.path.join(output_dir, f"ar_order_selection_folds_{current_time}.csv"), index=False)

    model = AutoReg(
        endog=trainval_endog,
        lags=AR_ORDER,
        exog=trainval_exog_s,
        trend="c", # constant term
    )

    print("\nFitting final AutoReg model on full train+val pool...")
    result = model.fit()
    print(result.summary())
    print(f"Number of parameters: {len(result.params)}")

    all_outputs = {}

    for label, steps in FORECAST_STEPS.items():
        print(f"\nRunning rolling {label} forecast...")

        outputs = []
        storm_metrics = {}

        for storm_id in test_ids:
            storm = test_df[test_df["storm_id"] == storm_id].copy()

            pred_df = rolling_forecast_one_storm(
                fitted_model=result,
                storm_df=storm,
                exog_cols=exog_cols,
                exog_mean=exog_mean,
                exog_std=exog_std,
                steps_ahead=steps,
                ar_order=AR_ORDER,
            )

            pred_df["storm_id"] = storm_id
            outputs.append(pred_df)

            storm_metrics[storm_id] = metrics(pred_df["y_true"], pred_df["y_pred"])

        final_pred = pd.concat(outputs, ignore_index=True)

        score = metrics(final_pred["y_true"], final_pred["y_pred"])

        print(f"\n{label} forecast metrics (overall):")
        for k, v in score.items():
            print(f"  {k}: {v:.4f}")

        print(f"{label} forecast metrics (per storm):")
        for storm_id, storm_score in storm_metrics.items():
            print(f"  {storm_id}: " + ", ".join(f"{k}={v:.4f}" for k, v in storm_score.items()))

        outname = f"ar_nightside_6RE_Bz_predictions_{label}_{current_time}.csv"
        final_pred.to_csv(os.path.join(output_dir, outname), index=False)
        print("Saved:", os.path.join(output_dir, outname))

        storm_metrics_outname = f"ar_storm_metrics_{label}_{current_time}.csv"
        storm_metrics_df = (
            pd.DataFrame.from_dict(storm_metrics, orient="index")
            .rename_axis("storm_id")
            .reset_index()
        )
        storm_metrics_df.to_csv(os.path.join(output_dir, storm_metrics_outname), index=False)
        print("Saved:", os.path.join(output_dir, storm_metrics_outname))

        all_outputs[label] = {
            "metrics": score,
            "storm_metrics": storm_metrics,
            "output_file": str(os.path.join(output_dir, outname)),
        }

    summary = {
        "trainval_storms": list(trainval_ids),
        "n_cv_splits": N_SPLITS,
        "test_storms": list(test_ids),
        "target_variable": TARGET_COL,
        "exogenous_features": exog_cols,
        "forecast_results": all_outputs,
        "ar_order_candidates": AR_ORDER_CANDIDATES,
        "ar_order_selected": AR_ORDER,
        "ar_order_cv_rmse": order_results.to_dict(orient="records"),
        "ar_order_cv_rmse_by_fold": fold_results.to_dict(orient="records"),
        "model": f"AutoReg(lags={AR_ORDER})",
    }

    pd.Series(summary).to_json(os.path.join(output_dir, f"ar_summary_{current_time}.json"), indent=2)
    print("\nSaved summary: ar_summary.json")
    # --- per-storm prediction plots ---
    for label in FORECAST_STEPS.keys():
        pred_file = all_outputs[label]["output_file"]
        pred_df = pd.read_csv(pred_file)
        pred_df[DATETIME_COL] = pd.to_datetime(pred_df[DATETIME_COL])

        for storm_id in test_ids:
            storm_pred = pred_df[pred_df["storm_id"] == storm_id]

            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax2 = ax.twinx()
            ax.plot(storm_pred[DATETIME_COL], storm_pred["y_true"], label="True", color="black")
            ax.plot(storm_pred[DATETIME_COL], storm_pred["y_pred"], label="Predicted", color="blue")
            ax2.plot(storm_pred[DATETIME_COL], storm_pred["residual"], label="Residual", color="red", linewidth=0.5, alpha=0.8)
            ax.set_title(f"Storm {storm_id} - {label} Bz_Nightside_6RE Forecast")
            ax.set_xlabel("Time", fontsize=14)
            ax.set_ylabel(TARGET_COL, fontsize=14)
            ax2.set_ylabel("Residual", fontsize=14)
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.set_ylim(-100, 100)
            storm_score = summary["forecast_results"][label]["storm_metrics"][storm_id]
            fig.text(
                0.01, 0.95,
                f"RMSE: {storm_score['RMSE']:.2f}\n"
                f"MAE: {storm_score['MAE']:.2f}\n"
                f"R²: {storm_score['R2']:.2f}",
                transform=fig.gca().transAxes, fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax.legend()
            ax2.grid(alpha=0.3, which="both")
            plot_name = f"ar_{storm_id}_{label}_forecast_{current_time}.png"
            plt.savefig(os.path.join(output_dir, plot_name))
            plt.close()
            print(f"Saved plot: {plot_name}")


if __name__ == "__main__":
    main()
