# varmax_swmf_dayside_bz.py
'''
Aim: to predict the storm time magnetospheric dynamics using an AutoReg model
with Exogenous inputs (ARX) for a single target variable: Bz_dayside_6RE.
 - This model predicts Bz_dayside_6RE at 1 hour and 2 hours ahead.
 - Implemented via statsmodels AutoReg with lags=p and exog (no seasonal component).
 - The exogenous features are all the non-lagged numeric columns in the dataset (solar wind parameters),
   excluding the target variable and its future versions.
 - The data is split storm-wise into training, validation, and test sets.
 - All exogenous features are standardized using training-set statistics.
 - Rolling forecast applies the training-fitted parameters to each storm history window
   without re-running optimization (AutoReg.predict with params=fitted_model.params).
'''
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")


# ----------------------------
# User settings
# ----------------------------
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "storm_csvs"
FILE_GLOB = "*.csv"

DATETIME_COL = "datetime"

TARGET_COL = "target_Bz_dayside_6RE_t0"

TARGET_1H = "target_Bz_dayside_6RE_tplus_60m"
TARGET_2H = "target_Bz_dayside_6RE_tplus_120m"

FORECAST_STEPS = {
    "1h": 12,   # 12 x 5 min = 60 min
    "2h": 24,   # 24 x 5 min = 120 min
}

AR_ORDER = 2        # autoregressive lag order p in AR(p)X

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10
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
        TARGET_COL,
        TARGET_1H,
        TARGET_2H,
    }

    exog_cols = []

    for col in df.columns:
        if col in exclude:
            continue
        if "_lag_" in col or "lag_" in col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            exog_cols.append(col)

    return exog_cols


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
# Prepare AR data
# ----------------------------
def prepare_ar_data(df, exog_cols):
    needed = [TARGET_COL] + exog_cols
    data = df.dropna(subset=needed).copy()

    endog = data[TARGET_COL]          # 1-D Series
    exog = data[exog_cols].copy()

    return data, endog, exog


def standardize_train_val_test(train_exog, val_exog, test_exog):
    mean = train_exog.mean()
    std = train_exog.std().replace(0, 1)

    train_scaled = (train_exog - mean) / std
    val_scaled = (val_exog - mean) / std
    test_scaled = (test_exog - mean) / std

    return train_scaled, val_scaled, test_scaled, mean, std


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
):
    storm_df = storm_df.dropna(subset=[TARGET_COL] + exog_cols).copy()
    storm_df = storm_df.sort_values(DATETIME_COL).reset_index(drop=True)

    y_true = []
    y_pred = []
    pred_time = []

    n = len(storm_df)

    for i in range(AR_ORDER + 5, n - steps_ahead):
        history = storm_df.iloc[: i + 1]
        future = storm_df.iloc[i + 1 : i + 1 + steps_ahead]

        endog_hist = history[TARGET_COL]                          # Series
        exog_hist = (history[exog_cols] - exog_mean) / exog_std
        exog_future = (future[exog_cols] - exog_mean) / exog_std

        try:
            window_model = AutoReg(
                endog=endog_hist,
                lags=AR_ORDER,
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
# Main
# ----------------------------
def main():
    df = load_all_storms(DATA_DIR)
    exog_cols = get_exog_columns(df)
    output_dir = Path(f"AR_results_{current_time}")
    output_dir.mkdir(exist_ok=True)

    print("Number of storms:", df["storm_id"].nunique())
    print("Target variable:", TARGET_COL)
    print("AR order (p):", AR_ORDER)
    print("Number of exogenous features:", len(exog_cols))
    print("Exogenous features:")
    for c in exog_cols:
        print("  ", c)

    train_ids, val_ids, test_ids = split_storms(df)

    print("\nStorm split:")
    print("Train:", train_ids)
    print("Val  :", val_ids)
    print("Test :", test_ids)

    train_df = subset(df, train_ids)
    val_df = subset(df, val_ids)
    test_df = subset(df, test_ids)

    train_data, train_endog, train_exog = prepare_ar_data(train_df, exog_cols)
    val_data, val_endog, val_exog = prepare_ar_data(val_df, exog_cols)
    test_data, test_endog, test_exog = prepare_ar_data(test_df, exog_cols)

    train_exog_s, val_exog_s, test_exog_s, exog_mean, exog_std = standardize_train_val_test(
        train_exog,
        val_exog,
        test_exog,
    )

    model = AutoReg(
        endog=train_endog,
        lags=AR_ORDER,
        exog=train_exog_s,
        trend="c",
    )

    print("\nFitting AutoReg model...")
    result = model.fit()
    print(result.summary())

    all_outputs = {}

    for label, steps in FORECAST_STEPS.items():
        print(f"\nRunning rolling {label} forecast...")

        outputs = []

        for storm_id in test_ids:
            storm = test_df[test_df["storm_id"] == storm_id].copy()

            pred_df = rolling_forecast_one_storm(
                fitted_model=result,
                storm_df=storm,
                exog_cols=exog_cols,
                exog_mean=exog_mean,
                exog_std=exog_std,
                steps_ahead=steps,
            )

            pred_df["storm_id"] = storm_id
            outputs.append(pred_df)

        final_pred = pd.concat(outputs, ignore_index=True)

        score = metrics(final_pred["y_true"], final_pred["y_pred"])

        print(f"\n{label} forecast metrics:")
        for k, v in score.items():
            print(f"  {k}: {v:.4f}")

        outname = f"ar_dayside_6RE_Bz_predictions_{label}_{current_time}.csv"
        final_pred.to_csv(output_dir / outname, index=False)
        print("Saved:", output_dir / outname)

        all_outputs[label] = {
            "metrics": score,
            "output_file": str(output_dir / outname),
        }

    summary = {
        "train_storms": list(train_ids),
        "validation_storms": list(val_ids),
        "test_storms": list(test_ids),
        "target_variable": TARGET_COL,
        "exogenous_features": exog_cols,
        "forecast_results": all_outputs,
        "model": f"AutoReg(lags={AR_ORDER})",
    }

    pd.Series(summary).to_json(output_dir / f"ar_summary_{current_time}.json", indent=2)
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
            ax.set_title(f"Storm {storm_id} - {label} Bz_dayside_6RE Forecast")
            ax.set_xlabel("Time")
            ax.set_ylabel(TARGET_COL)
            ax2.set_ylabel("Residual")
            fig.text(
                0.01, 0.95,
                f"RMSE: {summary['forecast_results'][label]['metrics']['RMSE']:.2f}\n"
                f"MAE: {summary['forecast_results'][label]['metrics']['MAE']:.2f}\n"
                f"R²: {summary['forecast_results'][label]['metrics']['R2']:.2f}",
                transform=fig.gca().transAxes, fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )
            ax.legend()
            ax.grid()
            plot_name = f"ar_{storm_id}_{label}_forecast_{current_time}.png"
            plt.savefig(output_dir / plot_name)
            plt.close()
            print(f"Saved plot: {plot_name}")


if __name__ == "__main__":
    main()
