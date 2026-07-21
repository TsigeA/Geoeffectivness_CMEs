# GrangerCausality_swmf_dayside_bz.py
'''
Aim: confirm the SW-variable screening from CrossCorr_swmf_dayside_bz.py with a
proper Granger-causality test. Cross-correlation only looks at one variable at a
time, so a variable that is merely collinear with a true driver can look important.
This script fits all candidate SW variables jointly (and the target's own AR
history) via varxModel.varx(), which tests whether each exogenous channel's full
lag block still adds significant explanatory power once every other channel is
already in the model.

Model: Y(t) = A*Y(t-1) + B*X(t) + e(t), with Y = target_Bz_dayside_6RE (single
output channel here) and X = the candidate SW variables. For each exogenous
channel, varx() returns:
  - B_pval:    p-value from a Deviance (likelihood-ratio-style) test of whether
               dropping that channel's entire lag block significantly hurts the fit
  - B_Rvalue:  generalized R = sqrt(1 - exp(-Deviance/T)), i.e. the effect size

Fit only on the training storms (same split as AR_swmf_dayside_bz.py /
varx_swmf_dayside_bz.py, same RANDOM_SEED) so this screening step cannot leak
information from the held-out validation/test storms.
'''
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import varxModel

warnings.filterwarnings("ignore")


# ----------------------------
# User settings
# ----------------------------
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "Storm_csv_nolag"
FILE_GLOB = "*.csv"

DATETIME_COL = "datetime"
TARGET_COL = "target_Bz_dayside_6RE_t0"

# Same candidate solar wind variables screened in CrossCorr_swmf_dayside_bz.py
SW_PARAMS = ["Bmag", "BX", "BY_used", "BZ_used", "V", "Pdyn", "SYMH", "Ey", "Es",
             "theta", "Newell_Coupling"]

AR_ORDER = 4    # na: lags of the target itself (1 hour of AR history at 15-min cadence)
MA_ORDER = 16   # nb: lags of X (4 hours of SW history, matching the widened CCF window)
GAMMA = 1.0     # ridge shrinkage passed to varx()

TRAIN_FRAC = 0.80
VAL_FRAC = 0.10
TEST_FRAC = 0.10
RANDOM_SEED = 42

OUTPUT_DIR = "GrangerCausality_results"


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
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)


# ----------------------------
# Storm-wise split (same convention as AR_swmf_dayside_bz.py / varx_swmf_dayside_bz.py)
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
# Per-storm array preparation
# ----------------------------
def storm_arrays(storm_df, sw_params):
    '''NaNs are kept as-is; varx()/myxcorr() exclude them internally.'''
    d = storm_df.sort_values(DATETIME_COL).reset_index(drop=True)
    Y = d[[TARGET_COL]].to_numpy(dtype=float)
    X = d[sw_params].to_numpy(dtype=float)
    return Y, X


def fit_standardizer(arrays):
    stacked = np.concatenate(arrays, axis=0)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std[std == 0] = 1.0
    return mean, std


def standardize(arr, mean, std):
    return (arr - mean) / std


# ----------------------------
# Plotting
# ----------------------------
def plot_granger_summary(table, output_dir):
    ordered = table.sort_values("B_Rvalue")
    colors = ["tab:green" if p < 0.05 else "tab:gray" for p in ordered["B_pval"]]

    fig, ax = plt.subplots(figsize=(8, 0.45 * len(ordered) + 1))
    ax.barh(ordered["variable"], ordered["B_Rvalue"], color=colors)
    for i, (r, p) in enumerate(zip(ordered["B_Rvalue"], ordered["B_pval"])):
        ax.text(r + 0.01, i, f"p={p:.3g}", va="center", fontsize=8)
    ax.set_xlabel("generalized R (effect size)")
    ax.set_title(f"Granger causality on {TARGET_COL}\n(green = p<0.05, gray = not significant)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"granger_summary_{current_time}.png"), dpi=150)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    df = load_all_storms(DATA_DIR)
    output_dir = os.path.join(OUTPUT_DIR, f"GrangerCausality_results_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    sw_params = [c for c in SW_PARAMS if c in df.columns]
    missing = set(SW_PARAMS) - set(sw_params)
    if missing:
        print("Warning: missing columns skipped:", missing)

    train_ids, val_ids, test_ids = split_storms(df)
    print("Number of storms:", df["storm_id"].nunique())
    print("Train storms:", len(train_ids), " Val:", len(val_ids), " Test:", len(test_ids))
    print("Target variable:", TARGET_COL)
    print("Candidate exogenous variables:", sw_params)
    print(f"AR_ORDER (na)={AR_ORDER}, MA_ORDER (nb)={MA_ORDER}, GAMMA={GAMMA}")

    train_df = subset(df, train_ids)
    train_storms = [storm_arrays(g, sw_params) for _, g in train_df.groupby("storm_id")]

    Y_mean, Y_std_ = fit_standardizer([Y for Y, _ in train_storms])
    X_mean, X_std_ = fit_standardizer([X for _, X in train_storms])

    Y_list = [standardize(Y, Y_mean, Y_std_) for Y, _ in train_storms]
    X_list = [standardize(X, X_mean, X_std_) for _, X in train_storms]

    print("\nFitting VARX (single-target ARX with Granger test) on", len(Y_list), "training storms...")
    model = varxModel.varx(Y_list, AR_ORDER, X=X_list, nb=MA_ORDER, gamma=GAMMA)

    print(f"\nSelf-AR term (target's own history) p-value: {model['A_pval'][0, 0]:.4g}, "
          f"R={model['A_Rvalue'][0, 0]:.3f}")

    table = pd.DataFrame({
        "variable": sw_params,
        "B_pval": model["B_pval"][0, :],
        "B_Deviance": model["B_Deviance"][0, :],
        "B_Rvalue": model["B_Rvalue"][0, :],
    }).sort_values("B_Rvalue", ascending=False).reset_index(drop=True)
    table["significant_p<0.05"] = table["B_pval"] < 0.05

    print("\nGranger causality ranking (exogenous SW variables):")
    print(table.to_string(index=False))

    table.to_csv(os.path.join(output_dir, f"granger_ranking_{current_time}.csv"), index=False)
    plot_granger_summary(table, output_dir)

    print("\nSaved results to:", output_dir)


if __name__ == "__main__":
    main()
