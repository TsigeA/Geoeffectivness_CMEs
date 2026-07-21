# CrossCorr_swmf_dayside_bz.py
'''
Aim: screen candidate solar wind exogenous variables for the AR/VARX Bz_dayside_6RE
models by computing, for each candidate, the cross-correlation function (CCF) against
the target across a range of lags.

    r(lag) = corr( X(t - lag), Y(t) )

Positive lag means the solar wind variable X leads the target Y by that many minutes
(the physically relevant direction for feature/lag selection); negative lag means the
target leads X (reverse-causality check / sanity check). Correlations are pooled across
all storms rather than concatenating storms into a single continuous series, so no
spurious correlation is introduced across storm boundaries.

This is an exploratory screening step -- a strong pairwise linear correlation does not
guarantee predictive value once other regressors are in the model (see the Granger
causality test built into varxModel.varx for that). It is meant to build intuition
about which variables and lags to try in AR_swmf_dayside_bz.py / varx_swmf_dayside_bz.py.
'''
import os
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ----------------------------
# User settings
# ----------------------------
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "Storm_csv_nolag"
FILE_GLOB = "*.csv"

DATETIME_COL = "datetime"
TARGET_COL = "target_Bz_dayside_6RE_t0"

# Candidate solar wind exogenous variables (same set used in varx_swmf_dayside_bz.py)
SW_PARAMS = ["Bmag", "BX", "BY_used", "BZ_used", "V", "Pdyn", "SYMH", "Ey", "Es",
             "theta", "Newell_Coupling"]

MAX_LAG_MINUTES = 240  # +/- lag range explored around 0

OUTPUT_DIR = "CrossCorr_results"


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


def infer_cadence_minutes(df):
    diffs = df.groupby("storm_id")[DATETIME_COL].diff().dropna()
    return diffs.median().total_seconds() / 60.0


# ----------------------------
# Cross-correlation function
# ----------------------------
def compute_ccf(df, var_col, target_col, max_lag_samples):
    '''Pooled (across storms) Pearson correlation between var_col shifted by `lag`
    samples and target_col, for lag in [-max_lag_samples, +max_lag_samples].
    Shifting is done per storm so no pairs cross a storm boundary.'''
    rows = []
    groups = [g.sort_values(DATETIME_COL) for _, g in df.groupby("storm_id")]

    for lag in range(-max_lag_samples, max_lag_samples + 1):
        x_parts, y_parts = [], []
        for g in groups:
            x_shifted = g[var_col].shift(lag)
            y = g[target_col]
            valid = x_shifted.notna() & y.notna()
            x_parts.append(x_shifted[valid])
            y_parts.append(y[valid])

        x_all = pd.concat(x_parts, ignore_index=True)
        y_all = pd.concat(y_parts, ignore_index=True)
        n = len(x_all)
        r = np.corrcoef(x_all, y_all)[0, 1] if n > 2 else np.nan
        rows.append({"lag_samples": lag, "r": r, "n": n})

    return pd.DataFrame(rows)


# ----------------------------
# Plotting
# ----------------------------
def plot_ccf_grid(ccf_tables, cadence_min, output_dir):
    n_vars = len(ccf_tables)
    n_cols = 3
    n_rows = int(np.ceil(n_vars / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.2 * n_rows), squeeze=False)

    for ax, (var, table) in zip(axes.ravel(), ccf_tables.items()):
        lag_min = table["lag_samples"] * cadence_min
        thresh = 1.96 / np.sqrt(table["n"].clip(lower=1))

        ax.plot(lag_min, table["r"], color="tab:blue", linewidth=1.2)
        ax.plot(lag_min, thresh, "k--", linewidth=0.7)
        ax.plot(lag_min, -thresh, "k--", linewidth=0.7)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.axhline(0, color="gray", linewidth=0.5)

        peak_idx = table["r"].abs().idxmax()
        peak_lag_min = table.loc[peak_idx, "lag_samples"] * cadence_min
        peak_r = table.loc[peak_idx, "r"]
        ax.scatter([peak_lag_min], [peak_r], color="red", zorder=5, s=25)

        ax.set_title(f"{var}\npeak r={peak_r:.2f} @ lag={peak_lag_min:.0f} min", fontsize=10)
        ax.set_xlabel("lag (min); +ve = SW var leads target", fontsize=8)
        ax.set_ylabel("r", fontsize=8)
        ax.tick_params(labelsize=8)

    for ax in axes.ravel()[n_vars:]:
        ax.axis("off")

    fig.suptitle(f"Cross-correlation vs {TARGET_COL}", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ccf_grid_{current_time}.png"), dpi=150)
    plt.close(fig)


def plot_peak_summary(summary_df, output_dir):
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(summary_df) + 1))
    ordered = summary_df.sort_values("peak_abs_r")
    colors = ["tab:red" if r < 0 else "tab:blue" for r in ordered["peak_r"]]
    ax.barh(ordered["variable"], ordered["peak_r"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("peak correlation r (signed)")
    ax.set_title("Peak |cross-correlation| with " + TARGET_COL + " per SW variable")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"ccf_peak_summary_{current_time}.png"), dpi=150)
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    df = load_all_storms(DATA_DIR)
    output_dir = os.path.join(OUTPUT_DIR, f"CrossCorr_results_{current_time}")
    os.makedirs(output_dir, exist_ok=True)

    cadence_min = infer_cadence_minutes(df)
    max_lag_samples = int(round(MAX_LAG_MINUTES / cadence_min))

    print("Number of storms:", df["storm_id"].nunique())
    print("Target variable:", TARGET_COL)
    print(f"Inferred cadence: {cadence_min:.1f} min")
    print(f"Lag range: +/-{MAX_LAG_MINUTES} min ({max_lag_samples} samples)")

    sw_params = [c for c in SW_PARAMS if c in df.columns]
    missing = set(SW_PARAMS) - set(sw_params)
    if missing:
        print("Warning: missing columns skipped:", missing)

    ccf_tables = {}
    summary_rows = []
    long_rows = []

    for var in sw_params:
        print(f"Computing CCF for {var}...")
        table = compute_ccf(df, var, TARGET_COL, max_lag_samples)
        ccf_tables[var] = table

        peak_idx = table["r"].abs().idxmax()
        peak_row = table.loc[peak_idx]
        summary_rows.append({
            "variable": var,
            "peak_lag_samples": int(peak_row["lag_samples"]),
            "peak_lag_minutes": peak_row["lag_samples"] * cadence_min,
            "peak_r": peak_row["r"],
            "peak_abs_r": abs(peak_row["r"]),
            "zero_lag_r": table.loc[table["lag_samples"] == 0, "r"].iloc[0],
        })

        table_long = table.copy()
        table_long["variable"] = var
        table_long["lag_minutes"] = table_long["lag_samples"] * cadence_min
        long_rows.append(table_long)

    summary_df = pd.DataFrame(summary_rows).sort_values("peak_abs_r", ascending=False).reset_index(drop=True)
    print("\nRanked by peak |correlation| with target:")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(os.path.join(output_dir, f"ccf_summary_{current_time}.csv"), index=False)
    pd.concat(long_rows, ignore_index=True).to_csv(
        os.path.join(output_dir, f"ccf_full_{current_time}.csv"), index=False
    )

    plot_ccf_grid(ccf_tables, cadence_min, output_dir)
    plot_peak_summary(summary_df, output_dir)

    print("\nSaved results to:", output_dir)


if __name__ == "__main__":
    main()
