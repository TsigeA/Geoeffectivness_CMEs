"""Plot regional Bz time series and their ACF/PACF for each storm event."""

from pathlib import Path
import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


DATA_DIR = Path("Storm_csvs_4regions2") #Path("Storm_csvs_4regions") Storm_csv_nolag
OUTPUT_DIR = Path("ACF_PACF_regional_Bz") #Path("ACF_PACF_regional_Bz") ACF_PACF_Bz_6RE_dayside
FILE_GLOB = "*.csv"
MAX_LAGS = 50

# CSV column, short display label, and a consistent color.
REGIONS = [
    ("target_Bz_dayside_t0", "Dayside", "tab:blue"),
    ("target_Bz_dawnside_t0", "Dawnside", "tab:orange"),
    ("target_Bz_duskside_t0", "Duskside", "tab:green"),
    ("target_Bz_nightside_t0", "Nightside", "tab:red"),
]
# REGIONS = [
#     ("target_Bz_dayside_6RE_t0", "Dayside", "tab:blue")]

def plot_storm(filepath: Path, storm_time: dict[str, pd.Timestamp]) -> Path:
    """Create one summary figure for a storm CSV and return its output path."""
    storm_id = filepath.stem
    df = pd.read_csv(filepath, parse_dates=["datetime"])

    required = {"datetime", *(column for column, _, _ in REGIONS)}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{filepath.name} is missing columns: {', '.join(missing)}")

    df = df.sort_values("datetime")
    fig, axes = plt.subplots(
        nrows=len(REGIONS),
        ncols=3,
        figsize=(16, 12),  #figsize=(16, 12)
        constrained_layout=True,
    )

    for row, (column, label, color) in enumerate(REGIONS):
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) < 4:
            raise ValueError(f"{column} in {filepath.name} has fewer than 4 valid values")

        # PACF requires the lag count to be strictly less than half the sample size.
        nlags = min(MAX_LAGS, len(series) // 2 - 1)
        if len(REGIONS)>1:
            time_ax, acf_ax, pacf_ax = axes[row]
        else:
            time_ax, acf_ax, pacf_ax = axes

        time_ax.plot(df["datetime"], df[column], color=color, linewidth=1.2)
        for phase, phase_time, phase_color in (
            ("SSC", storm_time["SSC"], "crimson"),
            ("Main", storm_time["Main"], "darkgreen"),
            ("Recovery", storm_time["Recovery"], "magenta"),
        ):
            if pd.notna(phase_time):
                time_ax.axvline(
                    phase_time,
                    color=phase_color,
                    linestyle="--",
                    linewidth=1.5,
                    alpha=0.9,
                    label=phase,
                )
        # time_ax.axhline(series.mean(), color="0.35", linestyle="--", linewidth=0.8,
        #                 label=f"mean = {series.mean():.1f} nT")
        time_ax.set_title(label, loc="left", fontweight="bold")
        time_ax.set_ylabel("Bz (nT)")
        time_ax.grid(alpha=0.25)
        time_ax.legend(loc="best", fontsize=8, frameon=False, ncols=3)
        time_ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        time_ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3, maxticks=6))
        time_ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
            time_ax.xaxis.get_major_locator()
        ))

        plot_acf(series, ax=acf_ax, lags=nlags, color=color, zero=False)
        plot_pacf(series, ax=pacf_ax, lags=nlags, method="ywm", color=color, zero=False)
        acf_ax.set_title(f"ACF ({label})")
        pacf_ax.set_title(f"PACF ({label})")
        for correlation_ax in (acf_ax, pacf_ax):
            correlation_ax.set_xlabel("Lag (samples)")
            correlation_ax.set_ylabel("Correlation")
            correlation_ax.set_ylim(-1.05, 1.05)
            correlation_ax.grid(alpha=0.2)

    # axes[-1, 0].set_xlabel("Datetime")
    fig.suptitle(
        f"Regional Bz temporal variation and autocorrelation\n{storm_id}",
        fontsize=16,
        fontweight="bold",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{storm_id}_regional_Bz_ACF_PACF.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path

################################################
# read storm info data to get SSC, Main, and Recovery time
storminf = pd.read_csv("StormList_23events.csv")
for time_column in ("SSC_Stime", "Main_Stime", "Recov_Stime"):
    storminf[time_column] = pd.to_datetime(
        storminf[time_column], format="%m/%d/%y %H:%M", errors="coerce"
    )

paths = sorted(DATA_DIR.glob(FILE_GLOB))
if not paths:
    raise FileNotFoundError(f"No {FILE_GLOB} files found in {DATA_DIR}")
if len(paths) != len(storminf):
    raise ValueError(
        f"Found {len(paths)} storm CSVs but {len(storminf)} rows in "
        "StormList_23events.csv; cannot safely match storm times to files."
    )

for filepath, (_, storm) in zip(paths, storminf.iterrows()):
    storm_time = {
        "SSC": storm["SSC_Stime"],
        "Main": storm["Main_Stime"],
        "Recovery": storm["Recov_Stime"],
    }
    print(f'working on storm event {os.path.basename(filepath)}')
    print(f'storm starting time from storm_info file: SSC{storm["SSC_Stime"]}')
    output_path = plot_storm(filepath, storm_time)
    print(f"Saved {output_path}")
