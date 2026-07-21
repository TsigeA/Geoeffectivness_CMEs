'''
Aim: to prepare a training table for predicting nightside Bz at 6 RE using OMNI data as features and SWMF Bz as targets, focusing on the storm peak period.
It performs the following steps:
1. Reads and processes the OMNI data, including handling fill values and computing derived quantities like Ey and Es.
2. Reads the SWMF Bz data and extracts the mean Bz at 6 RE on the nightside.
3. Restricts the data to a specified time window and identifies the storm peak time based on the minimum SYM-H value.
4. Builds a training table with lagged OMNI features and future Bz targets at specified forecast horizons, 
ensuring that only rows with valid future targets up to the storm peak time are included.
5. Saves the resulting training table to a CSV file.

@author: TsigeA
@date: Jul 21 2026
update: 
    - Jul 21 2026: added function to exract nightside Bz at 6 RE to use for ARx model and updated the grid tolerance from 2*0.25 in y axix to 0.25 for both x and y axis 

'''
import os
from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# USER SETTINGS
# ============================================================
# all OMNI files
OMNI_flist = sorted(glob("omni_5min_*.txt"))
# all SWMF files
SWMF_flist = sorted(glob("z=0_var_2_e*.txt"))
print(f"OMNI files found: {len(OMNI_flist)}")
print(f"SWMF files found: {len(SWMF_flist)}")
# storm list with quiet, SSC, peak and recovery period info
storminf=pd.read_csv("StormList_23events.csv")
prestorm=storminf['prestorm_Stime']
SSC=storminf['SSC_Stime']
Main=storminf['Main_Stime']
Recov=storminf['Recov_Stime']
Recov_end=storminf['Recov_endtime']

# Radius selection for target
TARGET_RADIUS_RE = 6.0
R_TOL = 0.5
NIGHTSIDE_ONLY= True # X < 0

# OMNI history length used as features
HISTORY_HOURS = 0     # 2 similar spirit to the GBM paper. 0 is used here since we are not using lagged features for the AR model training.
OMNI_STEP_MIN = 15  # SWMF cadence is 15 min, so use 15 min steps for lagged features to align with SWMF targets.

# Forecast horizons
HORIZONS_MIN = [60, 120]   # 1 hr and 2 hr

# Past SWMF Bz lag features
# Analogous to past SYM-H in Iong et al. (2022): use the previous 1 hr of the
# target variable itself as input features. Lag columns are named Bz_6RE_lag_{k}m
# to match the OMNI naming convention; they are picked up automatically by the
# training script as regular feature columns (not targets).
PAST_BZ_HISTORY_HOURS = 1   # history window in hours (1 hr = 4 steps at 15-min cadence)
# # Optional event window; set to None to use full file overlap
# START_TIME = None
# END_TIME = None

# If True, use GSM By/Bz
USE_GSM = True

# ============================================================
# READ OMNI
# ============================================================

def read_omni_5min(filepath: Path) -> pd.DataFrame:
    rows = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) < 13:
                continue

            year = int(parts[0])
            doy = int(parts[1])
            hr = int(parts[2])
            mn = int(parts[3])
            vals = list(map(float, parts[4:13]))

            rows.append([
                year, doy, hr, mn,
                vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
                vals[6], vals[7], vals[8]
            ])

    cols = [
        "year", "doy", "hour", "minute",
        "Bmag", "BX", "BY_GSE", "BZ_GSE", "BY_GSM", "BZ_GSM",
        "V", "Pdyn", "SYMH"
    ]
    df = pd.DataFrame(rows, columns=cols)

    df["datetime"] = (
        pd.to_datetime(df["year"].astype(str), format="%Y") +
        pd.to_timedelta(df["doy"] - 1, unit="D") +
        pd.to_timedelta(df["hour"], unit="h") +
        pd.to_timedelta(df["minute"], unit="m")
    )

    # replace fill values with NaN
    for c in ["Bmag", "BX", "BY_GSE", "BZ_GSE", "BY_GSM", "BZ_GSM", "V", "Pdyn", "SYMH"]:
        df.loc[np.isclose(df[c], 99999.9), c] = np.nan
        df.loc[np.isclose(df[c], 9999.99), c] = np.nan
        df.loc[np.isclose(df[c], 999.99), c] = np.nan
        df.loc[np.isclose(df[c], 99.99), c] = np.nan

    df = df.set_index("datetime").sort_index()
    

    numeric_cols = ["Bmag", "BX", "BY_GSE", "BZ_GSE", "BY_GSM", "BZ_GSM", "V", "Pdyn", "SYMH"]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both") # linear interpolation to fill NaNs, limit_direction="both" allows filling at the start and end of the series

    by = df["BY_GSM"] if USE_GSM else df["BY_GSE"]
    bz = df["BZ_GSM"] if USE_GSM else df["BZ_GSE"]

    df["BY_used"] = by
    df["BZ_used"] = bz

    # derived quantities 
    #y component of the interplanetary electric field Ey = VxBz 
    # characterizes the amount of north-south magnetic flux carried by the solar wind
    # Ey in mV/m when V is km/s and B is nT
    df["Ey"] = -df["V"] * df["BZ_used"] * 1e-3
    df["Es"] = np.maximum(df["Ey"], 0.0) # max(0,Ey) dayside reconnection electric field proxy
    # calculate the clock angle 
    df["theta"] = np.arctan2(by, bz) # in radians
    # calculate Newell coupling function as another solar wind-magnetosphere coupling proxy
    # C = V^(4/3) * B^(2/3) * sin^8/3(theta/2)
    df["Newell_Coupling"] = (df["V"] ** (4/3)) * (df["Bmag"] ** (2/3)) * (np.sin(df["theta"] / 2) ** (8/3))
    # add other doy  or seasonal features 
    # df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365.25)
    # df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365.25)
    df = df.drop(columns=["year", "doy", "hour", "minute","BY_GSE", "BZ_GSE", "BY_GSM", "BZ_GSM"]) # drop original time columns and unused B components
    return df


# ============================================================
# READ SWMF
# ============================================================

def read_swmf_bz(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        names=["date", "time", "X_RE", "Y_RE", "Bz"]
    )
    # df = pd.read_csv(filepath,delim_whitespace=True,skiprows=1,names=["date", "time", "X_RE", "Y_RE", "Bz"] ,parse_dates={'Timestamp':[0,1]},index_col=0)
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str).str.zfill(6),
        format="%Y%m%d %H%M%S"
    )
    df = df.set_index("datetime").sort_index()
    # df.sort_values("datetime").reset_index(drop=True)
    # df=df.sort_index(inplace=True) # sort by datetime index
    df = df.drop(columns=["date", "time"]) # drop the original date and time columns since we have the datetime index now
    return df 


def extract_nightside_bz_at_6re(swmf_df: pd.DataFrame,
                              target_radius_re: float = 6.0,
                              tol: float = 0.25,
                              nightside_only: bool =True) -> pd.DataFrame:
    df = swmf_df.copy()
    df["r_re"] = np.sqrt(df["X_RE"] ** 2 + df["Y_RE"] ** 2)

    mask = np.abs(df["r_re"] - target_radius_re) <= tol

    if nightside_only:
        mask &= df["X_RE"] < 0
        mask &= (df["Y_RE"] > -(0.125+tol))&(df["Y_RE"]<0.125+tol) # correction added

    subset = df[mask].copy()

    if subset.empty:
        raise ValueError(
            f"No SWMF points found near r={target_radius_re} RE "
            f"with tolerance {tol} on the selected side."
        )

    # target = (
    #     subset.groupby("datetime", as_index=False)
    #     .agg(
    #         target_Bz_dayside_6RE=("Bz", "mean"),
    #         n_points_target=("Bz", "size")
    #     )
    #     .sort_values("datetime")
    #     .reset_index(drop=True)
    # )
    target = (
        subset.groupby("datetime")
        .agg(
            target_Bz_nightside_6RE=("Bz", "mean"),
            n_points_target=("Bz", "size")
        )
        .sort_index()
        )
    return target

# ============================================================
# EVENT WINDOW AND STORM PEAK
# ============================================================

def restrict_time_window(omni_df: pd.DataFrame,
                         target_df: pd.DataFrame,
                         start_time=None,
                         end_time=None,
                         history_hours: int = 0,
                         step_min: int = 5):
    # overlap first
    # overlap_start = max(omni_df["datetime"].min(), target_df["datetime"].min())
    # overlap_end = min(omni_df["datetime"].max(), target_df["datetime"].max())
    overlap_start = max(omni_df.index.min(), target_df.index.min())
    overlap_end = min(omni_df.index.max(), target_df.index.max())
    if overlap_start > overlap_end:
        raise ValueError("No overlapping time interval between OMNI and SWMF data.")

    if start_time is None:
        start_time = overlap_start
    # else:
    #     start_time = max(pd.to_datetime(start_time), overlap_start)

    if end_time is None:
        end_time = overlap_end
    # else:
    #     end_time = min(pd.to_datetime(end_time), overlap_end)

    # omni_sub = omni_df[(omni_df["datetime"] >= start_time) & (omni_df["datetime"] <= end_time)].copy()
    # Keep extra OMNI history before the target window so lagged features can be built.
    # Include one extra cadence step so "asof" can still find the prior sample
    # when the target timestamps fall between OMNI timestamps.
    omni_start = start_time - pd.Timedelta(hours=history_hours) - pd.Timedelta(minutes=step_min)
    omni_start = max(omni_start, omni_df.index.min())

    # omni_sub = omni_df[(omni_df["datetime"] >= omni_start) & (omni_df["datetime"] <= end_time)].copy()
    # target_sub = target_df[(target_df["datetime"] >= start_time) & (target_df["datetime"] <= end_time)].copy()

    omni_sub = omni_df[(omni_df.index >= omni_start) & (omni_df.index <= end_time)].copy()
    target_sub = target_df[(target_df.index >= start_time) & (target_df.index <= end_time)].copy()

    return omni_sub, target_sub, start_time, end_time


def find_storm_peak_time(omni_df: pd.DataFrame) -> pd.Timestamp:
    """Use minimum SYMH as storm peak."""
    idx = omni_df["SYMH"].idxmin()
    return omni_df.loc[idx, "datetime"]


# ============================================================
# FEATURE TABLE
# ============================================================

def build_feature_vector(omni_df: pd.DataFrame,
                         current_time: pd.Timestamp,
                         history_hours: int = 2,
                         step_min: int = 5) -> dict:
    """
    Build lagged OMNI features ending at current_time.
    """
    n_lags = int(history_hours * 60 / step_min)

    row = {"datetime": current_time}
    row["sin_doy"] = np.sin(2 * np.pi * current_time.dayofyear / 365.25)
    row["cos_doy"] = np.cos(2 * np.pi * current_time.dayofyear / 365.25)
    # if USE_GSM:
    #     base_cols = ["Bmag", "BX", "BY_GSM", "BZ_GSM", "V", "Pdyn", "SYMH", "Ey", "Es"]
    # else:
    #     base_cols  = ["Bmag", "BX", "BY_GSE", "BZ_GSE", "V", "Pdyn", "SYMH", "Ey", "Es"]
    # base_cols = ["Bmag", "BX", "BY_used", "BZ_used", "V", "Pdyn", "SYMH", "Ey", "Es"]
    base_cols = ["Bmag", "BX", "BY_used", "BZ_used", "V", "Pdyn", "SYMH", "Ey", "Es","theta","Newell_Coupling"]
    # omni_indexed = omni_df.set_index("datetime")
    # omni_indexed = omni_df.sort_values("datetime").set_index("datetime")
    # omni_times = omni_indexed.index
    omni_indexed = omni_df
    omni_times = omni_indexed.index

    for lag in range(n_lags + 1):
        t_lag = current_time - pd.Timedelta(minutes=lag * step_min)
        matched_time = omni_times.asof(t_lag) # find the most recent OMNI timestamp at or before t_lag

        # if t_lag not in omni_indexed.index:
        if pd.isna(matched_time) or (t_lag - matched_time) > pd.Timedelta(minutes=step_min): # if no OMNI timestamp at or before t_lag, or if the closest one is too far in the past (more than one step), then we consider it missing
            return None

        #vals = omni_indexed.loc[t_lag]
        vals= omni_indexed.loc[matched_time] # use the matched_time to get the OMNI values, which is the most recent time at or before t_lag

        for col in base_cols:
            if lag == 0:
                row[f"{col}"] = vals[col] # to avoid lag in the variable name for the current time value, we just use the base column name without "_lag_0m"
            else:
                row[f"{col}_lag_{lag * step_min}m"] = vals[col]

    return row


def build_training_table(omni_df: pd.DataFrame,
                         target_df: pd.DataFrame,
                         Recov_endtime: pd.Timestamp,
                         history_hours: int = 2,
                         step_min: int = 5,
                         horizons_min=(30,60, 120)) -> pd.DataFrame:
    """
    Build one row per SWMF target timestamp.
    Keep only rows whose future targets remain <= peak_time.
    update: include the recovery phase in the training table, so we will keep all rows with valid future targets up to the end of the recovery phase.
    Each lag uses the most recent OMNI sample at or before the requested lag time.
    """
    # target_indexed = target_df.set_index("datetime")
    target_indexed = target_df # already indexed by datetime
    omni_indexed = omni_df.sort_index()
    rows = []

    for t in target_df.index:
        valid = True # flag to check if all future targets are valid (i.e., exist and are <= peak_time or Recov_endtime in the updated version)
        target_vals = {}

        for h in horizons_min:
            t_future = t + pd.Timedelta(minutes=h)
            if t_future > Recov_endtime:
                valid = False
                break
            if t_future not in target_indexed.index:
                valid = False
                break
            target_vals[f"target_Bz_nightside_6RE_tplus_{h}m"] = target_indexed.loc[t_future, "target_Bz_nightside_6RE"]

        if not valid:
            continue

        feat = build_feature_vector(
            omni_df=omni_indexed,
            current_time=t,
            history_hours=history_hours,
            step_min=step_min
        )
        if feat is None:
            continue

        feat["target_Bz_nightside_6RE_t0"] = target_indexed.loc[t, "target_Bz_nightside_6RE"]
        feat["n_points_target"] = target_indexed.loc[t, "n_points_target"]
        feat["phase"] = target_indexed.loc[t, "phase"]
        # feat["doy"] = t.dayofyear # already have doy from the OMNI features

        feat.update(target_vals)
        rows.append(feat)
      

    # out = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)
    out = pd.DataFrame(rows).sort_index()
    return out


# ============================================================
# PAST SWMF BZ LAGS
# ============================================================

def add_swmf_bz_lags(
    df: pd.DataFrame,
    history_hours: int = 1,
    step_min: int = 15,
) -> pd.DataFrame:
    """
    Append past SWMF Bz lag columns to the training table.

    Creates Bz_6RE_lag_0m (= current t0 value) and Bz_6RE_lag_{k}m for each
    step k back in time, using the same naming convention as OMNI lag columns.
    Rows at the start of the table that lack sufficient history are dropped.

    Since each call operates on a single-storm CSV, there is no cross-storm
    leakage from pd.shift().
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    n_lags = int(history_hours * 60 / step_min)

    df["Bz_6RE_lag_0m"] = df["target_Bz_dayside_6RE_t0"]

    lag_cols = []
    for lag in range(1, n_lags + 1):
        col = f"Bz_6RE_lag_{lag * step_min}m"
        df[col] = df["target_Bz_dayside_6RE_t0"].shift(lag)
        lag_cols.append(col)

    n_before = len(df)
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    print(
        f"  [swmf_lags] {len(lag_cols) + 1} Bz_6RE_lag columns added "
        f"({history_hours}h history, {step_min}-min step). "
        f"Dropped {n_before - len(df)} rows with insufficient history."
    )
    return df


# ============================================================
# MAIN
# ============================================================
for i in range(len(OMNI_flist)):
    OMNI_FILE = OMNI_flist[i]
    SWMF_FILE = SWMF_flist[i]
    # OMNI_FILE = Path("omni_5min_20151218-20151225_BxyzPdynSYMH.txt")
    # SWMF_FILE = Path("z=0_var_2_e20151219-101300-000_20151221-041300-000Bz.txt")
    print (f"Processing OMNI file: {OMNI_FILE}")
    print (f"Processing SWMF file: {SWMF_FILE}")
    # OUTPUT_CSV = f"{OMNI_FILE[10:-17]}_dayside_6RE_peak_1h_2h_nolagg.csv"
    OUTPUT_CSV = f"{OMNI_FILE[10:-17]}_nightside_6RE_peak_1h_2h_nolagg.csv"
    omni_df = read_omni_5min(OMNI_FILE)
    omni_df.head()
    swmf_df = read_swmf_bz(SWMF_FILE)
    swmf_df.head()

    prestorm_Stime=pd.to_datetime(prestorm[i])
    SSC_Stime=pd.to_datetime(SSC[i])
    Main_Stime=pd.to_datetime(Main[i])
    Recov_Stime=pd.to_datetime(Recov[i])
    Recov_endtime=pd.to_datetime(Recov_end[i])
    ###################
    START_TIME = prestorm_Stime
    # END_TIME = Recov_Stime # since we want to focus on up to the storm peak period for now. we will use data from recovery period for the prediction of the recovery phase in the future. 
    END_TIME = Recov_endtime # use the end of the recovery period as the end time for now, which will include some data after the storm peak. we can always restrict it later if needed.
    ###########
    #SWMF data sets for each phase
    # some of the events don't have quiet phase data in the SWMF output, so we will just use whatever data is available for each phase.
    if pd.isna(prestorm_Stime) and SSC_Stime < swmf_df.index.min():
        print(f"Warning: prestorm_Stime is missing and SSC_Stime {SSC_Stime} is before the start of the SWMF data {swmf_df.index.min()}. \n Skipping quiet phase for this event and adjusting SSC_Stime to match SWMF data start time.")
        Quietphase=None
        omni_Quietphase=None
        SSC_Stime = swmf_df.index.min()
        target_Qphase = None
        START_TIME = SSC_Stime
    elif pd.isna(prestorm_Stime) and SSC_Stime >= swmf_df.index.min():
        print(f"Warning: prestorm_Stime is missing. Skipping quiet phase for this event.")
        Quietphase=None
        omni_Quietphase=None
        target_Qphase = None
        START_TIME = SSC_Stime
    # elif prestorm_Stime < swmf_df.index.min():
    #     print(f"Warning: prestorm_Stime {prestorm_Stime} is before the start of the SWMF data {swmf_df.index.min()}. Adjusting prestorm_Stime to match SWMF data start time.")  
    #     prestorm_Stime = swmf_df.index.min()
    else:
        Quietphase=swmf_df.loc[prestorm_Stime:(SSC_Stime-pd.Timedelta('1ns'))]
        omni_Quietphase=omni_df.loc[prestorm_Stime:(SSC_Stime-pd.Timedelta('1ns'))]
        target_Qphase = extract_nightside_bz_at_6re(
            swmf_df=Quietphase,  
            target_radius_re=TARGET_RADIUS_RE,
            tol=R_TOL,
            nightside_only=NIGHTSIDE_ONLY)
        target_Qphase["phase"] = "Quiet"
    ########################################
    # Quietphase=swmf_df.loc[prestorm_Stime:(SSC_Stime-pd.Timedelta('1ns'))]
    SSCphase=swmf_df.loc[SSC_Stime:(Main_Stime-pd.Timedelta('1ns'))]
    Mainphase=swmf_df.loc[Main_Stime:(Recov_Stime-pd.Timedelta('1ns'))] #find the index for the coresponding time interval
    Recphase=swmf_df.loc[Recov_Stime:Recov_endtime]
    ##########
    # omni df set for each phase
    # omni_Quietphase=omni_df.loc[prestorm_Stime:(SSC_Stime-pd.Timedelta('1ns'))]
    # omni_SSCphase=omni_df.loc[SSC_Stime:(Main_Stime-pd.Timedelta('1ns'))]
    # omni_Mainphase=omni_df.loc[Main_Stime:(Recov_Stime-pd.Timedelta('1ns'))]
    # omni_Recphase=omni_df.loc[Recov_Stime:Recov_endtime]
    # target_Qphase = extract_dayside_bz_at_6re(
    #     swmf_df=Quietphase,  
    #     target_radius_re=TARGET_RADIUS_RE,
    #     tol=R_TOL,
    #     dayside_only=DAYSIDE_ONLY
    # )
    # target_Qphase["phase"] = "Quiet"  
    target_SSC = extract_nightside_bz_at_6re(
        swmf_df=SSCphase,  
        target_radius_re=TARGET_RADIUS_RE,
        tol=R_TOL,
        nightside_only=NIGHTSIDE_ONLY
    ) 
    target_SSC["phase"] = "SSC" 
    target_Main = extract_nightside_bz_at_6re(
        swmf_df=Mainphase,  
        target_radius_re=TARGET_RADIUS_RE,
        tol=R_TOL,
        nightside_only=NIGHTSIDE_ONLY
    )
    target_Main["phase"] = "Main"

    target_Rec = extract_nightside_bz_at_6re(
        swmf_df=Recphase,  
        target_radius_re=TARGET_RADIUS_RE,
        tol=R_TOL,
        nightside_only=NIGHTSIDE_ONLY
    )
    target_Rec["phase"] = "Recovery"
    #################
    # concatenate the target dataframes for different phases
    target_df = pd.concat([target_Qphase, target_SSC, target_Main, target_Rec]).sort_index()
    # target_df = target_df.sort_values("datetime").reset_index(drop=True)
    omni_sub, target_sub, start_time, end_time = restrict_time_window(
        omni_df, target_df,
        start_time=START_TIME,
        end_time=END_TIME,
        history_hours=HISTORY_HOURS,
        step_min=OMNI_STEP_MIN
    )

    # peak_time = find_storm_peak_time(omni_sub)
    peak_time= Recov_Stime # beigning of the recovery phase is same as the storm peak time. 
    print("Using time window:")
    print("Start:", start_time)
    print("End  :", end_time)
    # print("Peak time from OMNI SYMH minimum:", peak_time)
    print()

    # print("OMNI cadence check:")
    # print(omni_sub.index.diff().dropna().value_counts().head())
    # print()

    # print("SWMF target cadence check:")
    # print(target_sub.index.diff().dropna().value_counts().head())
    # print()

    train_df = build_training_table(
        omni_df=omni_sub,
        target_df=target_sub,
        Recov_endtime=Recov_endtime,
        history_hours=HISTORY_HOURS,
        step_min=OMNI_STEP_MIN,
        horizons_min=HORIZONS_MIN
    )

    # print("Training table shape:", train_df.shape)
    # print(train_df.head())
    # uncomment the following line to add past SWMF Bz lag features to the training table. 
    # train_df = add_swmf_bz_lags(
    #     train_df,
    #     history_hours=PAST_BZ_HISTORY_HOURS,
    #     step_min=OMNI_STEP_MIN,
    # )

    train_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved training table to: {OUTPUT_CSV}")

