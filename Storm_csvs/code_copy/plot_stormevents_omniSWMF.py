#plot_stormevents_omniSWMF.py
'''
Aim: to plot OMNI data and Bz values at 6 RE dayside from SWMF simulation to check the input data quality. 
Event list: StormList_23event.csv
@author: TsigeA
@date: Apr 26, 2026
'''
import os
from glob import glob
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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
# OMNI history length used as features
HISTORY_HOURS = 2     # similar spirit to the GBM paper
OMNI_STEP_MIN = 15  # SWMF cadence is 15 min, so use 15 min steps for lagged features to align with SWMF targets.
# Radius selection for target
TARGET_RADIUS_RE = 6.0
R_TOL = 0.25
DAYSIDE_ONLY = True   # X > 0
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

def extract_dayside_bz_at_6re(swmf_df: pd.DataFrame,
                              target_radius_re: float = 6.0,
                              tol: float = 0.25,
                              dayside_only: bool = True) -> pd.DataFrame:
    df = swmf_df.copy()
    df["r_re"] = np.sqrt(df["X_RE"] ** 2 + df["Y_RE"] ** 2)

    mask = np.abs(df["r_re"] - target_radius_re) <= tol

    if dayside_only:
        mask &= df["X_RE"] > 0

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
            target_Bz_dayside_6RE=("Bz", "mean"),
            n_points_target=("Bz", "size")
        )
        .sort_index()
        )
    return target

# ============================================================
# EVENT WINDOW RESTRICTION
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

for i in range(23):
    OMNI_FILE = OMNI_flist[i]
    SWMF_FILE = SWMF_flist[i]
    # OMNI_FILE = Path("omni_5min_20151218-20151225_BxyzPdynSYMH.txt")
    # SWMF_FILE = Path("z=0_var_2_e20151219-101300-000_20151221-041300-000Bz.txt")
    print (f"Processing OMNI file: {OMNI_FILE}")
    print (f"Processing SWMF file: {SWMF_FILE}")
    OUTPUT_CSV = f"{OMNI_FILE[10:-17]}_dayside_6RE_peak_1h_2h_withRecphase.csv"
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
        target_Qphase = extract_dayside_bz_at_6re(
            swmf_df=Quietphase,  
            target_radius_re=TARGET_RADIUS_RE,
            tol=R_TOL,
            dayside_only=DAYSIDE_ONLY)
        target_Qphase["phase"] = "Quiet"
    ########################################
    # Quietphase=swmf_df.loc[prestorm_Stime:(SSC_Stime-pd.Timedelta('1ns'))]
    SSCphase=swmf_df.loc[SSC_Stime:(Main_Stime-pd.Timedelta('1ns'))]
    Mainphase=swmf_df.loc[Main_Stime:(Recov_Stime-pd.Timedelta('1ns'))] #find the index for the coresponding time interval
    Recphase=swmf_df.loc[Recov_Stime:Recov_endtime]
    ##########
    # omni df set for each phase
    # omni_Quietphase=omni_df.loc[prestorm_Stime:(SSC_Stime-pd.Timedelta('1ns'))]
    omni_SSCphase=omni_df.loc[SSC_Stime:(Main_Stime-pd.Timedelta('1ns'))]
    omni_Mainphase=omni_df.loc[Main_Stime:(Recov_Stime-pd.Timedelta('1ns'))]
    omni_Recphase=omni_df.loc[Recov_Stime:Recov_endtime]
    # target_Qphase = extract_dayside_bz_at_6re(
    #     swmf_df=Quietphase,  
    #     target_radius_re=TARGET_RADIUS_RE,
    #     tol=R_TOL,
    #     dayside_only=DAYSIDE_ONLY
    # )
    # target_Qphase["phase"] = "Quiet"
    target_SSC = extract_dayside_bz_at_6re(
        swmf_df=SSCphase,  
        target_radius_re=TARGET_RADIUS_RE,
        tol=R_TOL,
        dayside_only=DAYSIDE_ONLY
    )   
    target_SSC["phase"] = "SSC" 
    target_Main = extract_dayside_bz_at_6re(
        swmf_df=Mainphase,  
        tol=R_TOL,
        dayside_only=DAYSIDE_ONLY
    )
    target_Main["phase"] = "Main"
    target_Rec = extract_dayside_bz_at_6re(
        swmf_df=Recphase,  
        tol=R_TOL,
        dayside_only=DAYSIDE_ONLY
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
    print(f"Peak time from OMNI SYMH minimum:{omni_sub['SYMH'].idxmin()} with value {omni_sub['SYMH'].min()} nT")

    #plotting OMNI SYMH and target Bz at 6 RE for the storm event with vertical lines indicating the SSC, main phase and recovery phase start times.

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(omni_sub.index, omni_sub["SYMH"], label="OMNI SYM-H", color="black")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("OMNI SYM-H [nT]", color="black", fontsize=14)
    ax1.tick_params(axis="y", labelcolor="black")
    ax1.grid(alpha=0.3, which="both")   
    ax1.legend(loc="upper left")
    ax2 = ax1.twinx()
    ax2.plot(target_sub.index, target_sub["target_Bz_dayside_6RE"], label="SWMF Bz at 6 RE dayside", color="orange")
    ax2.set_ylabel("SWMF Bz at 6 RE dayside [nT]", color="orange", fontsize=14)
    ax2.tick_params(axis="y", labelcolor="orange")
    ax2.grid(alpha=0.3, which="both")
    ax1.set_title(f"OMNI SYM-H and SWMF Bz at 6 RE dayside for storm event on {omni_sub.index[0].date()}", fontsize=16)
    ax1.vlines(SSC_Stime, ymin=omni_sub["SYMH"].min()-10, ymax=omni_sub["SYMH"].max()+2, colors="red", linewidth=2, label="SSC Phase")
    ax1.vlines(Main_Stime, ymin=omni_sub["SYMH"].min()-10, ymax=omni_sub["SYMH"].max()+2, colors="green", linewidth=2, label="Main Phase")
    ax1.vlines(Recov_Stime, ymin=omni_sub["SYMH"].min()-10, ymax=omni_sub["SYMH"].max()+2, colors="magenta", linewidth=2, label="Recovery Phase") 
    ax1.set_ylim(omni_sub["SYMH"].min()-10, omni_sub["SYMH"].max()+2 )
    ax2.set_ylim(target_sub["target_Bz_dayside_6RE"].min()-10, target_sub["target_Bz_dayside_6RE"].max()+10)
    plt.legend(loc="upper right")
    plt.xlim(start_time - pd.Timedelta(minutes=30), end_time)
    plt.tight_layout()
    plt.show()

    
