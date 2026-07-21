"""
Aim: to do stasionary test on the SWMF Bz data before applying AR model. 
- use the code from https://github.com/RezaSaadatyar/Time-Series-Analysis-in-Python/blob/main/Code/Test_Stationary.py
- Check Stationary Time Series: 
1)Rolling statistics: plot the moving average/variance and see if it varies with time. 
2) Augmented Dickey-Fuller Test: result[0]: When the test statistic is lower than the critical value shown, 
the time series is stationary result[1]: p-value >>>> If Test statistic < Critical Value and 
p-value < 0.05 >>>> the time series is stationary. 
Stationary means >>> mean, variance and covariance is constant over periods and auto-covariance that does not depend on time.

@autor: TsigeA
@date: Jul 13, 2026
"""
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa import stattools
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def test_stationary(data, window,filename=None):
    """
    param Data: Data is a ndarray and often 1 * N

    param window: Size of the moving window. If an integer, the fixed number of observations used for each window.
    If an offset, the time period of each window. Each window will be a variable sized based on the observations included
    in the time-period.

    return: Convert non-stationary data to stationary data if Data is Non-Stationary.

    Check Stationary Time Series: 1)Rolling statistics: plot the moving average/variance and see if it varies with
    time. 2) Augmented Dickey-Fuller Test: result[0]: When the test statistic is lower than the critical value shown,
    the time series is stationary result[1]: p-value >>>> If Test statistic < Critical Value and p-value < 0.05 >>>>
    the time series is stationary. Stationary means >>> mean, variance and covariance is constant over periods and
    auto-covariance that does not depend on time.

    Converting Non-stationary data to stationary dataset:
    Log: np.log(Data)
    Differencing simple moving average: MA = Data.rolling(window=window).mean()
    Data = Data - MA
    Data.dropna(inplace=True)
    """
    # ================================ Step 2: Check Stationary Time Series ========================================
    data1 = data
    # sns.set(style='white')
    sns.set_theme(style='white', palette='muted')
    result = stattools.adfuller(data)     # Perform Agumented Dickey-Fuller Test
    #result[0] = ADF test statistic, result[4] = dict of critical values at 1%/5%/10% confidence.
    #if the test statistic is more negative than the 5% critical value, the null hypothesis is rejected → series is stationary. Otherwise it's treated as non-stationary,
    if result[0] < result[4]["5%"]:
        fig, ax1 = plt.subplots(1, 1, sharey='row', figsize=(10, 6))
        plt.rcParams.update({'font.size': 11})
        ax1.set_title(filename[:17]+' Rolling Mean & Standard Deviation; ' + 'p-value:' + str(round(result[1], 3)) + '; Data is Stationary', fontsize=14)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharey='row', figsize=(10, 6))
        plt.rcParams.update({'font.size': 11})
        ax1.set_title(filename[:17]+' Rolling Mean & Standard Deviation; ' + 'p-value:' + str(round(result[1], 3)) + '; Data is Non-Stationary', fontsize=14)
        data = data - data.rolling(window=window).mean()                 # X.diff(periods=1)
        data.dropna(inplace=True)
        data.index = (np.linspace(0, len(data), num=len(data), endpoint=False, dtype='int'))
        data = pd.Series(data)
        result = stattools.adfuller(data)                                # Perform Dickey-Fuller Test
        if result[1] < 0.05:
            print('Data is Stationary after differencing')
            title = 'Rolling Mean & Standard Deviation; ' + 'p-value:' + str(round(result[1], 3)) + '; Data is Stationary after differencing'
        else:
            print('Data is Non-Stationary after differencing')
            title = 'Rolling Mean & Standard Deviation; ' + 'p-value:' + str(round(result[1], 3)) + '; Data is Non-Stationary after differencing'
        ax2.plot(data)
        ax2.plot(data.rolling(window=window).mean())                     # Determine rolling statistics
        ax2.plot(data.rolling(window=window).std())
        ax2.set_title(title, fontsize=14)
    output_result = pd.Series(result[0:4], index=['Test Statistic', 'p-value', '#lags used', 'number of observations used']) #p < 0.05 implies stationarity
    for key, value in result[4].items():
        output_result['critical value (%s)' % key] = value
    print(output_result)
    ax1.plot(data1, label='Data')
    ax1.plot(data1.rolling(window=window).mean(), label='Rolling Mean')   # Determine rolling statistics
    ax1.plot(data1.rolling(window=window).std(), label='Rolling Std')
    ax1.set_title(filename[:17]+' Rolling Mean & Standard Deviation', fontsize=14)
    ax1.legend(loc='best'), plt.tight_layout(), plt.show()
    return data

# ----------------------------
# User settings
# ----------------------------
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
DATA_DIR = "Storm_csv_nolag"
FILE_GLOB = "*.csv"

DATETIME_COL = "datetime"

TARGET_COL = "target_Bz_dayside_6RE_t0"

TARGET_1H = "target_Bz_dayside_6RE_tplus_60m"
TARGET_2H = "target_Bz_dayside_6RE_tplus_120m"

# read the csv files and apply the test_stationary function to the target column
import glob
import os
csv_files = sorted(glob.glob(os.path.join(DATA_DIR, FILE_GLOB)))
concat_data = []
output_dir = "Stationary_test_results"
os.makedirs(output_dir, exist_ok=True) 

for csv_file in csv_files:
    df = pd.read_csv(csv_file, parse_dates=[DATETIME_COL])
    df.set_index(DATETIME_COL, inplace=True)
    # print(f"Processing file: {csv_file}")
    concat_data.append(df[TARGET_COL])
    # plot ACF and PACF for the target column
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(df[TARGET_COL].dropna(), ax=axes[0], lags=50)
    plot_pacf(df[TARGET_COL].dropna(), ax=axes[1], lags=50)
    axes[0].set_title(f"ACF of {TARGET_COL} for {os.path.basename(csv_file)[:17]}")
    axes[1].set_title(f"PACF of {TARGET_COL} for {os.path.basename(csv_file)[:17]}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"ACF_PACF_{os.path.basename(csv_file)[:17]}.png"))

concat_data = pd.concat(concat_data, axis=0)
concat_data.reset_index(drop=True, inplace=True)
print(f"Concatenated data from {len(csv_files)} storm events: {len(concat_data)} total observations")

# do the stationary test on the concatenated data
stationary_data = test_stationary(concat_data, window=14, filename="All_23_Storms")  # Using a window of 14 for rolling statistics