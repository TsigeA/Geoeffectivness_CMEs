'''
Aim: To plot the model results.
Input: csv files with test_predictions tplus 60 and tplus 120.
    - validation_predictions_tplus_60m.csv
    - validation_predictions_tplus_120m.csv
Output: Plots of the predictions vs actual values for both tplus 60 and tplus 120.
'''

import os
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
# read storm_list.csv to get the storm phase information
storm_list_df = pd.read_csv('StormList_23events.csv')
# read the csv files
tplus_60_df = pd.read_csv('test_predictions_tplus_60m_20260428_124055.csv')
tplus_120_df = pd.read_csv('test_predictions_tplus_120m_20260428_124055.csv')
# plot the results for only one storm at a time
# find all the unique storm ids in the tplus 60 dataframe
unique_storm_ids = tplus_60_df['storm_id'].unique()
print(unique_storm_ids)
for storm_id in unique_storm_ids:
    print(storm_id)
    storm_df60 = tplus_60_df[tplus_60_df['storm_id'] == storm_id]
    # convert the datetime column to datetime format
    storm_df60['datetime'] = pd.to_datetime(storm_df60['datetime'])
    # use the storm datetime to get the storm phase information from the storm_list_df
    storm_datetime = storm_df60['datetime'].iloc[0].date()
    storm_list_df['prestorm_Stime'] = pd.to_datetime(storm_list_df['prestorm_Stime'])
    storm_list_df['SSC_Stime'] = pd.to_datetime(storm_list_df['SSC_Stime'])
    storm_list_df['Main_Stime'] = pd.to_datetime(storm_list_df['Main_Stime'])
    storm_list_df['Recov_Stime'] = pd.to_datetime(storm_list_df['Recov_Stime'])
    storm_list_df['Recov_endtime'] = pd.to_datetime(storm_list_df['Recov_endtime'])
    # filter the storm_list_df to get the storm phase information for the current storm
    # storm_phase_info = storm_list_df[storm_list_df['prestorm_Stime'].dt.date == storm_datetime]
    storm_phase_info = storm_list_df[storm_list_df['SSC_Stime'].dt.date == storm_datetime]
    # plot the predictions vs actual values for tplus 60
    ymin60, ymax60 = min(storm_df60['y_true'].min(), storm_df60['y_pred'].min()), max(storm_df60['y_true'].max(), storm_df60['y_pred'].max())
    ymin120, ymax120 = min(storm_df60['y_true'].min(), storm_df60['y_pred'].min()), max(storm_df60['y_true'].max(), storm_df60['y_pred'].max())
    fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax2 = ax[0].twinx()
    ax[0].plot(storm_df60['datetime'], storm_df60['y_true'],'k', label='SWMF')
    ax[0].plot(storm_df60['datetime'], storm_df60['y_pred'],color='blue', label='Predicted t= +60 minutes')
    ax2.plot(storm_df60['datetime'], storm_df60['residual'],'r', label='Residuals',linewidth=0.5,alpha=0.8)
    ax2.set_ylabel('Residuals [nT]', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(-100, 100)
    ax2.grid(alpha=0.3, which='both')
    ax[0].set_title(f'{storm_df60["datetime"].iloc[0]}: Predictions vs Actual Values SYM-H = {storm_phase_info["SYM-H"].values[0]}nT', fontsize=16)
    # ax[0].set_xlabel('Time')
    ax[0].vlines(storm_phase_info['SSC_Stime'], ymin=ymin60-1, ymax=ymax60+1, colors='red', linewidth=2)
    ax[0].vlines(storm_phase_info['Main_Stime'], ymin=ymin60-1, ymax=ymax60+1, colors='green', linewidth=2)
    ax[0].vlines(storm_phase_info['Recov_Stime'], ymin=ymin60-1, ymax=ymax60+1, colors='magenta', linewidth=2)
    ax[0].set_ylim(ymin60-1, ymax60+1)
    ax[0].set_ylabel('Bz[nT]',fontsize=14)
    ax[0].legend()
    ax[0].set_xlim(storm_df60['datetime'].min(), storm_df60['datetime'].max())
    # plot the predictions vs actual values for tplus 120
    storm_df120 = tplus_120_df[tplus_120_df['storm_id'] == storm_id]
    storm_df120['datetime'] = pd.to_datetime(storm_df120['datetime'])
    ax2 = ax[1].twinx()
    ax[1].plot(storm_df120['datetime'], storm_df120['y_true'],'k', label='SWMF')
    ax[1].plot(storm_df120['datetime'], storm_df120['y_pred'],color='blue', label='Predicted t= +120 minutes')
    ax2.plot(storm_df120['datetime'], storm_df120['residual'],'r', label='Residuals',linewidth=0.5,alpha=0.8)
    ax2.set_ylabel('Residuals [nT]', color='r', fontsize=14)
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(-100, 100)
    ax2.grid(alpha=0.3, which='both')
    # ax[1].set_title(f'{storm_df120["datetime"].iloc[0]}: Predictions vs Actual Values for tplus 120', fontsize=16)
    ax[1].vlines(storm_phase_info['SSC_Stime'], ymin=ymin120-1, ymax=ymax120+1, colors='red', linewidth=2)
    ax[1].vlines(storm_phase_info['Main_Stime'], ymin=ymin120-1, ymax=ymax120+1, colors='green', linewidth=2)
    ax[1].vlines(storm_phase_info['Recov_Stime'], ymin=ymin120-1, ymax=ymax120+1, colors='magenta', linewidth=2)
    ax[1].set_ylabel('Bz[nT]',fontsize=14)
    ax[1].set_ylim(ymin120-1, ymax120+1)
    ax[1].legend()
    ax[1].set_xlim(storm_df120['datetime'].min(), storm_df120['datetime'].max())
    plt.tight_layout()
    plt.show()

