# -*- coding: utf-8 -*-
"""
#Created on Tue Mar 15 2022

@author: Administrator
"""


#%% loading and processing newly uploaded NBM data:

import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy
import scipy.io
# import scipy.linalg
import pandas as pd
import logging
import gym
from gym import spaces, logger
from gym.utils import seeding
from pathlib import Path


rawNBMdf_pt1 = pd.read_csv('full_combined_report_pt1.csv')
rawNBMdf_pt2 = pd.read_csv('full_combined_report_pt2.csv')
filteredNBMdf = pd.concat([rawNBMdf_pt1, rawNBMdf_pt2])
serialanon = np.unique(filteredNBMdf['serialanon'].to_numpy())
filteredNBMdf.set_index(['serialanon', 'year'], inplace=True)

filteredNBMdf["Total CO2 emissions (kg/year)"] = filteredNBMdf["272 Total CO2 emissions"] + filteredNBMdf["383 Total CO2 emissions"]
filteredNBMdf["Total energy all uses (kWh/year)"] = filteredNBMdf["238 Annual delivered total energy all uses"] + filteredNBMdf["338 Annual delivered total energy all uses"]
filteredNBMdf["Electricity use (60% of total kWh/year)"] = filteredNBMdf["Total energy all uses (kWh/year)"]*0.6
filteredNBMdf["Gas use (40% of total kWh/year)"] = filteredNBMdf["Total energy all uses (kWh/year)"]*0.4
filteredNBMdf.drop(columns=['238 Annual delivered total energy all uses', 
    '338 Annual delivered total energy all uses',
    '272 Total CO2 emissions',
    '383 Total CO2 emissions'], inplace=True)

filteredNBMdf["walls/wall_area"].fillna(0, inplace=True)
filteredNBMdf["Total wall insulation area (m^2)"] = 0.0
for i in serialanon:
    filteredNBMdf.loc[[(i,1001)],"Total wall insulation area (m^2)"] = filteredNBMdf.loc[[(i,1001)],"walls/wall_area"].sum()
    filteredNBMdf.loc[[(i,1011)],"Total wall insulation area (m^2)"] = filteredNBMdf.loc[[(i,1011)],"walls/wall_area"].sum()
    filteredNBMdf.loc[[(i,1101)],"Total wall insulation area (m^2)"] = filteredNBMdf.loc[[(i,1101)],"walls/wall_area"].sum()
    filteredNBMdf.loc[[(i,1111)],"Total wall insulation area (m^2)"] = filteredNBMdf.loc[[(i,1111)],"walls/wall_area"].sum()

filteredNBMdf.loc[filteredNBMdf["custom/heat_pump_size_req"] == -1, "custom/heat_pump_size_req"] = 0.0
filteredNBMdf["Installed heat pumps power in Megawatts"] = 0.0
# filteredNBMdf.loc[list(zip(serialanon,np.full(len(serialanon),1100))),"Installed heat pumps power in Megawatts"] = filteredNBMdf.loc[list(zip(serialanon,np.full(len(serialanon),1100))),"custom/heat_pump_size_req"]/1000.0
# filteredNBMdf.loc[list(zip(serialanon,np.full(len(serialanon),1101))),"Installed heat pumps power in Megawatts"] = filteredNBMdf.loc[list(zip(serialanon,np.full(len(serialanon),1101))),"custom/heat_pump_size_req"]/1000.0
# filteredNBMdf.loc[list(zip(serialanon,np.full(len(serialanon),1110))),"Installed heat pumps power in Megawatts"] = filteredNBMdf.loc[list(zip(serialanon,np.full(len(serialanon),1110))),"custom/heat_pump_size_req"]/1000.0
# filteredNBMdf.loc[list(zip(serialanon,np.full(len(serialanon),1111))),"Installed heat pumps power in Megawatts"] = filteredNBMdf.loc[list(zip(serialanon,np.full(len(serialanon),1111))),"custom/heat_pump_size_req"]/1000.0
filteredNBMdf.loc[filteredNBMdf["other/measures_installed"].str.contains(
    'measure_air_source_heat_pump', case=False, na=False),"Installed heat pumps power in Megawatts"] = filteredNBMdf.loc[
        filteredNBMdf["other/measures_installed"].str.contains(
            'measure_air_source_heat_pump', case=False, na=False),"custom/heat_pump_size_req"]/1000.0

filteredNBMdf_singleRow = filteredNBMdf[filteredNBMdf["index"] == 1]

filteredNBMdf_singleRow.to_csv('processedNBMsingleRow.csv')
filteredNBMdf.to_csv('processedNBM.csv')

