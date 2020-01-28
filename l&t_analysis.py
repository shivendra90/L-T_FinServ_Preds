#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:47:33 2020

@author: shivendra
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor, XGBRFRegressor
from yellowbrick.regressor import residuals_plot, prediction_error
from yellowbrick.features import rank2d, rank1d
from yellowbrick.model_selection import RFECV, ValidationCurve, LearningCurve
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV, SGDRegressor, PassiveAggressiveRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

from fastai.collab import add_datepart
plt.ion()

"""
Part 1
Data Loading and Exploration
"""

prim_data = pd.read_csv("train.csv", engine="c")
test = pd.read_csv("test.csv", engine="c")
target = prim_data["case_count"]
cols = list(prim_data.columns)

prim_data["application_date"] = pd.to_datetime(prim_data.application_date,
                                               format="%Y-%m-%d").copy()  # Copy to prevent warning

print("Number of missing values out of {0} rows/samples: \n{1}".format(len(prim_data),
                                                                       prim_data.isna().sum()))

msno.bar(prim_data, figsize=(10,5), fontsize=12)  # Plot columns with missing values

# For a more detailed dissection is states
states = list(prim_data["state"].unique())
for state in states:
    print(f"State: {state}:: Branch IDs: {prim_data.loc[prim_data.state == state].branch_id.unique()}")

# Explore some state data to get idea about data distributio
karnataka_dat = prim_data.loc[prim_data.state == 'KARNATAKA'].copy()
karnataka_dat.application_date = pd.to_datetime(karnataka_dat.application_date, format="%Y-%m-%d").copy()
karnataka_dat.set_index("application_date", inplace=True)
karnataka_dat["branch_id"].fillna(0, inplace=True)

figure, axes = plt.subplots(2, 2)
figure.tight_layout(pad=1)
axes[0][0].plot(karnataka_dat["case_count"].loc[karnataka_dat.branch_id == 5.0])
axes[0][1].plot((karnataka_dat["case_count"].loc[karnataka_dat.branch_id == 64.0]), color='r')
axes[1][0].plot((karnataka_dat["case_count"].loc[karnataka_dat.branch_id == 121.0]), color='darkgreen')
axes[1][1].plot((karnataka_dat["case_count"].loc[karnataka_dat.branch_id == 0]), color='orange')


bihar_dat = prim_data.loc[prim_data.state == 'BIHAR'].copy()
bihar_dat.application_date = pd.to_datetime(bihar_dat.application_date, format="%Y-%m-%d").copy()
bihar_dat.set_index("application_date", inplace=True)
bihar_dat["branch_id"].fillna(0, inplace=True)

figure, axes = plt.subplots(2, 2)
figure.tight_layout(pad=1)
axes[0][0].plot((bihar_dat["case_count"].loc[bihar_dat.branch_id == 103.0]))
axes[0][1].plot((bihar_dat["case_count"].loc[bihar_dat.branch_id == 159.0]), color='r')
axes[1][0].plot((bihar_dat["case_count"].loc[bihar_dat.branch_id == 217.0]), color='darkgreen')
axes[1][1].plot((bihar_dat["case_count"].loc[bihar_dat.branch_id == 0]), color='orange')

"""
Part 2
Preprocessing
"""
# We can see missing values for only second segment
# Hence replace these with 0s and also their respective
# zones
null_data = prim_data[prim_data.isna().any(axis=1)]

null_data.branch_id.fillna(int(0), axis=0, inplace=True)

# Test filled zones
null_data.loc[(null_data.state == "WEST BENGAL"), ['zone']] = 'EAST'
null_data.loc[(null_data.state == "DELHI"), ['zone']] = 'NORTH'
null_data.loc[(null_data.state == "KARNATAKA"), ['zone']] = 'SOUTH'
null_data.loc[(null_data.state == "TAMIL NADU"), ['zone']] = 'SOUTH'
null_data.loc[(null_data.state == "UTTAR PRADESH"), ['zone']] = 'NORTH'
null_data.loc[(null_data.state == "PUNJAB"), ['zone']] = 'NORTH'
null_data.loc[(null_data.state == "TELANGANA"), ['zone']] = 'SOUTH'
null_data.loc[(null_data.state == "ASSAM"), ['zone']] = 'EAST'
null_data.loc[(null_data.state == "ANDHRA PRADESH"), ['zone']] = 'SOUTH'
null_data.loc[(null_data.state == "MAHARASHTRA"), ['zone']] = 'WEST'
null_data.loc[(null_data.state == "ORISSA"), ['zone']] = 'SOUTH'  # Merge SOUTH zone into EAST
null_data.loc[(null_data.state == "GUJARAT"), ['zone']] = 'WEST'
null_data.loc[(null_data.state == "ASSAM"), ['zone']] = 'EAST'
null_data.loc[(null_data.state == "JHARKHAND"), ['zone']] = 'EAST'
null_data.loc[(null_data.state == "UTTARAKHAND"), ['zone']] = 'NORTH'
null_data.loc[(null_data.state == "KERALA"), ['zone']] = 'SOUTH'
null_data.loc[(null_data.state == "CHHATTISGARH"), ['zone']] = 'CENTRAL'
null_data.loc[(null_data.state == "BIHAR"), ['zone']] = 'EAST'
null_data.loc[(null_data.state == "TRIPURA"), ['zone']] = 'EAST'
null_data.loc[(null_data.state == "MADHYA PRADESH"), ['zone']] = 'CENTRAL'
null_data.loc[(null_data.state == "HARYANA"), ['zone']] = 'NORTH'

# Fill branch_ids
null_data.loc[(null_data.state == "WEST BENGAL"), ['branch_id']] = 44
null_data.loc[(null_data.state == "KARNATAKA"), ['branch_id']] = 122
null_data.loc[(null_data.state == "TAMIL NADU"), ['branch_id']] = 272
null_data.loc[(null_data.state == "UTTAR PRADESH"), ['branch_id']] = 273
null_data.loc[(null_data.state == "PUNJAB"), ['branch_id']] = 274
null_data.loc[(null_data.state == "ASSAM"), ['branch_id']] = 256
null_data.loc[(null_data.state == "MAHARASHTRA"), ['branch_id']] = 252
null_data.loc[(null_data.state == "ORISSA"), ['branch_id']] = 166
null_data.loc[(null_data.state == "GUJARAT"), ['branch_id']] = 275
null_data.loc[(null_data.state == "JHARKHAND"), ['branch_id']] = 106
null_data.loc[(null_data.state == "KERALA"), ['branch_id']] = 83
null_data.loc[(null_data.state == "CHHATTISGARH"), ['branch_id']] = 86
null_data.loc[(null_data.state == "BIHAR"), ['branch_id']] = 218
null_data.loc[(null_data.state == "TRIPURA"), ['branch_id']] = 252
null_data.loc[(null_data.state == "MADHYA PRADESH"), ['branch_id']] = 148
null_data.loc[(null_data.state == "HARYANA"), ['branch_id']] = 276

# Fill values in primary data
prim_data.loc[(prim_data.state == "WEST BENGAL"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "DELHI"), ['zone']] = 'NORTH'
prim_data.loc[(prim_data.state == "KARNATAKA"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "TAMIL NADU"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "UTTAR PRADESH"), ['zone']] = 'NORTH'
prim_data.loc[(prim_data.state == "PUNJAB"), ['zone']] = 'NORTH'
prim_data.loc[(prim_data.state == "TELANGANA"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "ASSAM"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "ANDHRA PRADESH"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "MAHARASHTRA"), ['zone']] = 'WEST'
prim_data.loc[(prim_data.state == "ORISSA") & (prim_data.segment == 2), ['zone']] = 'SOUTH'  # Merge SOUTH zone into EAST
prim_data.loc[(prim_data.state == "GUJARAT"), ['zone']] = 'WEST'
prim_data.loc[(prim_data.state == "ASSAM"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "JHARKHAND"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "UTTARAKHAND"), ['zone']] = 'NORTH'
prim_data.loc[(prim_data.state == "KERALA"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "CHHATTISGARH"), ['zone']] = 'CENTRAL'
prim_data.loc[(prim_data.state == "BIHAR"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "TRIPURA"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "MADHYA PRADESH"), ['zone']] = 'CENTRAL'
prim_data.loc[(prim_data.state == "HARYANA"), ['zone']] = 'NORTH'

# Fill IDs in primary data
prim_data.branch_id.fillna(int(0), axis=0, inplace=True)

prim_data.loc[(prim_data.state == "WEST BENGAL") & (prim_data.branch_id == 0.0), ['branch_id']] = 44
prim_data.loc[(prim_data.state == "KARNATAKA") & (prim_data.branch_id == 0.0), ['branch_id']] = 122
prim_data.loc[(prim_data.state == "TAMIL NADU") & (prim_data.branch_id == 0.0), ['branch_id']] = 272
prim_data.loc[(prim_data.state == "UTTAR PRADESH") & (prim_data.branch_id == 0.0), ['branch_id']] = 273
prim_data.loc[(prim_data.state == "PUNJAB") & (prim_data.branch_id == 0.0), ['branch_id']] = 274
prim_data.loc[(prim_data.state == "ASSAM") & (prim_data.branch_id == 0.0), ['branch_id']] = 256
prim_data.loc[(prim_data.state == "MAHARASHTRA") & (prim_data.branch_id == 0.0), ['branch_id']] = 252
prim_data.loc[(prim_data.state == "ORISSA") & (prim_data.branch_id == 0.0), ['branch_id']] = 166
prim_data.loc[(prim_data.state == "GUJARAT") & (prim_data.branch_id == 0.0), ['branch_id']] = 275
prim_data.loc[(prim_data.state == "JHARKHAND") & (prim_data.branch_id == 0.0), ['branch_id']] = 106
prim_data.loc[(prim_data.state == "KERALA") & (prim_data.branch_id == 0.0), ['branch_id']] = 83
prim_data.loc[(prim_data.state == "CHHATTISGARH") & (prim_data.branch_id == 0.0), ['branch_id']] = 86
prim_data.loc[(prim_data.state == "BIHAR") & (prim_data.branch_id == 0.0), ['branch_id']] = 218
prim_data.loc[(prim_data.state == "TRIPURA") & (prim_data.branch_id == 0.0), ['branch_id']] = 252
prim_data.loc[(prim_data.state == "MADHYA PRADESH") & (prim_data.branch_id == 0.0), ['branch_id']] = 148
prim_data.loc[(prim_data.state == "HARYANA") & (prim_data.branch_id == 0.0), ['branch_id']] = 276

# Engineer new variables
prim_data["Temperature"] = 0
prim_data["Rainfall"] = 0
prim_data["GSDP"] = "Categorical"
prim_data["Rainfall_type"] = "Categorical"
prim_data["irrigation"] = "Categorical"
prim_data["industrialisation"] = "Categorical"
prim_data["number_of_branches"] = "Categorical"
prim_data["Season"] = "Categorical"
prim_data["weekend"] = "Categorical"
prim_data["festive season"] = "Categorical"
prim_data["National Holiday"] = "Categorical"
prim_data["Regional Holiday"] = "Categorical"

prim_data["Year"] = prim_data.index.year
prim_data["Month"] = prim_data.index.month
prim_data["Day"] = prim_data.index.day

# Fill seasons for each region
# North India
prim_data.loc[prim_data.index.to_series().between("April 2017", "April 2017") &
              prim_data.state.eq("DELHI") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("May 2017", "August 2017") & 
              prim_data.state.eq("DELHI"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("August 2017", "October 2017") & 
              prim_data.state.eq("DELHI"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2017", "February 2018") & 
              prim_data.state.eq("DELHI"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("February 2018", "April 2018") & 
              prim_data.state.eq("DELHI"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Apr 2018", "August 2018") & 
              prim_data.state.eq("DELHI"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("August 2018", "October 2018") & 
              prim_data.state.eq("DELHI"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2018", "February 2019") & 
              prim_data.state.eq("DELHI"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("February 2019", "April 2019") & 
              prim_data.state.eq("DELHI"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("April 2019", "July 2019") & 
              prim_data.state.eq("DELHI"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("DELHI"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("February 2017", "March 2017") &
              prim_data.state.eq("UTTAR PRADESH") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2017", "August 2017") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("August 2017", "October 2017") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2017", "January 2018") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2018", "March 2018") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Mar 2018", "August 2018") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("August 2018", "October 2018") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2018", "January 2019") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2019", "March 2019") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "August 2019") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("UTTAR PRADESH"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("March 2017", "March 2017") &
              prim_data.state.eq("PUNJAB") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2017", "August 2017") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("August 2017", "October 2017") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2017", "February 2018") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("February 2018", "March 2018") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Mar 2018", "August 2018") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("August 2018", "October 2018") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2018", "February 2019") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("February 2019", "March 2019") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "August 2019") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("PUNJAB"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("March 2017", "March 2017") &
              prim_data.state.eq("HARYANA") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2017", "August 2017") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("August 2017", "October 2017") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2017", "February 2018") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("February 2018", "March 2018") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Mar 2018", "August 2018") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("August 2018", "October 2018") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2018", "February 2019") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("February 2019", "March 2019") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "August 2019") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("HARYANA"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("March, 2017", "April 2017") &
              prim_data.state.eq("UTTARAKHAND") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("April 2017", "July 2017") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("July 2017", "September 2017") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("September 2017", "February 2018") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("February 2018", "April 2018") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Apr 2018", "July 2018") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("July 2018", "September 2018") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("September 2018", "February 2019") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("February 2019", "April 2019") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("April 2019", "July 2019") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("UTTARAKHAND"), "Season"] = "Monsoon"


# Eastern parts
prim_data.loc[prim_data.index.to_series().between("April 2017", "July 2017") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("July 2017", "October 2017") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2017", "January 2018") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2018", "March 2018") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("March 2018", "July 2018") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("July 2018", "October 2018") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2018", "January 2019") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2019", "March 2019") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "July 2019") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("WEST BENGAL"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("February 2017", "March 2017") &
              prim_data.state.eq("ASSAM") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2017", "June 2017") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("June 2017", "September 2017") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("September 2017", "January 2018") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2018", "March 2018") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("March 2018", "June 2018") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("June 2018", "September 2018") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("September 2018", "January 2019") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2019", "March 2019") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "June 2019") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("June 2019", "Aug 2019") & 
              prim_data.state.eq("ASSAM"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("February 2017", "March 2017") &
              prim_data.state.eq("ORISSA") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2017", "June 2017") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("June 2017", "September 2017") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("September 2017", "January 2018") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2018", "March 2018") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("March 2018", "June 2018") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jun 2018", "September 2018") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Sept 2018", "January 2019") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2019", "March 2019") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "June 2019") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jun 2019", "Aug 2019") & 
              prim_data.state.eq("ORISSA"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("February 2017", "March 2017") &
              prim_data.state.eq("JHARKHAND") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("April 2017", "July 2017") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("August 2017", "October 2017") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("November 2017", "January 2018") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2018", "March 2018") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("March 2018", "July 2018") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("July 2018", "October 2018") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2018", "January 2019") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2019", "March 2019") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "July 2019") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("JHARKHAND"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("February 2017", "March 2017") &
              prim_data.state.eq("BIHAR") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Mar 2017", "July 2017") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2017", "October 2017") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("oct 2017", "January 2018") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2018", "March 2018") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("March 2018", "July 2018") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("July 2018", "October 2018") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2018", "January 2019") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2019", "March 2019") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "July 2019") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("BIHAR"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("January 2017", "March 2017") &
              prim_data.state.eq("TRIPURA") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2017", "June 2017") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("June 2017", "October 2017") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2017", "December 2018") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("December 2018", "March 2018") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("March 2018", "June 2018") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("June 2018", "October 2018") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2018", "December 2019") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("December 2019", "March 2019") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "July 2019") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("TRIPURA"), "Season"] = "Monsoon"

# Southern part
prim_data.loc[prim_data.index.to_series().between("February 2017", "February 2017") &
              prim_data.state.eq("KARNATAKA") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("February 2017", "June 2017") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("June 2017", "October 2017") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2017", "Jan 2018") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2018", "February 2018") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("February 2018", "June 2018") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("June 2018", "October 2018") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("October 2018", "Jan 2019") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("December 2019", "February 2019") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("February 2019", "July 2019") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("KARNATAKA"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("February 2017", "February 2017") &
              prim_data.state.eq("TAMIL NADU") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("February 2017", "June 2017") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jun 2017", "October 2017") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2017", "Jan 2018") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2018", "February 2018") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Feb 2018", "June 2018") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jun 2018", "October 2018") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2018", "Jan 2019") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "February 2019") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Feb 2019", "July 2019") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("TAMIL NADU"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("February 2017", "February 2017") &
              prim_data.state.eq("TELANGANA") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Feb 2017", "June 2017") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jun 2017", "October 2017") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2017", "Jan 2018") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2018", "February 2018") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Feb 2018", "June 2018") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jun 2018", "October 2018") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2018", "Jan 2019") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "February 2019") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Feb 2019", "July 2019") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("TELANGANA"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("January 2017", "January 2017") &
              prim_data.state.eq("KERALA") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Jan 2017", "May 2017") & 
              prim_data.state.eq("KERALA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("May 2017", "November 2017") & 
              prim_data.state.eq("KERALA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Nov 2017", "Jan 2018") & 
              prim_data.state.eq("KERALA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2018", "January 2018") & 
              prim_data.state.eq("KERALA"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Jan 2018", "May 2018") & 
              prim_data.state.eq("KERALA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("May 2018", "November 2018") & 
              prim_data.state.eq("KERALA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Nov 2018", "Jan 2019") & 
              prim_data.state.eq("KERALA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "January 2019") & 
              prim_data.state.eq("KERALA"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "Jun 2019") & 
              prim_data.state.eq("KERALA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jun 2019", "Aug 2019") & 
              prim_data.state.eq("KERALA"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("January 2017", "January 2017") &
              prim_data.state.eq("ANDHRA PRADESH") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Feb 2017", "May 2017") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("May 2017", "November 2017") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("November 2017", "Jan 2018") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2018", "Feb 2018") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Feb 2018", "May 2018") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("May 2018", "October 2018") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2018", "Jan 2019") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "January 2019") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "May 2019") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("May 2019", "Jul 2019") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("ANDHRA PRADESH"), "Season"] = "Monsoon"


# West India
prim_data.loc[prim_data.index.to_series().between("January 2017", "January 2017") &
              prim_data.state.eq("MAHARASHTRA") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Jan 2017", "June 2017") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jun 2017", "October 2017") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2017", "December 2017") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("December 2017", "January 2018") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Jan 2018", "May 2018") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("May 2018", "October 2018") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2018", "Jan 2019") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "January 2019") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "June 2019") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jun 2019", "July 2019") &
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("MAHARASHTRA"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("January 2017", "February 2017") &
              prim_data.state.eq("GUJARAT") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Feb 2017", "July 2017") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2017", "October 2017") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2017", "January 2018") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2018", "February 2018") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Feb 2018", "July 2018") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2018", "October 2018") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2018", "January 2019") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("January 2019", "February,2019") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Feb 2019", "July 2019") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("GUJARAT"), "Season"] = "Monsoon"



# Central region
prim_data.loc[prim_data.index.to_series().between("February 2017", "March 2017") &
              prim_data.state.eq("CHHATTISGARH") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Mar 2017", "July 2017") & 
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2017", "October 2017") & 
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2017", "January 2018") & 
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2018", "March 2018") & 
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Mar 2018", "July 2018") & 
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2018", "October 2018") & 
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2018", "January 2019") & 
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "March 2019") & 
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2019", "July 2019") & 
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("July 2019", "Aug 2019") &
              prim_data.state.eq("CHHATTISGARH"), "Season"] = "Monsoon"


prim_data.loc[prim_data.index.to_series().between("February 2017", "March 2017") &
              prim_data.state.eq("MADHYA PRADESH") ,"Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("March 2017", "August,2017") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Aug 2017", "October 2017") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2017", "January 2018") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2018", "March 2018") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Spring"

prim_data.loc[prim_data.index.to_series().between("Mar 2018", "July 2018") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2018", "October 2018") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Monsoon"
prim_data.loc[prim_data.index.to_series().between("Oct 2018", "January 2019") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Winter"
prim_data.loc[prim_data.index.to_series().between("Jan 2019", "March 2019") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Spring"
prim_data.loc[prim_data.index.to_series().between("Mar 2019", "July 2019") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Summer"
prim_data.loc[prim_data.index.to_series().between("Jul 2019", "Aug 2019") & 
              prim_data.state.eq("MADHYA PRADESH"), "Season"] = "Summer"

# Fill these variables
prim_data.loc[(prim_data.state == "WEST BENGAL"), ['GSDP']] = 'High'
prim_data.loc[(prim_data.state == "DELHI"), ['GSDP']] = 'High'
prim_data.loc[(prim_data.state == "KARNATAKA"), ['GSDP']] = 'Very High'
prim_data.loc[(prim_data.state == "TAMIL NADU"), ['GSDP']] = 'Very High'
prim_data.loc[(prim_data.state == "UTTAR PRADESH"), ['GSDP']] = 'Very High'
prim_data.loc[(prim_data.state == "PUNJAB"), ['GSDP']] = 'Low'
prim_data.loc[(prim_data.state == "TELANGANA"), ['GSDP']] = 'Moderate'
prim_data.loc[(prim_data.state == "ASSAM"), ['GSDP']] = 'Low'
prim_data.loc[(prim_data.state == "ANDHRA PRADESH"), ['GSDP']] = 'High'
prim_data.loc[(prim_data.state == 'MAHARASHTRA'), ['GSDP']] = "Ultra High"
prim_data.loc[(prim_data.state == "ORISSA"), ['GSDP']] = 'Low'
prim_data.loc[(prim_data.state == "GUJARAT"), ['GSDP']] = 'Very High'
prim_data.loc[(prim_data.state == "JHARKHAND"), ['GSDP']] = 'Low'
prim_data.loc[(prim_data.state == "UTTARAKHAND"), ['GSDP']] = 'Very Low'
prim_data.loc[(prim_data.state == "KERALA"), ['GSDP']] = 'Moderate'
prim_data.loc[(prim_data.state == "CHHATTISGARH"), ['GSDP']] = 'Low'
prim_data.loc[(prim_data.state == "BIHAR"), ['GSDP']] = 'Low'
prim_data.loc[(prim_data.state == "TRIPURA"), ['GSDP']] = 'Extremely Low'
prim_data.loc[(prim_data.state == "MADHYA PRADESH"), ['GSDP']] = 'Moderate'
prim_data.loc[(prim_data.state == "HARYANA"), ['GSDP']] = 'High'

prim_data.loc[(prim_data.state == "WEST BENGAL"), ['Rainfall_type']] = 'High'
prim_data.loc[(prim_data.state == "DELHI"), ['Rainfall_type']] = 'Low'
prim_data.loc[(prim_data.state == "KARNATAKA"), ['Rainfall_type']] = 'Moderate'
prim_data.loc[(prim_data.state == "TAMIL NADU"), ['Rainfall_type']] = 'High'
prim_data.loc[(prim_data.state == "UTTAR PRADESH"), ['Rainfall_type']] = 'Moderate'
prim_data.loc[(prim_data.state == "PUNJAB"), ['Rainfall_type']] = 'Low'
prim_data.loc[(prim_data.state == "TELANGANA"), ['Rainfall_type']] = 'High'
prim_data.loc[(prim_data.state == "ASSAM"), ['Rainfall_type']] = 'Very High'
prim_data.loc[(prim_data.state == "ANDHRA PRADESH"), ['Rainfall_type']] = 'High'
prim_data.loc[(prim_data.state == 'MAHARASHTRA'), ['Rainfall_type']] = "Moderate"
prim_data.loc[(prim_data.state == "ORISSA"), ['Rainfall_type']] = 'High'
prim_data.loc[(prim_data.state == "GUJARAT"), ['Rainfall_type']] = 'Low'
prim_data.loc[(prim_data.state == "JHARKHAND"), ['Rainfall_type']] = 'Low'
prim_data.loc[(prim_data.state == "UTTARAKHAND"), ['Rainfall_type']] = 'High'
prim_data.loc[(prim_data.state == "KERALA"), ['Rainfall_type']] = 'Very High'
prim_data.loc[(prim_data.state == "CHHATTISGARH"), ['Rainfall_type']] = 'High'
prim_data.loc[(prim_data.state == "BIHAR"), ['Rainfall_type']] = 'High'
prim_data.loc[(prim_data.state == "TRIPURA"), ['Rainfall_type']] = 'High'
prim_data.loc[(prim_data.state == "MADHYA PRADESH"), ['Rainfall_type']] = 'Moderate'
prim_data.loc[(prim_data.state == "HARYANA"), ['Rainfall_type']] = 'Low'

prim_data.loc[(prim_data.state == "WEST BENGAL"), ['irrigation']] = 'High'
prim_data.loc[(prim_data.state == "DELHI"), ['irrigation']] = 'High'
prim_data.loc[(prim_data.state == "KARNATAKA"), ['irrigation']] = 'Very low'
prim_data.loc[(prim_data.state == "TAMIL NADU"), ['irrigation']] = 'Moderate'
prim_data.loc[(prim_data.state == "UTTAR PRADESH"), ['irrigation']] = 'High'
prim_data.loc[(prim_data.state == "PUNJAB"), ['irrigation']] = 'Very High'
prim_data.loc[(prim_data.state == "TELANGANA"), ['irrigation']] = 'Low'
prim_data.loc[(prim_data.state == "ASSAM"), ['irrigation']] = 'Very Low'
prim_data.loc[(prim_data.state == "ANDHRA PRADESH"), ['irrigation']] = 'Very Low'
prim_data.loc[(prim_data.state == 'MAHARASHTRA'), ['irrigation']] = "Very Low"
prim_data.loc[(prim_data.state == "ORISSA"), ['irrigation']] = 'Low'
prim_data.loc[(prim_data.state == "GUJARAT"), ['irrigation']] = 'Low'
prim_data.loc[(prim_data.state == "JHARKHAND"), ['irrigation']] = 'Very Low'
prim_data.loc[(prim_data.state == "UTTARAKHAND"), ['irrigation']] = 'Moderate'
prim_data.loc[(prim_data.state == "KERALA"), ['irrigation']] = 'Low'
prim_data.loc[(prim_data.state == "CHHATTISGARH"), ['irrigation']] = 'Very Low'
prim_data.loc[(prim_data.state == "BIHAR"), ['irrigation']] = 'Moderate'
prim_data.loc[(prim_data.state == "TRIPURA"), ['irrigation']] = 'Very Low'
prim_data.loc[(prim_data.state == "MADHYA PRADESH"), ['irrigation']] = 'Low'
prim_data.loc[(prim_data.state == "HARYANA"), ['irrigation']] = 'Very High'

prim_data.loc[(prim_data.state == "WEST BENGAL"), ['industrialisation']] = 'Moderate'
prim_data.loc[(prim_data.state == "DELHI"), ['industrialisation']] = 'High'
prim_data.loc[(prim_data.state == "KARNATAKA"), ['industrialisation']] = 'Very High'
prim_data.loc[(prim_data.state == "TAMIL NADU"), ['industrialisation']] = 'Very High'
prim_data.loc[(prim_data.state == "UTTAR PRADESH"), ['industrialisation']] = 'Low'
prim_data.loc[(prim_data.state == "PUNJAB"), ['industrialisation']] = 'Moderate'
prim_data.loc[(prim_data.state == "TELANGANA"), ['industrialisation']] = 'Low'
prim_data.loc[(prim_data.state == "ASSAM"), ['industrialisation']] = 'Very Low'
prim_data.loc[(prim_data.state == "ANDHRA PRADESH"), ['industrialisation']] = 'High'
prim_data.loc[(prim_data.state == 'MAHARASHTRA'), ['industrialisation']] = "Very High"
prim_data.loc[(prim_data.state == "ORISSA"), ['industrialisation']] = 'Low'
prim_data.loc[(prim_data.state == "GUJARAT"), ['industrialisation']] = 'High'
prim_data.loc[(prim_data.state == "JHARKHAND"), ['industrialisation']] = 'Very Low'
prim_data.loc[(prim_data.state == "UTTARAKHAND"), ['industrialisation']] = 'Low'
prim_data.loc[(prim_data.state == "KERALA"), ['industrialisation']] = 'Moderate'
prim_data.loc[(prim_data.state == "CHHATTISGARH"), ['industrialisation']] = 'Very Low'
prim_data.loc[(prim_data.state == "BIHAR"), ['industrialisation']] = 'Low'
prim_data.loc[(prim_data.state == "TRIPURA"), ['industrialisation']] = 'Very Low'
prim_data.loc[(prim_data.state == "MADHYA PRADESH"), ['industrialisation']] = 'Moderate'
prim_data.loc[(prim_data.state == "HARYANA"), ['industrialisation']] = 'High'

prim_data.loc[(prim_data.state == "WEST BENGAL"), ['number_of_branches']] = 'High'
prim_data.loc[(prim_data.state == "DELHI"), ['number_of_branches']] = 'Very Low'
prim_data.loc[(prim_data.state == "KARNATAKA"), ['number_of_branches']] = 'Moderate'
prim_data.loc[(prim_data.state == "TAMIL NADU"), ['number_of_branches']] = 'High'
prim_data.loc[(prim_data.state == "UTTAR PRADESH"), ['number_of_branches']] = 'High'
prim_data.loc[(prim_data.state == "PUNJAB"), ['number_of_branches']] = 'High'
prim_data.loc[(prim_data.state == "TELANGANA"), ['number_of_branches']] = 'Very Low'
prim_data.loc[(prim_data.state == "ASSAM"), ['number_of_branches']] = 'Low'
prim_data.loc[(prim_data.state == "ANDHRA PRADESH"), ['number_of_branches']] = 'Moderate'
prim_data.loc[(prim_data.state == 'MAHARASHTRA'), ['number_of_branches']] = "Very High"
prim_data.loc[(prim_data.state == "ORISSA"), ['number_of_branches']] = 'Low'
prim_data.loc[(prim_data.state == "GUJARAT"), ['number_of_branches']] = 'High'
prim_data.loc[(prim_data.state == "JHARKHAND"), ['number_of_branches']] = 'Very Low'
prim_data.loc[(prim_data.state == "UTTARAKHAND"), ['number_of_branches']] = 'Very Low'
prim_data.loc[(prim_data.state == "KERALA"), ['number_of_branches']] = 'Very Low'
prim_data.loc[(prim_data.state == "CHHATTISGARH"), ['number_of_branches']] = 'Very Low'
prim_data.loc[(prim_data.state == "BIHAR"), ['number_of_branches']] = 'Low'
prim_data.loc[(prim_data.state == "TRIPURA"), ['number_of_branches']] = 'Very Low'
prim_data.loc[(prim_data.state == "MADHYA PRADESH"), ['number_of_branches']] = 'Very Low'
prim_data.loc[(prim_data.state == "HARYANA"), ['number_of_branches']] = 'Moderate'

# Fill in some dates/days
prim_data.reset_index(inplace=True)
add_datepart(prim_data, "application_date", drop=False)

prim_data["weekend"] = prim_data["application_Dayofweek"].ge(5).astype(np.int8)
prim_data.set_index("application_date", inplace=True)

# Fill national holidays
prim_data["National_holiday"], prim_data["Regional_holiday"] = np.int8(0), np.int8(0)

prim_data.loc["Jan 25, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["Jan 25, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["April 10, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["April 10, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["April 10, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["April 14, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["April 14, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["April 14, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["May 22, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["May 22, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["May 22, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["July 31, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["July 31, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["July 31, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["August 15, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["August 15, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["August 15, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["August 29, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["August 29, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["August 29, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["October 02, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["October 02, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["October 02, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["October 25, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["October 25, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["October 25, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["October 29, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["October 29, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["October 29, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["December 25, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["December 25, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["December 25, 2019", "National_holiday"] = np.int8(1)

prim_data.loc["December 31, 2017", "National_holiday"] = np.int8(1)
prim_data.loc["December 31, 2018", "National_holiday"] = np.int8(1)
prim_data.loc["December 31, 2019", "National_holiday"] = np.int8(1)


# Makar Sakranti/Pongal
prim_data.loc[prim_data.index.to_series().between("January 13, 2018", "January 15, 2018") & 
              prim_data.state.eq("MAHARASHTRA","UTTAR PRADESH",'PUNJAB'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("January 13, 2019", "January 15, 2019") & 
              prim_data.state.eq("MAHARASHTRA","UTTAR PRADESH",'PUNJAB'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("January 13, 2018", "January 15, 2018") & 
              prim_data.state.eq("HARYANA","UTTARAKHAND",'DELHI'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("January 13, 2019", "January 15, 2019") & 
              prim_data.state.eq("HARYANA","UTTARAKHAND",'DELHI'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("January 13, 2018", "January 15, 2018") & 
              prim_data.state.eq("HARYANA","UTTARAKHAND",'DELHI'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("January 13, 2018", "January 15, 2018") & 
              prim_data.state.eq("KARNATAKA","TAMIL NADU",'KERALA'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("January 13, 2018", "January 15, 2018") & 
              prim_data.state.eq("ANDHRA PRADESH","TELANGANA"), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("January 13, 2019", "January 15, 2019") & 
              prim_data.state.eq("KARNATAKA","TAMIL NADU",'KERALA'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("January 13, 2019", "January 15, 2019") & 
              prim_data.state.eq("ANDHRA PRADESH","TELANGANA"), "Regional_holiday"] = np.int8(1)

# Holi
prim_data.loc[prim_data.index.to_series().between("March 20, 2018", "March 21, 2018") & 
              prim_data.state.eq("GUJARAT","MADHYA PRADESH",'BIHAR'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("March 20, 2019", "March 21, 2019") & 
              prim_data.state.eq("GUJARAT","MADHYA PRADESH", "BIHAR"), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("March 20, 2018", "March 21, 2018") & 
              prim_data.state.eq("ORISSA","UTTAR PRADESH",'PUNJAB'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("March 20, 2019", "March 21, 2019") & 
              prim_data.state.eq("ORISSA","UTTAR PRADESH",'PUNJAB'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("March 20, 2018", "January 15, 2018") & 
              prim_data.state.eq("HARYANA","UTTARAKHAND",'DELHI'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("March 20, 2019", "January 15, 2019") & 
              prim_data.state.eq("HARYANA","UTTARAKHAND",'DELHI'), "Regional_holiday"] = np.int8(1)

# Diwali
prim_data.loc[prim_data.index.to_series().between("October 27, 2018", "October 28, 2018") & 
              prim_data.state.eq("GUJARAT","MADHYA PRADESH",'BIHAR'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("October 27, 2019", "October 28, 2019") & 
              prim_data.state.eq("GUJARAT","MADHYA PRADESH", "BIHAR"), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("October 27, 2018", "October 28, 2018") & 
              prim_data.state.eq("ORISSA","UTTAR PRADESH",'PUNJAB'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("October 27, 2019", "October 28, 2019") & 
              prim_data.state.eq("ORISSA","UTTAR PRADESH",'PUNJAB'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("October 27, 2018", "October 28, 2018") & 
              prim_data.state.eq("HARYANA","UTTARAKHAND",'DELHI'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("October 27, 2019", "October 28, 2019") & 
              prim_data.state.eq("HARYANA","UTTARAKHAND",'DELHI'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("October 27, 2018", "October 28, 2018") & 
              prim_data.state.eq("MAHARASHTRA","WEST BENGAL",'JHARKHAND'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("October 27, 2019", "October 28, 2019") & 
              prim_data.state.eq("MAHARASHTRA","WEST BENGAL",'JHARKHAND'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("October 27, 2018", "October 28, 2018") & 
              prim_data.state.eq("CHHATTISGARH","ASSAM",'TRIPURA'), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("October 27, 2019", "October 28, 2019") & 
              prim_data.state.eq("CHHATTISGARH","ASSAM",'TRIPURA'), "Regional_holiday"] = np.int8(1)


# Ganesh Chaturthi
prim_data.loc[prim_data.index.to_series().between("August 21, 2017", "August 22, 2017") & 
              prim_data.state.eq("GUJARAT","MAHARASHTRA"), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("August 21, 2018", "August 22, 2018") & 
              prim_data.state.eq("GUJARAT", "MAHARASHTRA"), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().between("August 21, 2019", "August 22, 2019") & 
              prim_data.state.eq("GUJARAT", "MAHARASHTRA"), "Regional_holiday"] = np.int8(1)

# Shivaji Jayanti
prim_data.loc[prim_data.index.to_series().eq("February 19, 2018") & 
              prim_data.state.eq("MAHARASHTRA"), "Regional_holiday"] = np.int8(1)

prim_data.loc[prim_data.index.to_series().eq("February 19, 2018") & 
              prim_data.state.eq("MAHARASHTRA"), "Regional_holiday"] = np.int8(1)


# Convert Trues and Falses
prim_data.application_Is_month_end = prim_data.application_Is_month_end.astype(np.int8)
prim_data.application_Is_month_start = prim_data.application_Is_month_start.astype(np.int8)
prim_data.application_Is_quarter_end = prim_data.application_Is_quarter_end.astype(np.int8)
prim_data.application_Is_quarter_start = prim_data.application_Is_quarter_start.astype(np.int8)
prim_data.application_Is_year_end = prim_data.application_Is_year_end.astype(np.int8)
prim_data.application_Is_year_start = prim_data.application_Is_year_start.astype(np.int8)

# Simulate weather data if needed
sim = 0
months = ["Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
          "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
          "Apr", "May", "Jun", "Jul"]


def sim_temp(nDays=30):
    """Generate simulated data for 30 days."""
    for year in range(17, 19+1):
        if year == 17:
            for month in months[:12]:
                high = np.float64(input("High temperature for {0} 20{1}: ".format(month, year)))
                low = np.float64(input("Low temperature for {0} 20{1}: ".format(month, year)))
        
                sim = list(np.random.uniform(low, high, nDays))
    
        if year == 18:
            for month in months[12:24]:
                high = np.float64(input("High temperature for {0} 20{1}: ".format(month, year)))
                low = np.float64(input("Low temperature for {0} 20{1}: ".format(month, year)))
        
                sim = list(np.random.uniform(low, high, nDays))
    
        if year == 19:
            for month in months[24:]:
                high = np.float64(input("High temperature for {0} 20{1}: ".format(month, year)))
                low = np.float64(input("Low temperature for {0} 20{1}: ".format(month, year)))
        
                sim = list(np.random.uniform(low, high, nDays))
    
        return sim[:5]
    # for arr in temp_data:
    #     for sub_list in arr:
    #         temp.append(sub_list)

cols_to_drop = ['application_Day', 'application_Dayofweek', 'festive season',
                'application_Year', 'application_Month', 'application_Week',
                'application_Dayofyear', 'application_Elapsed', "Temperature",
                "Rainfall", "National Holiday", "Regional Holiday", "Day", "case_count"]
prim_data.drop(cols_to_drop, aixs=1, inplace=True)

# One hot encode vars

segDumm = pd.get_dummies(prim_data['segment'], prefix="segment")
branch_idDumm = pd.get_dummies(prim_data['branch_id'], prefix="branch_id")
stateDumm = pd.get_dummies(prim_data['state'], prefix="state")
zoneDumm = pd.get_dummies(prim_data['zone'], prefix="zone")
devDumm = pd.get_dummies(prim_data['GSDP'], prefix="gsdp")
industryDumm = pd.get_dummies(prim_data['industrialisation'], prefix="industrial")
nBranch = pd.get_dummies(prim_data['number_of_branches'], prefix="n_branch")
rainDumm = pd.get_dummies(prim_data['Rainfall_type'], prefix="rain_type")
irrDum = pd.get_dummies(prim_data['irrigation'], prefix="irrigation")
seasonDumm = pd.get_dummies(prim_data['Season'], prefix="season")
weekendDumm = pd.get_dummies(prim_data['weekend'], prefix="weekend")
yearDumm = pd.get_dummies(prim_data['Year'], prefix="Year")
monthDumm = pd.get_dummies(prim_data['Month'], prefix="Month")
monthEnd = pd.get_dummies(prim_data['application_Is_month_end'], prefix="application_Is_month_end")
monthStart = pd.get_dummies(prim_data['application_Is_month_start'], prefix="application_Is_month_start")
quartEnd = pd.get_dummies(prim_data['application_Is_quarter_end'], prefix="application_Is_quarter_end")
quartStart = pd.get_dummies(prim_data['application_Is_quarter_start'], prefix="application_Is_quarter_start")
yearEnd = pd.get_dummies(prim_data['application_Is_year_end'], prefix="application_Is_year_end")
yearStart = pd.get_dummies(prim_data['application_Is_year_start'], prefix="application_year_start")
natHoliday = pd.get_dummies(prim_data['National_holiday'], prefix="nat_holiday")
regHoliday = pd.get_dummies(prim_data['Regional_holiday'], prefix="reg_holiday")

cols = list(prim_data.columns)

prim_data = pd.concat([prim_data, segDumm], axis=1)
prim_data = pd.concat([prim_data, branch_idDumm], axis=1)
prim_data = pd.concat([prim_data, stateDumm], axis=1)
prim_data = pd.concat([prim_data, zoneDumm], axis=1)
prim_data = pd.concat([prim_data, devDumm], axis=1)
prim_data = pd.concat([prim_data, industryDumm], axis=1)
prim_data = pd.concat([prim_data, nBranch], axis=1)
prim_data = pd.concat([prim_data, rainDumm], axis=1)
prim_data = pd.concat([prim_data, irrDum], axis=1)
prim_data = pd.concat([prim_data, seasonDumm], axis=1)
prim_data = pd.concat([prim_data, weekendDumm], axis=1)
prim_data = pd.concat([prim_data, yearDumm], axis=1)
prim_data = pd.concat([prim_data, monthDumm], axis=1)
prim_data = pd.concat([prim_data, monthEnd], axis=1)
prim_data = pd.concat([prim_data, monthStart], axis=1)
prim_data = pd.concat([prim_data, quartEnd], axis=1)
prim_data = pd.concat([prim_data, quartStart], axis=1)
prim_data = pd.concat([prim_data, yearEnd], axis=1)
prim_data = pd.concat([prim_data, yearStart], axis=1)
prim_data = pd.concat([prim_data, natHoliday], axis=1)
prim_data = pd.concat([prim_data, regHoliday], axis=1)

prim_data.drop(cols, axis=1, inplace=True)

del segDumm, branch_idDumm, stateDumm, zoneDumm, devDumm,\
    industryDumm,nBranch,rainDumm,irrDum,seasonDumm,weekendDumm,\
    yearDumm,monthDumm,monthEnd,monthStart,quartEnd,quartStart,\
    yearEnd,yearStart,natHoliday,regHoliday


models = [LinearRegression(),
          ElasticNetCV(max_iter=1500, cv=6),
          SGDRegressor(learning_rate='optimal'),
          PassiveAggressiveRegressor(C=0.5, ),
          Ridge(),
          RandomForestRegressor(max_depth=8),
          GradientBoostingRegressor(max_depth=8, alpha=0.5),
          AdaBoostRegressor(learning_rate=0.8, loss='exponential'),
          BaggingRegressor(),
          SVR(kernel='sigmoid'),
          NuSVR(kernel='sigmoid'),
          XGBRFRegressor(learning_rate=0.5, max_depth=5, objective="reg:squarederror", n_estimators=150),
          XGBRegressor(max_depth=5, objective="reg:squarederror", eta=0.2)]

train_len = int(len(prim_data)*0.70)

x_train, y_train = prim_data[:train_len], target[:train_len]
x_test, y_test = prim_data[train_len:], target[train_len:]

import sklearn.metrics as metrics

def show_score(x, y, estimator):
    """
    Returns MAE scores for specified models.
    Also returns r2 scores if applicable

    Arguments:
        x {[array/DataFrame]} -- [Array or matrix of features. Can also be dataframe]
        y {[array]} -- [Target values]
        estimator {[str]} -- [The estimator being used]
    """
    # Instantiate models and predict values
    estimator.fit(x, y)
    preds = estimator.predict(x_test)
    preds = abs(preds.astype(int))
    actuals = y_test

    # Print results
    print(f"{estimator.__class__.__name__}:: r2 score = {round(metrics.r2_score(actuals, preds), 2)} : MAE = {round(metrics.mean_absolute_error(actuals, preds), 2)}")


for model in models:
    show_score(x_train, y_train, model)

pList = [0, 1, 2, 4, 8, 10]
dList = [0, 1, 2, 3, 4]
qList = [0, 1, 2, 3]


def show_bestScore(train_set, test_set):
    """
    Returns best cross-validated
    MAE and (p,d,q) order
    for a ts model.
    """
    start = input("Do you have p, d and q values defined? ")
    if start == "No" or start == "no" or start == "N" or start == "n":
        print("Please define p, d, q values and retry.")
    else:
        print("Finding out...")
        target = [values for values in y_train]
        testVals = [values for values in y_test]
        target = y_train.astype("float32")
        testVals = y_test.astype("float32")
        score = [10000, (0, 0, 0)]
        for p in pList:
            for d in dList:
                for q in qList:
                    order = (p, d, q)
                    model = SARIMAX(target, order=order)
                    fit = model.fit(disp=False)
                    preds = fit.forecast(len(x_test))
                    error = mean_absolute_error(testVals, preds)
                    if score[0] != 0 and error < score[0]:
                        score.pop()
                        score.pop()
                        score.append(error)
                        score.append(order)

        best_score, best_order = score[0], score[1]
        out = print("Best SARIMAX: MAE = %.f :: Order = %s" %
                    (best_score, best_order))
        if not best_score:
            print("Invalid or missing value for MAE. Please retry.")
        elif not best_order:
            print("Invalid or missing order of values. Please retry.")
        else:
            return out  # Best MAE = 700 :: Order = (8, 3, 1)




