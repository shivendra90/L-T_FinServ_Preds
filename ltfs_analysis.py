#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 12:47:33 2020

@author: shivendra
"""
import holidays
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno

from statsmodels.tsa.statespace.sarimax import SARIMAX
from warnings import filterwarnings
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor, XGBRFRegressor
from yellowbrick.regressor import residuals_plot, prediction_error
from yellowbrick.features import rank2d, rank1d
from yellowbrick.model_selection import RFECV, ValidationCurve, LearningCurve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

from fastai.collab import add_datepart
plt.ion()

register_matplotlib_converters()
filterwarnings("ignore")
"""
Part 1
Data Loading and Exploration
"""

prim_data = pd.read_csv("train.csv", engine="c")
test = pd.read_csv("test.csv", engine="c")
target = prim_data["case_count"]
cols = list(prim_data.columns)


def mean_abs_percent_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


prim_data["application_date"] = pd.to_datetime(prim_data.application_date,
                                               format="%Y-%m-%d").copy()  # Copy to prevent warning
prim_data.set_index("application_date", inplace=True)

print("Number of missing values out of {0} rows/samples: \n{1}".format(len(prim_data),
                                                                       prim_data.isna().sum()))

msno.bar(prim_data, figsize=(10, 5), fontsize=12)  # Plot columns with missing values

# For a more detailed dissection of states
states = list(prim_data["state"].unique())
for state in states:
    print(f"State: {state}:: Branch IDs: {prim_data.loc[prim_data.state == state].branch_id.unique()}")

# Explore some state data to get idea about data distribution
karnataka_dat = prim_data.loc[prim_data.state == 'KARNATAKA'].copy()
karnataka_dat.set_index("application_date", inplace=True)
karnataka_dat["branch_id"].fillna(0, inplace=True)

figure, axes = plt.subplots(2, 2)
figure.tight_layout(pad=1)
axes[0][0].plot(karnataka_dat["case_count"].loc[karnataka_dat.branch_id == 5.0])
axes[0][1].plot((karnataka_dat["case_count"].loc[karnataka_dat.branch_id == 64.0]),
                color='r')
axes[1][0].plot((karnataka_dat["case_count"].loc[karnataka_dat.branch_id == 121.0]),
                color='darkgreen')
axes[1][1].plot((karnataka_dat["case_count"].loc[karnataka_dat.branch_id == 0]),
                color='orange')


bihar_dat = prim_data.loc[prim_data.state == 'BIHAR'].copy()
bihar_dat.set_index("application_date", inplace=True)
bihar_dat["branch_id"].fillna(0, inplace=True)

figure, axes = plt.subplots(2, 2)
figure.tight_layout(pad=1)
axes[0][0].plot((bihar_dat["case_count"].loc[bihar_dat.branch_id == 103.0]))
axes[0][1].plot((bihar_dat["case_count"].loc[bihar_dat.branch_id == 159.0]),
                color='r')
axes[1][0].plot((bihar_dat["case_count"].loc[bihar_dat.branch_id == 217.0]),
                color='darkgreen')
axes[1][1].plot((bihar_dat["case_count"].loc[bihar_dat.branch_id == 0]),
                color='orange')

"""
Part 2
Preprocessing
"""
# We can see missing values for only second segment
# Hence replace these with 0s and also their respective
# zones


# Fill values in primary data
prim_data.loc[(prim_data.state == "WEST BENGAL"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "DELHI"), ['zone']] = 'NORTH'
prim_data.loc[(prim_data.state == "KARNATAKA"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "TAMIL NADU"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "UTTAR PRADESH"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "PUNJAB"), ['zone']] = 'NORTH'
prim_data.loc[(prim_data.state == "TELANGANA"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "ASSAM"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "ANDHRA PRADESH"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "MAHARASHTRA"), ['zone']] = 'WEST'
prim_data.loc[(prim_data.state == "ORISSA") & (prim_data.segment == 2),
              ['zone']] = 'SOUTH'  # Merge SOUTH zone into EAST
prim_data.loc[(prim_data.state == "GUJARAT"), ['zone']] = 'WEST'
prim_data.loc[(prim_data.state == "JHARKHAND"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "UTTARAKHAND"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "KERALA"), ['zone']] = 'SOUTH'
prim_data.loc[(prim_data.state == "CHHATTISGARH"), ['zone']] = 'CENTRAL'
prim_data.loc[(prim_data.state == "BIHAR"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "TRIPURA"), ['zone']] = 'EAST'
prim_data.loc[(prim_data.state == "MADHYA PRADESH"), ['zone']] = 'CENTRAL'
prim_data.loc[(prim_data.state == "HARYANA"), ['zone']] = 'NORTH'

# Fill IDs in primary data
prim_data.branch_id.fillna(int(0), axis=0, inplace=True)

prim_data.loc[(prim_data.state == "WEST BENGAL") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 44
prim_data.loc[(prim_data.state == "KARNATAKA") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 122
prim_data.loc[(prim_data.state == "TAMIL NADU") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 272
prim_data.loc[(prim_data.state == "UTTAR PRADESH") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 273
prim_data.loc[(prim_data.state == "PUNJAB") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 274
prim_data.loc[(prim_data.state == "ASSAM") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 256
prim_data.loc[(prim_data.state == "MAHARASHTRA") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 252
prim_data.loc[(prim_data.state == "ORISSA") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 166
prim_data.loc[(prim_data.state == "GUJARAT") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 275
prim_data.loc[(prim_data.state == "JHARKHAND") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 106
prim_data.loc[(prim_data.state == "KERALA") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 83
prim_data.loc[(prim_data.state == "CHHATTISGARH") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 86
prim_data.loc[(prim_data.state == "BIHAR") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 218
prim_data.loc[(prim_data.state == "TRIPURA") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 252
prim_data.loc[(prim_data.state == "MADHYA PRADESH") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 148
prim_data.loc[(prim_data.state == "HARYANA") & (prim_data.branch_id == 0.0),
              ['branch_id']] = 276

# Group data according to segment and application date
grouped_df = pd.DataFrame(prim_data.groupby(["application_date", "segment"])
                          ["case_count"].sum())
grouped_df.columns

grouped_df.reset_index(inplace=True)

segment_1 = grouped_df.loc[grouped_df.segment == 1].copy()
segment_1.set_index("application_date", inplace=True)
segment_2 = grouped_df.loc[grouped_df.segment == 2].copy()
segment_2.set_index("application_date", inplace=True)

# plot case counts for each segment
fig, axes = plt.subplots(2, 1, figsize=(10, 8))
axes[0].plot(segment_1.case_count)
axes[1].plot(segment_2.case_count, color='r')
fig.tight_layout(pad=1)

# Start appending holidays
ind_holidays = holidays.India(years=[2017, 2018, 2019])
natHolidays = [date for date in ind_holidays.keys()]

grouped_df["is_NationalHoliday"] = 0
grouped_df["is_NationalHoliday"] = grouped_df["application_date"].isin(natHolidays).astype("int")

# Engineer regional holidays. These will be
# different from national holidays and will
# not coincide/overlap with existing national
# holidays

states = ['AS', 'CG', 'KA', 'GJ', 'BR', 'RJ',
          'OD', 'TN', 'AP', 'WB', 'KL', 'HR', 'MH',
          'MP', 'UP', 'UK', 'TN']
regHolidays = []

for state in states:
    for date in holidays.India(prov=state, years=[2017, 2018, 2019]).keys():
        if date not in natHolidays:
            regHolidays.append(date)

regHolidays = set(regHolidays)  # remove duplicate entries
regHolidays = list(regHolidays)  # Convert back to list

grouped_df["is_regHoliday"] = 0
grouped_df["is_regHoliday"] = grouped_df["application_date"].isin(regHolidays).astype("int")

years = [2017, 2018, 2019]
months = ["Jan", "Feb", "Mar", "Apr", "May",
          "Jun", "Jul", "Aug", "Sep", "Oct",
          "Nov", "Dec"]

for year in years:
    for ix, month in enumerate(months):
        try:
            print(f"{month} {year} :: {len(grouped_df.loc[f'{month} {year}'])} samples")
        except KeyError:
            ix += 1

for year in years:
    for ix, month in enumerate(months):
        try:
            print(f"{month} {year} :: {len(segment_1.loc[f'{month} {year}'])} samples")
        except KeyError:
            ix += 1


# Slice out segments for better results
segment_1 = grouped_df.loc[grouped_df.segment == 1].copy()
segment_2 = grouped_df.loc[grouped_df.segment == 2].copy()


def simTemp(df):
    """
    Return Series of simulated temperatures.
    -------
    df : pandas.DataFrame
        A pandas data frame with index
        set to date_time.
    """
    init_list = []
    temperatures = []
    for year in years:
        for ix, month in enumerate(months):
            try:
                length = len(df.loc[eval('"{} {}"'.format(month, year))])
                high = np.float64(input(f"Input Hi temperature for {month} {year} "))
                low = np.float64(input(f"Input low temperature for {month} {year} "))

                init_list.append(np.random.uniform(low=low, high=high, size=length))
            except KeyError:
                ix += 1

    # Convert appended list to flat list
    for arr in init_list:
        for sub_arr in arr:
            temperatures.append(sub_arr)  # flat list
    return pd.Series(temperatures).round(1).rename("temperatures")  # Return rounded results


temps = simTemp(segment_1)
# Plot simulated temperatures
figure = plt.figure(figsize=(15, 5))
plt.plot(temps)
figure.tight_layout(pad=1)
figure.suptitle('Simulated Temperatures')


def simPrecip(df):
    """
    Append Series of simulated rains.
    -------
    df : pandas.DataFrame
        A pandas data frame with index
        set to date_time.
    """
    # check if index is set to datetime
    init_list = []
    precipitation = []
    if str(type(df.index)) != "<class 'pandas.core.indexes.datetimes.DatetimeIndex'>":
        df.set_index("application_date", inplace=True)
        for year in years:
            for ix, month in enumerate(months):
                try:
                    length = len(df.loc[eval('"{} {}"'.format(month, year))])
                    high = np.float64(input(f"High precipitation for {month} {year} "))
                    low = np.float64(input(f"Low precipitation for {month} {year} "))

                    init_list.append(np.random.uniform(low=low, high=high, size=length))
                except KeyError:
                    ix += 1
    else:
        for year in years:
            for ix, month in enumerate(months):
                try:
                    length = len(df.loc[eval('"{} {}"'.format(month, year))])
                    high = np.float64(input(f"High precipitation for {month} {year} "))
                    low = np.float64(input(f"Low precipitation for {month} {year} "))

                    init_list.append(np.random.uniform(low=low, high=high, size=length))
                except KeyError:
                    ix += 1

    # Convert appended list to flat list
    for arr in init_list:
        for sub_arr in arr:
            precipitation.append(sub_arr)  # flat list
    precipitation = pd.Series(precipitation).round(1).rename("precipitation")  # Return rounded results
    return precipitation


# generate simulations
simRain = simPrecip(segment_1)
temps = simTemp(segment_1)

# Plot simulated temperatures
figure = plt.figure(figsize=(15, 5))
plt.plot(temps)
figure.tight_layout(pad=1)
figure.suptitle('Simulated Temperatures')

# Merge simulated fields
segment_1["temperatures"] = temps
segment_1["precipitation"] = simRain

segment_1["Year"] = segment_1["key_0"].dt.year
segment_1["Month"] = segment_1["key_0"].dt.month
segment_1["Day"] = segment_1["key_0"].dt.day
segment_1['quarter'] = segment_1.key_0.dt.quarter
segment_1['quarter_end'] = segment_1.key_0.dt.is_quarter_end.astype("int")
segment_1['dayofweek'] = segment_1.key_0.dt.dayofweek
segment_1['is_weekend'] = segment_1['dayofweek'].ge(5).astype(np.int8)
segment_1.drop('dayofweek', axis=1, inplace=True)

# Start scaling
scaler = MinMaxScaler((-1, 1))

temperatures = segment_1.temperatures.values.copy()
scaled_temps = temperatures.reshape(len(temperatures), 1)
scaled_temps = scaler.fit_transform(scaled_temps)

rains = segment_1.precipitation.values.copy()
scaled_rains = rains.reshape(len(rains), 1)
scaled_rains = scaler.fit_transform(scaled_temps)

del temperatures, rains

cols_to_drop = ["temperatures", "precipitation"]
segment_1.drop(cols_to_drop, axis=1, inplace=True)
segment_1["temperatures"] = scaled_temps
segment_1["precipitation"] = scaled_temps

# Split data into tran and test sets
train_len = int(len(segment_1) * 0.70)
x_train, y_train = segment_1[:train_len], target[:train_len]
x_test, y_test = segment_1[train_len:], target[train_len:]

# Compare model accuracy for XGB
best_order = [1000, (0, 0, 0)]
depths = [3, 5, 7, 9, 11]
eta = [0.1, 0.01, 0.001]
estimators = [300, 500, 700, 900, 1100]

for d in depths:
    for e in eta:
        for est in estimators:
            params = (d, e, est)
            model = XGBRegressor(d, e, est, objective="reg:squarederror", random_state=50)
            model.fit(x_train, y_train)
            preds = model.predict(x_test)
            error = mean_abs_percent_error(y_test, preds)
            print("Order: %s Error: %.2f R2: %.2f" % (params, error, abs(r2_score(y_test, preds))))
            if error < best_order[0]:
                best_order.pop(); best_order.pop()
                best_order.append(error); best_order.append(params)
print("Best order: %s Best error: %.1f" % (best_order[1], best_order[0]))

params = {"min_child_weight": [1, 5, 10],
          "gamma": [0.5, 1, 1.5, 2.5],
          "subsample": [0.6, 0.8, 1.0],
          "colsample_by_tree": [0.6, 0.8, 1.0],
          "max_depth": [3, 5, 7, 9, 11],
          "learning_rate": [0.1, 0.01, 0.001],
          "n_estimators": [300, 500, 700, 900, 1100]}

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1000)
model = XGBRegressor(5, 0.001, 500, objective="reg:squarederror")

rSearch = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error')
rSearch.fit(x_train, y_train)
print(rSearch.best_estimator_)
