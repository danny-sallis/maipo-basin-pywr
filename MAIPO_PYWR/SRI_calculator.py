# calculate the SRI at different temporal aggregations in the Maipo basin

import pandas as pd
import numpy as np
import datetime
from datetime import timedelta, date
from scipy.stats import norm
from scipy.stats import gamma
import matplotlib.pyplot as plt


def fit_gamma(data):
    params = gamma.fit(data.dropna())  # shape, loc, scale

    return params


# load the data
I_Yeso = pd.read_csv('data/YESO.csv')
I_Volcan = pd.read_csv('data/VOLCAN.csv')
I_Colorado = pd.read_csv('data/COLORADO.csv')
I_Maipo = pd.read_csv('data/MAIPO.csv')
I_Negra = pd.read_csv('data/LAGUNANEGRA.csv')
I_Extra = pd.read_csv('data/MAIPOEXTRA.csv')

# calculate total weekly inflow in the basin
I = I_Yeso.add(I_Volcan, fill_value=0)
I = I.add(I_Colorado, fill_value=0)
I = I.add(I_Maipo, fill_value=0)
I = I.add(I_Negra, fill_value=0)
I = I.add(I_Extra, fill_value=0)
I['Timestamp'] = I_Yeso['Timestamp']

# add updated timestamp column reflecting weekly time steps
start_date = datetime.datetime.strptime(I['Timestamp'][0], '%d-%m-%Y')
week_no = np.array(range(len(I)))
I['Timestamp_'] = start_date + week_no * timedelta(days=7)
I = I.drop(['Timestamp'], axis=1)
first_column = I.pop('Timestamp_')
I.insert(0, 'Timestamp_', first_column)

# calculate aggregate monthly inflow in the basin
I_ = I.resample(rule="M", on='Timestamp_').sum()

# calculate 3, 6, 12 month inflows in the basin
I_3 = I_.shift().rolling(3, min_periods=3).sum()
I_6 = I_.shift().rolling(6, min_periods=6).sum()
I_12 = I_.shift().rolling(12, min_periods=12).sum()

# fit gamma distribution for each month of each climate scenario
params_3 = []
params_6 = []
params_12 = []
for n in range(I_.shape[1]):  # fit gamma for each month and climate scenario
    params_3.append(I_3.iloc[:, n].groupby(by=I_3.index.month).apply(fit_gamma))
    params_6.append(I_6.iloc[:, n].groupby(by=I_6.index.month).apply(fit_gamma))
    params_12.append(I_12.iloc[:, n].groupby(by=I_12.index.month).apply(fit_gamma))

# calculate the resulting SRI values for each climate scenario
SRI_3 = np.zeros_like(I_3)
CDF_3 = np.zeros_like(I_3)
SRI_6 = np.zeros_like(I_6)
SRI_12 = np.zeros_like(I_12)
for n in range(I_.shape[1]):  # for climate scenario
    for m in range(I_.shape[0]):
        month = I_3.index[m].month

        SRI_3[m, n] = norm.ppf(gamma.cdf(I_3.iloc[m, n], a=params_3[n][month][0], loc=params_3[n][month][1], scale=params_3[n][month][2]))

        SRI_6[m, n] = norm.ppf(gamma.cdf(I_6.iloc[m, n], a=params_6[n][month][0], loc=params_6[n][month][1], scale=params_6[n][month][2]))

        SRI_12[m, n] = norm.ppf(gamma.cdf(I_12.iloc[m, n], a=params_12[n][month][0], loc=params_12[n][month][1], scale=params_12[n][month][2]))

# replace inf and -inf with min and max values
for n in range(I_.shape[1]):  # for climate scenario
    SRI_3 = pd.DataFrame(SRI_3, columns=I.columns[1::])
    SRI_3.iloc[3::, n] = SRI_3.iloc[3::, n].replace(-np.inf, np.NaN)
    SRI_3.iloc[3::, n] = SRI_3.iloc[3::, n].replace(np.NaN, np.nanmin(SRI_3.iloc[3::, n], axis=0))
    SRI_3.iloc[3::, n] = SRI_3.iloc[3::, n].replace(np.inf, np.NaN)
    SRI_3.iloc[:, n] = SRI_3.iloc[:, n].replace(np.NaN, np.nanmax(SRI_3.iloc[3::, n], axis=0))

    SRI_6 = pd.DataFrame(SRI_6, columns=I.columns[1::])
    SRI_6.iloc[6::, n] = SRI_6.iloc[6::, n].replace(-np.inf, np.NaN)
    SRI_6.iloc[6::, n] = SRI_6.iloc[6::, n].replace(np.NaN, np.nanmin(SRI_6.iloc[6::, n], axis=0))
    SRI_6.iloc[6::, n] = SRI_6.iloc[6::, n].replace(np.inf, np.NaN)
    SRI_6.iloc[:, n] = SRI_6.iloc[:, n].replace(np.NaN, np.nanmax(SRI_6.iloc[6::, n], axis=0))

    SRI_12 = pd.DataFrame(SRI_12, columns=I.columns[1::])
    SRI_12.iloc[12::, n] = SRI_12.iloc[12::, n].replace(-np.inf, np.NaN)
    SRI_12.iloc[12::, n] = SRI_12.iloc[12::, n].replace(np.NaN, np.nanmin(SRI_12.iloc[12::, n], axis=0))
    SRI_12.iloc[12::, n] = SRI_12.iloc[12::, n].replace(np.inf, np.NaN)
    SRI_12.iloc[:, n] = SRI_12.iloc[:, n].replace(np.NaN, np.nanmax(SRI_12.iloc[12::, n], axis=0))

# convert to dataframe and save in the data folder
X = pd.DataFrame(I['Timestamp_'])
X['MY'] = X['Timestamp_'].dt.to_period('M')

# create final SRI-3 dataframe and save to csv
SRI_3['Timestamp_'] = I_3.index
SRI_3['MY'] = SRI_3['Timestamp_'].dt.to_period('M')
Z = X.merge(SRI_3, on='MY', how='inner')
Z = Z.drop(['Timestamp__x', 'MY', 'Timestamp__y'], axis=1)
first_column = I_Yeso['Timestamp']
Z.insert(0, 'Timestamp', first_column)
# save with original timestamp convention
Z.to_csv('data/SRI3.csv', index=False)

# create final SRI-6 dataframe and save to csv
SRI_6['Timestamp_'] = I_6.index
SRI_6['MY'] = SRI_6['Timestamp_'].dt.to_period('M')
Z = X.merge(SRI_6, on='MY', how='inner')
Z = Z.drop(['Timestamp__x', 'MY', 'Timestamp__y'], axis=1)
first_column = I_Yeso['Timestamp']
Z.insert(0, 'Timestamp', first_column)
# save with original timestamp convention
Z.to_csv('data/SRI6.csv', index=False)

# create final SRI-12 dataframe and save to csv
SRI_12['Timestamp_'] = I_12.index
SRI_12['MY'] = SRI_12['Timestamp_'].dt.to_period('M')
Z = X.merge(SRI_12, on='MY', how='inner')
Z = Z.drop(['Timestamp__x', 'MY', 'Timestamp__y'], axis=1)
first_column = I_Yeso['Timestamp']
Z.insert(0, 'Timestamp', first_column)
# save with original timestamp convention
Z.to_csv('data/SRI12.csv', index=False)

#%% plot SRI time series for entire date range
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load drought indicators
SRI_3 = pd.read_csv('data/SRI3.csv')
SRI_6 = pd.read_csv('data/SRI6.csv')
SRI_12 = pd.read_csv('data/SRI12.csv')
zero = np.zeros((len(SRI_3)))

# updated timestamp column to reflecting weekly time steps
start_date = datetime.datetime.strptime(SRI_3['Timestamp'][0], '%d-%m-%Y')
week_no = np.array(range(len(SRI_3)))
t = start_date + week_no * timedelta(days=7)

# specify climate scenarios to plot
scenarios = [3, 13]
#scenarios = [1, 11]

# make plots of SRI-3,-6,and -12 for selected climate change scenarios
for i in scenarios:
    plt.figure()
    plt.subplots(3, 1)

    # SRI-3
    plt.subplot(3, 1, 1)
    plt.plot(t, SRI_3.iloc[:, i], color='gray')
    plt.axhline(0, linestyle='--', color='black')
    plt.axhline(-0.84, color='darkred')
    plt.fill_between(t, SRI_3.iloc[:, i], zero, where=(SRI_3.iloc[:, i] >= zero), color='steelblue', alpha=0.8, interpolate=True)
    plt.fill_between(t, SRI_3.iloc[:, i], zero, where=(SRI_3.iloc[:, i] < zero), color='chocolate', alpha=0.8, interpolate=True)
    plt.ylabel('SRI-3')
    plt.xticks([])
    plt.ylim([-3, 3])
    plt.xlim([datetime.date(2020, 4, 1), datetime.date(2040, 8, 1)])
    plt.title(SRI_3.columns[i])

    # SRI-6
    plt.subplot(3, 1, 2)
    plt.plot(t, SRI_6.iloc[:, i], color='gray')
    plt.axhline(0, linestyle='--', color='black')
    plt.axhline(-0.84, color='darkred')
    plt.fill_between(t, SRI_6.iloc[:, i], zero, where=(SRI_6.iloc[:, i] >= zero), color='steelblue', alpha=0.8, interpolate=True)
    plt.fill_between(t, SRI_6.iloc[:, i], zero, where=(SRI_6.iloc[:, i] < zero), color='chocolate', alpha=0.8, interpolate=True)
    plt.ylabel('SRI-6')
    plt.xticks([])
    plt.xlim([datetime.date(2020, 4, 1), datetime.date(2040, 8, 1)])
    plt.ylim([-3, 3])

    # SRI-12
    plt.subplot(3, 1, 3)
    plt.plot(t, SRI_12.iloc[:, i], color='gray')
    plt.axhline(0, linestyle='--', color='black')
    plt.axhline(-0.84, color='darkred')
    plt.fill_between(t, SRI_12.iloc[:, i], zero, where=(SRI_12.iloc[:, i] >= zero), color='steelblue', alpha=0.8, interpolate=True)
    plt.fill_between(t, SRI_12.iloc[:, i], zero, where=(SRI_12.iloc[:, i] < zero), color='chocolate', alpha=0.8, interpolate=True)
    plt.xlim([datetime.date(2020, 4, 1), datetime.date(2040, 8, 1)])
    plt.ylim([-3, 3])
    plt.ylabel('SRI-12')
    plt.xlabel('Time')

    plt.show()

#%% #%% plot SRI time series for ranges by START and END date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load drought indicators
SRI_3 = pd.read_csv('data/SRI3.csv')
SRI_6 = pd.read_csv('data/SRI6.csv')
SRI_12 = pd.read_csv('data/SRI12.csv')
k1 = -1.5  # threshold

# updated timestamp column to reflecting weekly time steps
start_date = datetime.datetime.strptime(SRI_3['Timestamp'][0], '%d-%m-%Y')
week_no = np.array(range(len(SRI_3)))
t = start_date + week_no * timedelta(days=7)

# corresponding date indices
"""
(a) 2020-2030:
    "start": "2006-07-12",
    "end": "2007-12-13",
(b) 2040-2050:
    "start": "2009-05-17",
    "end": "2010-10-18"
"""
periods = [[datetime.datetime.strptime("2006-07-12", '%Y-%m-%d'), datetime.datetime.strptime("2007-12-13", '%Y-%m-%d')],
           [datetime.datetime.strptime("2009-05-17", '%Y-%m-%d'), datetime.datetime.strptime("2010-10-18", '%Y-%m-%d')]]
periods_index = [[np.where(SRI_3['Timestamp'] == "12-07-2006")[0][0], np.where(SRI_3['Timestamp'] == "13-12-2007")[0][0]],
           [np.where(SRI_3['Timestamp'] == "17-05-2009")[0][0], np.where(SRI_3['Timestamp'] == "18-10-2010")[0][0]]]


# specify climate scenarios to plot
scenarios = [3, 13]
scenarios = [1]
scenarios = (np.arange(15) + 1).tolist()
week_no = np.arange(len(SRI_3)) % 52 + 1

# make plots of SRI-3,-6,and -12 for selected climate change scenarios
for i in scenarios:
    plt.figure()
    plt.subplots(3, 2)
    for p in range(len(periods)):

        zero = np.zeros((len(SRI_3.iloc[periods_index[p][0]:periods_index[p][1], 0])))
        week_no = np.arange(len(t[periods_index[p][0]:periods_index[p][1]])) % 52 + 1
        AprOct_idx = (week_no == 1) + (week_no == 27)

        # SRI-3
        plt.subplot(3, 2, p+1)
        plt.plot(t[periods_index[p][0]:periods_index[p][1]], SRI_3.iloc[periods_index[p][0]:periods_index[p][1], i], color='gray')
        plt.axhline(0, linestyle='--', color='black')
        plt.axhline(k1, color='darkred')
        threshold_exceeded = SRI_3.iloc[periods_index[p][0]:periods_index[p][1], i] <= k1
        threshold_AprOct = threshold_exceeded[AprOct_idx]
        threshold_idx = threshold_AprOct[threshold_AprOct == True].index
        plt.scatter(t[threshold_idx], SRI_3.iloc[periods_index[p][0]:periods_index[p][1], i][threshold_idx], color='k')
        plt.fill_between(t[periods_index[p][0]:periods_index[p][1]], SRI_3.iloc[periods_index[p][0]:periods_index[p][1], i], zero, where=(SRI_3.iloc[periods_index[p][0]:periods_index[p][1], i] >= zero), color='steelblue', alpha=0.8, interpolate=True)
        plt.fill_between(t[periods_index[p][0]:periods_index[p][1]], SRI_3.iloc[periods_index[p][0]:periods_index[p][1], i], zero, where=(SRI_3.iloc[periods_index[p][0]:periods_index[p][1], i] < zero), color='chocolate', alpha=0.8, interpolate=True)
        plt.ylabel('SRI-3')
        plt.xticks([])
        plt.ylim([-3, 3])
        plt.xlim(t[periods_index][p])

        # SRI-6
        plt.subplot(3, 2, p+3)
        plt.plot(t[periods_index[p][0]:periods_index[p][1]], SRI_6.iloc[periods_index[p][0]:periods_index[p][1], i], color='gray')
        plt.axhline(0, linestyle='--', color='black')
        plt.axhline(k1, color='darkred')
        threshold_exceeded = SRI_6.iloc[periods_index[p][0]:periods_index[p][1], i] <= k1
        threshold_AprOct = threshold_exceeded[AprOct_idx]
        threshold_idx = threshold_AprOct[threshold_AprOct == True].index
        plt.scatter(t[threshold_idx], SRI_6.iloc[periods_index[p][0]:periods_index[p][1], i][threshold_idx], color='k')
        plt.fill_between(t[periods_index[p][0]:periods_index[p][1]], SRI_6.iloc[periods_index[p][0]:periods_index[p][1], i], zero, where=(SRI_6.iloc[periods_index[p][0]:periods_index[p][1], i] >= zero), color='steelblue', alpha=0.8, interpolate=True)
        plt.fill_between(t[periods_index[p][0]:periods_index[p][1]], SRI_6.iloc[periods_index[p][0]:periods_index[p][1], i], zero, where=(SRI_6.iloc[periods_index[p][0]:periods_index[p][1], i] < zero), color='chocolate', alpha=0.8, interpolate=True)
        plt.ylabel('SRI-6')
        plt.xticks([])
        plt.xlim(t[periods_index][p])
        plt.ylim([-3, 3])

        # SRI-12
        plt.subplot(3, 2, p+5)
        plt.plot(t[periods_index[p][0]:periods_index[p][1]], SRI_12.iloc[periods_index[p][0]:periods_index[p][1], i], color='gray')
        plt.axhline(0, linestyle='--', color='black')
        plt.axhline(k1, color='darkred')
        threshold_exceeded = SRI_12.iloc[periods_index[p][0]:periods_index[p][1], i] <= k1
        threshold_AprOct = threshold_exceeded[AprOct_idx]
        threshold_idx = threshold_AprOct[threshold_AprOct == True].index
        plt.scatter(t[threshold_idx], SRI_12.iloc[periods_index[p][0]:periods_index[p][1], i][threshold_idx], color='k')
        plt.fill_between(t[periods_index[p][0]:periods_index[p][1]], SRI_12.iloc[periods_index[p][0]:periods_index[p][1], i], zero, where=(SRI_12.iloc[periods_index[p][0]:periods_index[p][1], i] >= zero), color='steelblue', alpha=0.8, interpolate=True)
        plt.fill_between(t[periods_index[p][0]:periods_index[p][1]], SRI_12.iloc[periods_index[p][0]:periods_index[p][1], i], zero, where=(SRI_12.iloc[periods_index[p][0]:periods_index[p][1], i] < zero), color='chocolate', alpha=0.8, interpolate=True)
        plt.xlim(t[periods_index][p])
        plt.ylim([-3, 3])
        plt.ylabel('SRI-12')
        plt.xlabel('Time')

    plt.suptitle(f'Climate Change Scenario: {SRI_3.columns[i]}')
    plt.gcf().set_size_inches(11, 4)
    plt.show()



#%% sample contract value output plot

SRI_3 = pd.read_csv('data/SRI3.csv')
threshold = -0.84
contract_values = SRI_3
week_no = contract_values.index % 52
week_no += 1
contract_values.iloc[:, 1::] = 50 * (contract_values.iloc[:, 1::] < -0.84)
index_notAprOct = np.where((contract_values.index % 52 + 1 != 1) & (contract_values.index + 1 % 52 != 27))[0]
contract_values.iloc[index_notAprOct, 1::] = np.NaN
contract_values = contract_values.fillna(method='ffill')
contract_values.set_index('Timestamp', inplace=True)

plt.figure()
contract_values.plot()
plt.show()

plt.figure()
contract_values.iloc[:, 14].plot()
plt.show()
