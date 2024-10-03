# calculate the SPI at different temporal aggregations in the Maipo basin

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

# Note: for now, I am using the same quantiles from the historical period to calculate SPI!
path_clima = 'data/UQM_weap_TempC_new/'
start_hist = 1988
end_hist = 2018
locations = pd.read_csv('data/MPC_pr_ACCESS-CM2_ssp126_1986_2100_week_UQM.csv').columns[2::]

# continue this if working to update datafile timestamps
Timestamp_convert = pd.DataFrame(pd.read_csv('data/YESO.csv')['Timestamp'])
start_date = datetime.datetime.strptime(Timestamp_convert['Timestamp'][0], '%d-%m-%Y')
week_no = np.array(range(len(Timestamp_convert)))
Timestamp_convert.insert(0, 'Timestamp_', start_date + week_no * timedelta(days=7))

# observed data
observed_data = pd.read_csv(path_clima + 'PP_semanal_ACCESS-CM2_ssp126.csv')
obs = observed_data.loc[(observed_data['year'] >= start_hist) & (observed_data['year'] <= end_hist)]
obs.insert(0, 'Date', pd.to_datetime(obs['year'].astype(str) + obs['week'].astype(str).str.zfill(2) + '-1', format='%Y%W-%w'))

models = ['ACCESS-CM2', 'CNRM-CM6-1', 'INM-CM4-8', 'INM-CM5-0',
          'MIROC6', 'MIROC-ES2L', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-MM']
experiments = ['ssp126', 'ssp245', 'ssp370', 'ssp585']

SPI_3_df = pd.DataFrame()
SPI_6_df = pd.DataFrame()
SPI_12_df = pd.DataFrame()

for i, mo in enumerate(models):
    for j, x in enumerate(experiments):
        #fig = plt.figure(figsize=(8, 10))

        # load downscaled precipitation
        I_1 = pd.read_csv(path_clima + f'PP_semanal_{mo}_{x}.csv')
        I_1.insert(0, 'Timestamp', pd.to_datetime(I_1['year'].astype(str) + I_1['week'].astype(str).str.zfill(2) + '-1', format='%Y%W-%w'))
        I_1.drop(columns=['year', 'week'], inplace=True)

        # calculate aggregate monthly precipitation in the basin
        I_ = I_1.resample(rule="M", on='Timestamp').sum()
        I_ = I_[locations].mean(axis=1)

        # calculate 3, 6, 12 month precipitation in the basin
        I_3 = I_.shift().rolling(3, min_periods=3).sum()
        I_6 = I_.shift().rolling(6, min_periods=6).sum()
        I_12 = I_.shift().rolling(12, min_periods=12).sum()

        # fit gamma distribution to historical period
        params_3 = I_3.loc[(I_3.index.year >= start_hist) & (I_3.index.year <= end_hist)].groupby(
            I_3.loc[(I_3.index.year >= start_hist) & (I_3.index.year <= end_hist)].index.month).apply(fit_gamma)
        params_6 = I_6.loc[(I_6.index.year >= start_hist) & (I_6.index.year <= end_hist)].groupby(
            I_6.loc[(I_6.index.year >= start_hist) & (I_6.index.year <= end_hist)].index.month).apply(fit_gamma)
        params_12 = I_12.loc[(I_12.index.year >= start_hist) & (I_12.index.year <= end_hist)].groupby(
            I_12.loc[(I_12.index.year >= start_hist) & (I_12.index.year <= end_hist)].index.month).apply(fit_gamma)

        # calculate the resulting SPI values for each climate scenario
        # SRI_3 = np.zeros_like(I_3.loc[I_3.index.year > end_hist]) # January 2019 - Dec 2099
        # SRI_6 = np.zeros_like(I_6.loc[I_3.index.year > end_hist])
        # SRI_12 = np.zeros_like(I_12.loc[I_12.index.year > end_hist])
        SRI_3 = np.zeros_like(I_3.iloc[303:1443])  # April 2004 - March 2099
        SRI_6 = np.zeros_like(I_6.iloc[303:1443])
        SRI_12 = np.zeros_like(I_12.iloc[303:1443])

        # calculate SPI for each month of future period
        for n, t in enumerate(I_3.iloc[303:1443].index):
            month = t.month

            SRI_3[n] = norm.ppf(gamma.cdf(I_3[t], a=params_3[month][0], loc=params_3[month][1],
                                          scale=params_3[month][2]))

            SRI_6[n] = norm.ppf(gamma.cdf(I_6[t], a=params_6[month][0], loc=params_6[month][1],
                                          scale=params_6[month][2]))

            SRI_12[n] = norm.ppf(gamma.cdf(I_12[t], a=params_12[month][0], loc=params_12[month][1],
                                           scale=params_12[month][2]))

        # replace inf and -inf with min and max values
        SRI_3 = pd.DataFrame(SRI_3, index=I_3.iloc[303:1443].index)
        SRI_3 = SRI_3.replace(-np.inf, np.NaN)
        SRI_3 = SRI_3.replace(np.NaN, np.nanmin(SRI_3))
        SRI_3 = SRI_3.replace(np.inf, np.NaN)
        SRI_3 = SRI_3.replace(np.NaN, np.nanmax(SRI_3))
        SPI_3_df[f'{mo}_{x}'] = SRI_3

        SRI_6 = pd.DataFrame(SRI_6, index=I_6.iloc[303:1443].index)
        SRI_6 = SRI_6.replace(-np.inf, np.NaN)
        SRI_6 = SRI_6.replace(np.NaN, np.nanmin(SRI_6))
        SRI_6 = SRI_6.replace(np.inf, np.NaN)
        SRI_6 = SRI_6.replace(np.NaN, np.nanmax(SRI_6))
        SPI_6_df[f'{mo}_{x}'] = SRI_6

        SRI_12 = pd.DataFrame(SRI_12, index=I_12.iloc[303:1443].index)
        SRI_12 = SRI_12.replace(-np.inf, np.NaN)
        SRI_12 = SRI_12.replace(np.NaN, np.nanmin(SRI_12))
        SRI_12 = SRI_12.replace(np.inf, np.NaN)
        SRI_12 = SRI_12.replace(np.NaN, np.nanmax(SRI_12))
        SPI_12_df[f'{mo}_{x}'] = SRI_12

SPI_3_df.reset_index(inplace=True)
SPI_6_df.reset_index(inplace=True)
SPI_12_df.reset_index(inplace=True)

# SPI_3_df['Weeks'] = SPI_3_df['Timestamp'].map(lambda x:
#                              pd.date_range(
#                                  start=pd.to_datetime(x),
#                                  end=(pd.to_datetime(x) + pd.offsets.MonthEnd()),
#                                  freq='W'))

#X = pd.DataFrame(I_1[I_1['Timestamp'].dt.year > 2018]['Timestamp'])
X = pd.DataFrame(I_1['Timestamp'][1313:6253])  # first week april 2004 (wk 14 2004) to last week march 2099 (wk13 2099)
X.insert(1, 'MY', X['Timestamp'].dt.to_period('M'))

# save SPI_3
SPI_3_df.insert(1, 'MY', SPI_3_df['Timestamp'].dt.to_period('M'))
Z = X.merge(SPI_3_df, on='MY', how='inner')
weeks = np.array(range(Z.shape[0]))
Z.insert(0, 'Timestamp', start_date + weeks * timedelta(days=1))
Z['Timestamp'] = Z['Timestamp'].dt.strftime('%d-%m-%Y')
Z.drop(['Timestamp_x', 'MY', 'Timestamp_y'], axis=1, inplace=True)
Z.to_csv('data/SPI3_2004Wk14-2099Wk13.csv', index=False)

# save SPI_6
SPI_6_df.insert(1, 'MY', SPI_6_df['Timestamp'].dt.to_period('M'))
Z = X.merge(SPI_6_df, on='MY', how='inner')
weeks = np.array(range(Z.shape[0]))
Z.insert(0, 'Timestamp', start_date + weeks * timedelta(days=1))
Z['Timestamp'] = Z['Timestamp'].dt.strftime('%d-%m-%Y')
Z.drop(['Timestamp_x', 'MY', 'Timestamp_y'], axis=1, inplace=True)
Z.to_csv('data/SPI6_2004Wk14-2099Wk13.csv', index=False)

# save SPI_12
SPI_12_df.insert(1, 'MY', SPI_12_df['Timestamp'].dt.to_period('M'))
Z = X.merge(SPI_12_df, on='MY', how='inner')
weeks = np.array(range(Z.shape[0]))
Z.insert(0, 'Timestamp', start_date + weeks * timedelta(days=1))
Z['Timestamp'] = Z['Timestamp'].dt.strftime('%d-%m-%Y')
Z.drop(['Timestamp_x', 'MY', 'Timestamp_y'], axis=1, inplace=True)
Z.to_csv('data/SPI12_2004Wk14-2099Wk13.csv', index=False)

#%% Plot time series

models = ['ACCESS-CM2', 'NorESM2-MM']
experiments = ['ssp126', 'ssp585']

for i, mo in enumerate(models):
    fig = plt.figure(figsize=(15, 6))
    for j, x in enumerate(experiments):
        for s, spi in enumerate([3, 6, 12]):
            if spi == 3:
                SPI = pd.read_csv('data/SPI3_CMIP6_2019-2099.csv')
            elif spi == 6:
                SPI = pd.read_csv('data/SPI6_CMIP6_2019-2099.csv')
            elif spi == 12:
                SPI = pd.read_csv('data/SPI12_CMIP6_2019-2099.csv')

            ax = plt.subplot(3, 2, j + (s*2) + 1)
            plt.plot(SPI['Timestamp'], SPI[f'{mo}_{x}'], color='gray')
            plt.axhline(0, linestyle='--', color='black')
            plt.axhline(-0.84, color='darkred')
            zero = np.zeros(SPI.shape[0])
            plt.fill_between(SPI['Timestamp'], SPI[f'{mo}_{x}'], zero, where=(SPI[f'{mo}_{x}'] >= zero),
                             color='steelblue', alpha=0.8, interpolate=True)
            plt.fill_between(SPI['Timestamp'], SPI[f'{mo}_{x}'], zero, where=(SPI[f'{mo}_{x}'] < zero),
                             color='chocolate', alpha=0.8, interpolate=True)
            plt.ylim([-7, 7])
            plt.xticks(np.array(range(0, len(SPI), 10*52)), np.array(range(2019, 2100, 10)))
            plt.title(f'SPI-{spi}: {x}')

    plt.suptitle(f'SPI for {mo}')
    fig.supxlabel('Date')
    fig.supylabel('SPI')
    plt.tight_layout()
    plt.show()

