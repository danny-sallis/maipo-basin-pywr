import pandas as pd
import numpy as np
import datetime
from datetime import timedelta, date
import matplotlib.pyplot as plt
import re

#%% Step 1: Load the datafiles

# PYWR data files
data_names = ['COLORADO', 'LAGUNANEGRA', 'MAIPO', 'VOLCAN', 'YESO', 'MAIPOEXTRA']
scenarios = pd.read_csv('data/COLORADO.csv').columns[1::]
scenario = 'RCP_45_Pp_50_Temp_50'  # lower average warming scenario

for j, s in enumerate(scenarios):
    scenario = s

    I = pd.DataFrame()

    for i, river in enumerate(data_names):
        I[river] = pd.read_csv('data/' + river + '.csv', usecols=[scenario])

    # add updated timestamp column reflecting weekly time steps
    start_date = datetime.datetime.strptime(pd.read_csv('data/' + river + '.csv', usecols=['Timestamp']).iloc[0,0], '%d-%m-%Y')
    week_no = np.array(range(len(I)))
    I['Timestamp'] = start_date + week_no * timedelta(days=7)
    I.insert(0, 'Timestamp', I.pop('Timestamp'))
    I.insert(1, 'Week', I['Timestamp'].dt.isocalendar().week)
    I.insert(2, 'Year', I['Timestamp'].dt.isocalendar().year)

    # PYWR data files
    start_index = I.loc[(I['Week'] == 14) & (I['Year'] == 2004)].index[0]
    end_index = I.loc[(I['Week'] == 11) & (I['Year'] == 2010)].index[0]
    I_ref = I.iloc[start_index:end_index + 1, :]
    I_ref = I_ref[I_ref.Week != 53]  # drop leap weeks for consistency with WEAP

    # Time series (line plot)

    # Pywr data
    plt.figure()
    I_ref.plot(x="Timestamp", y=I_ref.columns[3:-1],
               kind="line", figsize=(9, 6))
    plt.xlabel('Date')
    plt.ylabel('Headflow (MCM/week) (?)')
    plt.title('Headflow vs. Time from Pywr Data Folder (' + scenario + ')')
    plt.show()

    # Headflow by source over reference period (bar graph)

    # Pywr data
    plt.figure(figsize=(9, 6))
    plt.figure(figsize=(15, 6))
    plt.bar(I_ref.columns[3::], I_ref.iloc[:, 3::].sum(axis=0), color='#800080')
    plt.ylabel('Headflow (MCM)')
    plt.title('Total Headflow over Reference Period from Pywr Data Folder \n (' + scenario + ')')
    plt.show()


    # Proportion of magnitude by source for first day (pie chart)
    # def my_fmt(x):
    #     print(x)
    #     return '{:.1f}%\n({:.1f})'.format(x, total * x / 100)
    #
    #
    # plt.figure(figsize=(9, 6))
    # total = I_ref.iloc[0, 3::].values.sum()
    # plt.pie(I_ref.iloc[0, 3::], labels=I_ref.columns[3::], autopct=my_fmt)
    # plt.title('Headflow on first day from Pywr Data Folder \n(' + scenario + ')')
    # plt.show()
    #
