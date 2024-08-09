# This script performs some preliminary data analysis on the reference input data from Inigo

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

# Exported WEAP tributary results from Cordillera Model
Tr = pd.read_csv('PEGH_Maipo_Cordillera_2021_Results_SelectedStreamflow.csv', engine='python', encoding='latin1')
Tr.drop(["River", "Scenario"], axis=1, inplace=True)
Tr["Streamflow[Cubic Meter]"] = Tr["Streamflow[Cubic Meter]"] / 1E6  # covert m3 to Mm3
Tr.rename({Tr.columns[3]: "Streamflow"}, axis='columns', inplace=True)
Tr.insert(0, "Parent", Tr["Nodes and Reaches"].apply(lambda x: re.split('(\d+)', x.split(' \\ ')[0])[0].strip()))
Tr.insert(1, "Node ID", Tr["Nodes and Reaches"].apply(lambda x: re.split('(\d+)', x.split(' \\ ')[0])[1].strip()))
Tr.insert(2, "Node Name", Tr["Nodes and Reaches"].apply(lambda x: ''.join(re.split('(\d+)', x.split(' \\ ')[1])).strip()))
idx = np.where(Tr['Node Name'] == 'Headflow')
new_names = Tr['Parent'].iloc[idx] + ' ' + Tr['Node Name'].iloc[idx]
Tr.loc[new_names.index, 'Node Name'] = new_names

#Tr.drop(["Nodes and Reaches"], axis=1, inplace=True)
Tr.groupby("Node Name")

#%% Step 2: Select subset of data based on reference time period

# PYWR data files
start_index = I.loc[(I['Week'] == 14) & (I['Year'] == 2004)].index[0]
end_index = I.loc[(I['Week'] == 11) & (I['Year'] == 2010)].index[0]
I_ref = I.iloc[start_index:end_index + 1, :]
I_ref = I_ref[I_ref.Week != 53]  # drop leap weeks for consistency with WEAP

# WEAP model results (reformat to mirror Pywr data files)

W = pd.DataFrame()
W['Timestamp'] = I_ref['Timestamp']
W['Week'] = I_ref['Week']
W['Year'] = I_ref['Year']

# selected nodes
node_names = ['Rio Colorado Inflow', 'Laguna Negra', 'Rio Volcan Inflow', 'Rio Yeso Inflow', 'Canal Queltehues Inflow',
              'EtroManzEtroCanelo Inflow']
node_names = ['Rio Colorado Headflow', 'Laguna Negra', 'Rio Maipo Headflow', 'Rio Volcan Headflow', 'Embalse_El_Yeso',
               'Canal Queltehues Inflow', 'EtroManzEtroCanelo Inflow']
node_names = ['Rio Colorado Inflow', 'Laguna Negra', 'Rio Maipo Headflow', 'Rio Volcan Headflow', 'Embalse_El_Yeso',
               'Canal Queltehues Inflow', 'EtroManzEtroCanelo Inflow']
#node_names = ['Rio Colorado Inflow', 'Catchment Inflow Node 99', 'Rio Maipo Headflow', 'Rio Volcan Headflow',
#               'Catchment Inflow Node 26','Canal Queltehues Inflow', 'EtroManzEtroCanelo Inflow']

#node_names = ['Catchment Inflow Node 175', 'Catchment Inflow Node 99', 'Catchment Inflow Node 1', 'Catchment Inflow Node 245',
#               'Catchment Inflow Node 26','Canal Queltehues Inflow', 'EtroManzEtroCanelo Inflow']
# node_names = ['Colorado_antes_Maipo (gauge)', 'Laguna Negra', 'Maipo_en_Hualtatas (gauge)', 'QBocatomaVolcan_Nat (gauge)',
#               'Embalse_El_Yeso', 'MaipoManzano_NAT (gauge)', 'MaipoSanAlfonso_NAT (gauge)']
node_names = ['Colorado_antes_Maipo (gauge)', 'Laguna Negra', 'Maipo_en_Hualtatas (gauge)', 'QBocatomaVolcan_Nat (gauge)',
              'Embalse_El_Yeso', 'Canal Las Lajas II Inflow', 'EtroManzEtroCanelo Inflow', 'Canal Queltehues Inflow',
              'Canal Volcan Inflow']
node_names = ['Colorado_antes_Maipo (gauge)', 'Laguna Negra', 'Maipo_en_Hualtatas (gauge)', 'QBocatomaVolcan_Nat (gauge)',
              'Embalse_El_Yeso', 'Canal Las Lajas II Inflow', 'EtroManzEtroCanelo Inflow', 'Canal Queltehues Inflow',
              'Canal Volcan Inflow', 'Maipo_en_Hualtatas (gauge)', 'Maipo_en_SanAlfonso (gauge)', 'MaipoSanAlfonso_NAT (gauge)',
              'Maipo_en_Manzano (gauge)', 'MaipoManzano_NAT (gauge)']
node_names = ['Colorado_antes_Maipo (gauge)', 'Laguna Negra', 'Maipo_en_Hualtatas (gauge)', 'QBocatomaVolcan_Nat (gauge)',
              'Embalse_El_Yeso', 'Canal Queltehues Inflow']


for i, n in enumerate(node_names):
    x = Tr.groupby("Node Name").get_group(n).reset_index(drop=True)
    start_index = x.loc[(x['Week'] == 14) & (x['Year'] == 2004)].index[0]
    end_index = x.loc[(x['Week'] == 11) & (x['Year'] == 2010)].index[0]

    W[n] = x.iloc[start_index:end_index + 1, :].reset_index(drop=True)['Streamflow'].values

W['MAIPOEXTRA'] = W.iloc[:, 8::].sum(axis=1)  # W.iloc[:, -1] + W.iloc[:, -2]
# W.drop(['Canal Queltehues Inflow', 'EtroManzEtroCanelo Inflow'], axis=1, inplace=True)
W.drop(W.columns[8:-1].values, axis=1, inplace=True)

# identify Rio Maipo Tributaries
Tr.loc[Tr["Parent"] == "Rio Maipo", :]["Node Name"].unique()

#%% Step 3: Graph results 1

# Time series (line plot)

# Pywr data
plt.figure()
I_ref.plot(x="Timestamp", y=I_ref.columns[3::],
        kind="line", figsize=(9, 6))
plt.xlabel('Date')
plt.ylabel('Headflow (MCM/week) (?)')
plt.title('Headflow vs. Time from Pywr Data Folder (' + scenario + ')')
plt.show()

# WEAP results
plt.figure()
W.plot(x="Timestamp", y=W.columns[3::], kind="line", figsize=(9, 6))
plt.xlabel('Date')
plt.ylabel('Headflow (MCM/week)')
plt.title('Headflow vs. Time from WEAP: PEGH_Maipo_Cordillera_2021-vCGC_CC')
plt.show()

#%% Step 3: Graph results 2

# Headflow by source over reference period (bar graph)

# Pywr data
plt.figure(figsize=(11, 6))
plt.bar(I_ref.columns[3::], I_ref.iloc[:, 3::].sum(axis=0))
plt.ylabel('Headflow (MCM/week)')
plt.title('Total Headflow over Reference Period from Pywr Data Folder \n (' + scenario + ')')
plt.show()

# WEAP results
plt.figure(figsize=(11, 6))
#plt.bar(W.columns[3:-1], W.iloc[:, 3:-1].sum(axis=0))
plt.bar(['Colorado_antes\n_Maipo (gauge)', 'Laguna Negra', 'Maipo_en\n_Hualtatas(gauge)', 'QBocatomaVolcan\n_Nat(gauge)', 'Embalse_El_Yeso', 'MAIPOEXTRA'],
        W.iloc[:, 3::].sum(axis=0))
plt.ylabel('Headflow (MCM/week)')
plt.title('Total Headflow over Reference Period from WEAP: PEGH_Maipo_Cordillera_2021-vCGC_CC')
plt.show()

#%% Step 3: Graph results 3

# Headflow by source on Wk 14 2004 (bar graph)

# Pywr data
plt.figure(figsize=(9, 6))
plt.bar(I_ref.columns[3:-1], I_ref.iloc[0, 3:-1])
plt.ylabel('Headflow (MCM/week)')
plt.title('Headflow on Wk 14 2004  from Pywr Data Folder \n (' + scenario + ')')
plt.show()

# WEAP results


#%% Step 3: Graph results 4

# Proportion of magnitude by source for first day (pie chart)
def my_fmt(x):
    print(x)
    return '{:.1f}%\n({:.1f})'.format(x, total*x/100)

plt.figure(figsize=(9, 6))
total = I_ref.iloc[0, 3::].values.sum()
plt.pie(I_ref.iloc[0, 3::], labels=I_ref.columns[3::], autopct=my_fmt)
plt.title('Headflow on first day from Pywr Data Folder \n(' + scenario + ')')
plt.show()

# Proportion of magnitude by source over reference period
plt.figure(figsize=(9, 6))
total = I_ref.iloc[:, 3::].values.sum()
plt.pie(I_ref.iloc[:, 3::].sum(axis=0), labels=data_names, autopct=my_fmt)
plt.title('Headflow over Reference Period from Pywr Data Folder \n(' + scenario + ')')
plt.show()
