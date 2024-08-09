#% Initial simulation results figures for AGU 2023

"""
1. We test two different time periods which requires updating sc_MAIPO_sim_AGU.json
    (a) 2020-2030:
        "start": "2006-07-12",
        "end": "2007-12-13",
    (b) 2040-2050:
        "start": "2009-05-17",
        "end": "2010-10-18",
2. We test three different temporal aggregations of the SRI which requires updating "drought_status" in sc_MAIPO_sim_AGU.json
    (a) "url" : "data/SRI3.csv"
    (b) "url" : "data/SRI6.csv"
    (c) "url" : "data/SRI12.csv"
3. We test three different water option contract policies (hard coded policy parameter)
    (a) policy 1 (baseline):
        {if <=-0.84, contracts 30}
        {else 0}
    (b) policy 2:
        {if -0.5, contracts 10}
        {if -0.84, contracts 30}
        {if -1.15, contracts 50}
        {else 0}
    (c) policy 3:
        {if -1.15, contracts 10}
        {if -1.5, contracts 30}
        {if -1.85, contracts 50}
        {else 0}
"""

#%% Load the simulation result outputs into a dictionary

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import datetime
from datetime import timedelta, date
import os

# set root directory
os.chdir("/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR")

# specify scenarios
sri_temps = [3, 6, 12]  # months
policies = [0, 1, 2, 3, 4]  # see definitions above
simulation_periods = ['2020_2030', '2040_2050']

# recall climate change scenario names
ref = pd.read_csv('data/YESO.csv')
SRI_3 = pd.read_csv('data/SRI3.csv')
climate_names = ref.columns[1::]

data = {}
for s in sri_temps:
    for p in policies:
        for t in simulation_periods:
            DIR = f'outputs/AGU_2023_results/SRI{s}/Policy{p}/{t}'
            d = pd.read_excel(DIR + '/metrics.xlsx', 'scenarios', header=0, index_col=0)
            rel = 1 - d.loc["failure_frequency_PT1"].values
            d.loc['Reliability_PT1'] = rel
            d.columns = climate_names
            data.update({f's{s}_p{p}_t{t}': d})


# create a multi-layer dictionary to better organize the data

my_dictionary = {
    "2020-2030": {
        "Policy0": {
            "SRI3": {
                "metrics": data['s3_p0_t2020_2030'],
            },
            "SRI6": {
                "metrics": data['s6_p0_t2020_2030'],
            },
            "SRI12": {
                "metrics": data['s12_p0_t2020_2030'],
            },
        },
        "Policy1": {
            "SRI3": {
                "metrics": data['s3_p1_t2020_2030'],
            },
            "SRI6": {
                "metrics": data['s6_p1_t2020_2030'],
            },
            "SRI12": {
                "metrics": data['s12_p1_t2020_2030'],
            },
        },
        "Policy2": {
                    "SRI3": {
                        "metrics": data['s3_p2_t2020_2030'],
                    },
                    "SRI6": {
                        "metrics": data['s6_p2_t2020_2030'],
                    },
                    "SRI12": {
                        "metrics": data['s12_p2_t2020_2030'],
                    },
                },
        "Policy3": {
                    "SRI3": {
                        "metrics": data['s3_p3_t2020_2030'],
                    },
                    "SRI6": {
                        "metrics": data['s6_p3_t2020_2030'],
                    },
                    "SRI12": {
                        "metrics": data['s12_p3_t2020_2030'],
                    },
                },
        "Policy4": {
                    "SRI3": {
                        "metrics": data['s3_p4_t2020_2030'],
                    },
                    "SRI6": {
                        "metrics": data['s6_p4_t2020_2030'],
                    },
                    "SRI12": {
                        "metrics": data['s12_p4_t2020_2030'],
                    },
                },
            },
    "2040-2050": {
        "Policy0": {
            "SRI3": {
                "metrics": data['s3_p0_t2040_2050'],
            },
            "SRI6": {
                "metrics": data['s6_p0_t2040_2050'],
            },
            "SRI12": {
                "metrics": data['s12_p0_t2040_2050'],
            },
        },
        "Policy1": {
            "SRI3": {
                "metrics": data['s3_p1_t2040_2050'],
            },
            "SRI6": {
                "metrics": data['s6_p1_t2040_2050'],
            },
            "SRI12": {
                "metrics": data['s12_p1_t2040_2050'],
            },
        },
        "Policy2": {
            "SRI3": {
                "metrics": data['s3_p2_t2040_2050'],
            },
            "SRI6": {
                "metrics": data['s6_p2_t2040_2050'],
            },
            "SRI12": {
                "metrics": data['s12_p2_t2040_2050'],
            },
        },
        "Policy3": {
            "SRI3": {
                "metrics": data['s3_p3_t2040_2050'],
            },
            "SRI6": {
                "metrics": data['s6_p3_t2040_2050'],
            },
            "SRI12": {
                "metrics": data['s12_p3_t2040_2050'],
            },
        },
        "Policy4": {
            "SRI3": {
                "metrics": data['s3_p4_t2040_2050'],
            },
            "SRI6": {
                "metrics": data['s6_p4_t2040_2050'],
            },
            "SRI12": {
                "metrics": data['s12_p4_t2040_2050'],
            },
        },
    },
}

#%% (1) Data formatting for use with violin plot

# array specifying the metrics we wish to plot
metric_names = ['Total Contracts Made', 'TotalCost', 'Max Deficit Duration', 'Reliability_PT1']

# specify scenarios
sri_temps = [3, 12]  # months
policies = [0, 1]  # see definitions above
simulation_periods = ['2020-2030', '2040-2050']
periods_index = [[np.where(SRI_3['Timestamp'] == "12-07-2006")[0][0], np.where(SRI_3['Timestamp'] == "13-12-2007")[0][0]],
           [np.where(SRI_3['Timestamp'] == "17-05-2009")[0][0], np.where(SRI_3['Timestamp'] == "18-10-2010")[0][0]]]


# number of climate change scenarios
n = np.shape(my_dictionary['2020-2030']['Policy1']['SRI3']['metrics'])[1]  # number climate scenarios

# loading data-set to be compatible with seaborn
df_small = {}  # preallocate dictionary to hold dataframes
dict_metrics = {}
for m in metric_names:
    metric_append = []
    for i, s in enumerate(sri_temps):
        for p in policies:
            for t in simulation_periods:
                metric_small = pd.DataFrame(my_dictionary[t][f'Policy{p}'][f'SRI{s}']['metrics'].loc[m])
                metric_small['Period'] = np.repeat(t, n)
                metric_small['SRI'] = np.repeat(s, n)
                metric_small['Policy'] = np.repeat(p, n)
                metric_small['XOrder'] = np.repeat(i * 3 + p, n)

                metric_append.append(metric_small)

    dict_metrics.update({m: pd.concat(metric_append, axis=0)})
    if m == 'TotalCost':
        dict_metrics['TotalCost']['TotalCost'] = dict_metrics['TotalCost']['TotalCost']/1E6

#%% plot the data in a violin plot

metric_names = ['TotalCost', 'Reliability_PT1']
metric_ylabel = ['TotalCost [$M]', 'Urban Supply Reliability']
metric_names = ['Total Contracts Made', 'TotalCost', 'Max Deficit Duration', 'Reliability_PT1']
metric_ylabel = metric_names
sri_temps = [3, 12]  # months
policies = [1, 4]  # see definitions above
simulation_periods = ['2020-2030', '2040-2050']

# use to set style of background of plot
seaborn.set(style="whitegrid")
colors = ["#FDEEBF7", "#D79DAE"]
colors = [[0.8705882352941177, 0.9215686274509803, 0.9686274509803922],
          [0.8431372549019608, 0.615686274509804, 0.6823529411764706]]
seaborn.set_palette(seaborn.color_palette(colors))

# violin plot by category
plt.figure(figsize=(20, 10))
plt.subplots(len(metric_names), 1)

for i, m in enumerate(metric_names):
    plt.subplot(np.int64(f'{len(metric_names)}1{i+1}'))
    if m == 'Reliability_PT1':
        ax = seaborn.violinplot(data=dict_metrics[m], x='XOrder', y=m, hue='Period', cut=0, split=True, gap=0.2, inner='box')
    else:
        ax = seaborn.violinplot(data=dict_metrics[m], x='XOrder', y=m, hue='Period', cut=0, split=True, gap=0.2, inner='box')

    plt.ylabel(metric_ylabel[i])
    for j in range(len(policies)):
        plt.axvline(x=j * len(policies) + len(policies) - 0.5, color='darkgray')
    ax.set_xticks(range(len(policies)*len(sri_temps)), labels=np.tile(['Policy' + str(p) for p in policies], len(sri_temps)))
    ax.set(xlabel=None)
    if i < len(metric_names) - 1:
        plt.xticks([])
        #ax.legend_.remove()
    else:
        ax.legend_.remove()
        for j, s in enumerate(sri_temps):
            plt.text(0.5 + j*len(policies), ax.get_ylim()[0] - 0.08, f'SRI{s}', ha='center', fontweight='bold')
plt.suptitle('')
plt.show()

#seaborn.violinplot(x=m, y="value", hue="period", data=dict_metrics['m'], palette="Set2", dodge=True)

# ======================================================================================================================

#%% Approach 2: calculate results by hand using data for seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import datetime
from datetime import timedelta, date
import os
import tables
from pywr.model import Model
from pywr.notebook import draw_graph
from pywr.recorders import TablesRecorder, Recorder
from pywr.parameters import IndexParameter, Parameter, load_parameter

# set root directory
os.chdir("/Users/keaniw/Documents/Research/Chile Project/MAIPO_PYWR")

# specify scenarios
sri_temps = [3, 6, 12]  # months
policies = [0, 1, 2, 3, 4]  # see definitions above
simulation_periods = ['2020_2030', '2040_2050']
discount_rate = 0.035
m = Model.load("sc_MAIPO_sim_AGU.json")

metric_names = ['Total Deficit', 'Failure Frequency', 'Urban Reliability', 'Total Contracts',
                'Total Shortage Cost', 'Total Contract Cost', 'Total Cost']

# recall climate change scenario names and simulation indices
ref = pd.read_csv('data/YESO.csv')
climate_names = ref.columns[1::]
n = len(climate_names)  # number climate scenarios
periods_index = [[np.where(ref['Timestamp'] == "12-07-2006")[0][0], np.where(ref['Timestamp'] == "13-12-2007")[0][0]],
           [np.where(ref['Timestamp'] == "17-05-2009")[0][0], np.where(ref['Timestamp'] == "18-10-2010")[0][0]]]

# load demand data
extra_data = pd.read_csv('data/Extra data.csv', index_col=0)
demands = extra_data['PT1']

data_ = {}  # preallocate dictionary to hold dataframes
for i, s in enumerate(sri_temps):
    for p in policies:
        for j, t in enumerate(simulation_periods):

            OUTPUT_DIR = f'outputs/AGU_2023_results/SRI{s}/Policy{p}/{t}'
            d = pd.read_excel(OUTPUT_DIR + '/metrics.xlsx', 'scenarios', header=0, index_col=0)

            dem = demands[periods_index[j][0]:periods_index[j][1]]

            # load simulation results into a table
            with tables.open_file(os.path.join(OUTPUT_DIR, 'flows.h5')) as h5:
                tbl = h5.get_node('/time')
                date_index = pd.to_datetime({k: tbl.col(k) for k in ('year', 'month', 'day')})
                data = {}
                for ca in h5.walk_nodes('/', 'CArray'):
                    data[ca._v_name] = pd.DataFrame(ca.read(), index=date_index, columns=m.scenarios.multiindex)
            df = pd.concat(data, axis=1)

            # week indices
            week_no = np.arange(len(df)) % 52 + 1

            # calculate urban water supply deficits
            deficits = dem[0] - df['PT1']
            urban_deficits = deficits
            # if water option contract purchased, we assume demand is met from buying from agriculture
            ag_deficits = deficits[df['contract_value'] > 0]
            d.loc['Total Ag Deficit'] = np.nansum(ag_deficits[ag_deficits > 0], axis=0)
            urban_deficits[df['contract_value'] > 0] = 0
            # deficits[df['contract_value'] > 0] = deficits[df['contract_value'] > 0] - np.mean(deficits[deficits > 0])
            d.loc['Total Urban Deficit'] = np.nansum(urban_deficits[urban_deficits > 0], axis=0)
            d.loc['Total Ag Deficit'] = np.nansum(ag_deficits[ag_deficits > 0], axis=0)
            d.loc['Failure Frequency'] = np.sum(urban_deficits > 0) / len(urban_deficits)
            d.loc['Urban Reliability'] = np.ones_like(d.loc['Failure Frequency']) - d.loc['Failure Frequency']

            # total contracts made
            d.loc['Total Contracts'] = np.sum(df['contract_value'])

            # costs
            decision_dp = df['DP_index']
            discount_factor = 1 / (1 + discount_rate) ** ((decision_dp - 1) * 5)
            # d.loc['Total Shortage Cost'] = np.sum(deficits * 0.925 * discount_factor)
            d.loc['Total Urban Shortage Cost'] = np.sum(urban_deficits[urban_deficits > 0] * 0.925 * discount_factor)
            d.loc['Total Ag Shortage Cost'] = np.sum(ag_deficits[ag_deficits > 0] * 0.775 * discount_factor)
            d.loc['Total Shortage Cost'] = d.loc['Total Urban Shortage Cost'] + d.loc['Total Ag Shortage Cost']
            april_contract_cost = np.nansum(df['contract_value'][week_no == 1] * d.loc["AprilContractCost"] * discount_factor, axis=0)
            october_contract_cost = np.nansum(df['contract_value'][week_no == 27] * d.loc["OctoberContractCost"] * discount_factor, axis=0)
            d.loc['Total Contract Cost'] = april_contract_cost + october_contract_cost
            d.loc['Total Cost'] = d.loc['Total Contract Cost'] + d.loc['Total Shortage Cost']
            d.loc['Total Cost'] = d.loc['Total Contract Cost'] + d.loc['Total Urban Shortage Cost']

            # plotting keys
            d.loc['Period'] = np.repeat(t, n)
            d.loc['SRI'] = np.repeat(s, n)
            d.loc['Policy'] = np.repeat(p, n)
            d.loc['XOrder'] = np.repeat(i * 3 + p, n)

            data_.update({f's{s}_p{p}_t{t}': d})

# create a multi-layer dictionary to better organize the data

my_dictionary = {
    "2020-2030": {
        "Policy0": {
            "SRI3": {
                "metrics": data_['s3_p0_t2020_2030'],
            },
            "SRI6": {
                "metrics": data_['s6_p0_t2020_2030'],
            },
            "SRI12": {
                "metrics": data_['s12_p0_t2020_2030'],
            },
        },
        "Policy1": {
            "SRI3": {
                "metrics": data_['s3_p1_t2020_2030'],
            },
            "SRI6": {
                "metrics": data_['s6_p1_t2020_2030'],
            },
            "SRI12": {
                "metrics": data_['s12_p1_t2020_2030'],
            },
        },
        "Policy2": {
                    "SRI3": {
                        "metrics": data_['s3_p2_t2020_2030'],
                    },
                    "SRI6": {
                        "metrics": data_['s6_p2_t2020_2030'],
                    },
                    "SRI12": {
                        "metrics": data_['s12_p2_t2020_2030'],
                    },
                },
        "Policy3": {
                    "SRI3": {
                        "metrics": data_['s3_p3_t2020_2030'],
                    },
                    "SRI6": {
                        "metrics": data_['s6_p3_t2020_2030'],
                    },
                    "SRI12": {
                        "metrics": data_['s12_p3_t2020_2030'],
                    },
                },
        "Policy4": {
                    "SRI3": {
                        "metrics": data_['s3_p4_t2020_2030'],
                    },
                    "SRI6": {
                        "metrics": data_['s6_p4_t2020_2030'],
                    },
                    "SRI12": {
                        "metrics": data_['s12_p4_t2020_2030'],
                    },
                },
            },
    "2040-2050": {
        "Policy0": {
            "SRI3": {
                "metrics": data_['s3_p0_t2040_2050'],
            },
            "SRI6": {
                "metrics": data_['s6_p0_t2040_2050'],
            },
            "SRI12": {
                "metrics": data_['s12_p0_t2040_2050'],
            },
        },
        "Policy1": {
            "SRI3": {
                "metrics": data_['s3_p1_t2040_2050'],
            },
            "SRI6": {
                "metrics": data_['s6_p1_t2040_2050'],
            },
            "SRI12": {
                "metrics": data_['s12_p1_t2040_2050'],
            },
        },
        "Policy2": {
            "SRI3": {
                "metrics": data_['s3_p2_t2040_2050'],
            },
            "SRI6": {
                "metrics": data_['s6_p2_t2040_2050'],
            },
            "SRI12": {
                "metrics": data_['s12_p2_t2040_2050'],
            },
        },
        "Policy3": {
            "SRI3": {
                "metrics": data_['s3_p3_t2040_2050'],
            },
            "SRI6": {
                "metrics": data_['s6_p3_t2040_2050'],
            },
            "SRI12": {
                "metrics": data_['s12_p3_t2040_2050'],
            },
        },
        "Policy4": {
            "SRI3": {
                "metrics": data_['s3_p4_t2040_2050'],
            },
            "SRI6": {
                "metrics": data_['s6_p4_t2040_2050'],
            },
            "SRI12": {
                "metrics": data_['s12_p4_t2040_2050'],
            },
        },
    },
}

#%% Approach 2: loading data-set to be compatible with seaborn

sri_temps = [3, 12]  # months
policies = [1, 4]  # see definitions above
metric_names = ['Urban Reliability', 'Total Cost']

simulation_periods = ['2020-2030', '2040-2050']
df_small = {}  # preallocate dictionary to hold dataframes
dict_metrics = {}
for m in metric_names:
    metric_append = []
    for i, s in enumerate(sri_temps):
        for j, p in enumerate(policies):
            for t in simulation_periods:
                metric_small = pd.DataFrame(my_dictionary[t][f'Policy{p}'][f'SRI{s}']['metrics'].loc[m])
                metric_small['Period'] = np.repeat(t, n)
                metric_small['SRI'] = np.repeat(s, n)
                metric_small['Policy'] = np.repeat(p, n)
                metric_small['XOrder'] = np.repeat(i * len(sri_temps) + j + 1, n)

                metric_append.append(metric_small)

    dict_metrics.update({m: pd.concat(metric_append, axis=0)})
    if m == 'Total Cost':
        dict_metrics['Total Cost']['Total Cost'] = dict_metrics['Total Cost']['Total Cost']/1E6


#%% Approach 2: plot the data in a violin plot

metric_names = ['Total Deficit', 'Failure Frequency', 'Urban Reliability', 'Total Contracts',
                'Total Shortage Cost', 'Total Contract Cost', 'Total Cost']
metric_names = ['Urban Reliability', 'Total Cost']
metric_ylabel = ['Urban Reliability', 'Total Cost [M$]']

sri_temps = [3, 12]  # months
policies = [1, 4]  # see definitions above
simulation_periods = ['2020-2030', '2040-2050']

# use to set style of background of plot
seaborn.set(style="whitegrid")
colors = ["#FDEEBF7", "#D79DAE"]
colors = [[0.8705882352941177, 0.9215686274509803, 0.9686274509803922],
          [0.8431372549019608, 0.615686274509804, 0.6823529411764706]]
seaborn.set_palette(seaborn.color_palette(colors))

# violin plot by category
plt.figure(figsize=(20, 10))
plt.subplots(len(metric_names), 1)

for i, m in enumerate(metric_names):
    plt.subplot(np.int64(f'{len(metric_names)}1{i+1}'))
    ax = seaborn.violinplot(data=dict_metrics[m], x='XOrder', y=m, hue='Period', cut=0, split=True, gap=0.2, inner='quart')

    plt.ylabel(metric_ylabel[i])
    for j in range(len(policies)):
        plt.axvline(x=j * len(policies) + len(policies) - 0.5, color='darkgray')
    ax.set_xticks(range(len(policies)*len(sri_temps)), labels=np.tile(['Policy' + str(p) for p in policies], len(sri_temps)))
    ax.set_xticks(range(len(policies)*len(sri_temps)), labels=np.tile(['k1: -0.84', 'k1: -1.5'], len(sri_temps)))
    ax.set(xlabel=None)
    if i < len(metric_names) - 1:
        plt.xticks([])
        seaborn.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
        #ax.legend_.remove()
    else:
        ax.legend_.remove()
        for j, s in enumerate(sri_temps):
            plt.text(0.5 + j*len(policies), ax.get_ylim()[0] - 2, f'SRI-{s}', ha='center', fontweight='bold')
plt.suptitle('')
plt.gcf().set_size_inches(8, 4)
plt.savefig("outputs/AGU_2023_results/plots/AGU_InitialSimResults.pdf", format="pdf", bbox_inches="tight")
plt.show()

