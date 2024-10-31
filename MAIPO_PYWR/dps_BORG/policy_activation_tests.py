#%% Imports

# general package imports
import datetime
import pandas as pd
import tables
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pandas.plotting import parallel_coordinates
import os
import json
import warnings
import marshal, ujson as json
#from platypus import Problem, Real
#from pyborg import BorgMOEA

# pywr imports
from pywr.optimisation import *
from pywr.parameters._thresholds import ParameterThresholdParameter, StorageThresholdParameter
from pywr.recorders import DeficitFrequencyNodeRecorder, TotalDeficitNodeRecorder, MeanFlowNodeRecorder, \
    NumpyArrayParameterRecorder, RollingMeanFlowNodeRecorder
from pywr.model import Model
from pywr.recorders import TablesRecorder, Recorder
from pywr.core import Timestepper, Scenario, Node, ConstantParameter
from pywr.core import *
from pywr.parameters import *
from pywr.dataframe_tools import *
from pywr.parameters import IndexParameter, Parameter, load_parameter
from pywr.recorders import Recorder, load_recorder, NodeRecorder, AggregatedRecorder, ParameterRecorder
from pywr.recorders.events import EventRecorder, EventDurationRecorder
from pywr.notebook import *

from MAIPO_PYWR.dps_BORG.example_sim_opt import make_model

os.getcwd()
os.chdir('/Users/danny/Pywr projects/MAIPO_PYWR/dps_BORG')
exec(open("MAIPO_DPS.py").read())
exec(open("example_sim_opt.py").read())


#%% Run models with varying numbers of contracts purchased

# Will make dictionary with outputs of each model. Called outputs. Has ff_PT1, ff_Ag, deficit_PT1, deficit_Ag, cost

num_k = 1  # number of levels in policy tree
num_DP = 7  # number of decision periods
months_in_year = 12
contracts_purchased_list = [0, 30, 60, 90, 120, 150]  # 150 is max agricultural water rights

# thresholds = [-0.84]
threshold = -0.84

models = {}
for policy in ['contracts_only']:
    models[policy] = {}

for contracts_purchased in contracts_purchased_list:
    policies = {
        "contracts_only": {
            "contract_threshold_vals": threshold * np.ones(num_DP),  # threshold at which Chile's policies kick in
            "contract_action_vals": contracts_purchased * np.ones(num_DP)
        }
    }
    for policy, params in policies.items():
        cur_model = make_model(**params)
        cur_model.check()
        cur_model.check_graph()
        print(cur_model.find_orphaned_parameters())
        cur_model.run()

        models[policy][contracts_purchased] = cur_model


#%% Make dictionary of metrics for each model
metrics = {}
for policy, policy_purchases in models.items():
    metrics[policy] = {}
    for contracts_purchased, contracts_purchased_models in policy_purchases.items():
        metrics[policy][contracts_purchased] = {
            "failure_frequency_PT1": np.asarray(contracts_purchased_models.recorders['failure_frequency_PT1'].values()),
            "reliability_PT1": np.asarray(contracts_purchased_models.recorders['reliability_PT1'].values()),
            "failure_frequency_Ag": np.asarray(contracts_purchased_models.recorders['failure_frequency_Ag'].values()),
            "reliability_Ag": np.asarray(contracts_purchased_models.recorders['reliability_Ag'].values()),
            "deficit PT1": np.asarray(contracts_purchased_models.recorders['deficit PT1'].values()),
            "deficit Ag": np.asarray(contracts_purchased_models.recorders['deficit Ag'].values()),
            "Maximum Deficit PT1": np.asarray(contracts_purchased_models.recorders['Maximum Deficit PT1'].values()),
            "Maximum Deficit Ag": np.asarray(contracts_purchased_models.recorders['Maximum Deficit Ag'].values()),
            "TotalCost": np.asarray(contracts_purchased_models.recorders['TotalCost'].values())
        }

#%% Get total urban and ag demand

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
urban_demand = data["PT1"]
num_weeks = len(urban_demand)
total_urban_demand = urban_demand.sum()

agricultural_demand_profile = np.array([18.01638039, 17.94856465, 17.84307351, 17.8053981, 17.7225122, 17.74511745,
                                        17.76772269, 17.77525777, 17.78279286, 17.91088925, 17.97870498, 18.15201186])
total_ag_demand = np.mean(agricultural_demand_profile) * num_weeks  # technically slightly off, but very close

#%% Update deficit_PT1 and deficit_Ag to be ratios and not raw numbers

for policy, policy_purchases in models.items():
    for contracts_purchased, contracts_purchased_model in policy_purchases.items():
        metrics[policy][contracts_purchased]["deficit_proportion_PT1"] = metrics[policy][contracts_purchased]["deficit PT1"] / total_urban_demand
        metrics[policy][contracts_purchased]["deficit_proportion_Ag"] = metrics[policy][contracts_purchased]["deficit Ag"] / total_ag_demand
        # metrics[policy][contracts_purchased]["deficit_Ag"] /= total_ag_demand


#%% Preliminary plots of metrics aggregated over scenarios

func_list = {
    'mean': np.mean,
    'max': np.max,
    'min': np.min
}


def plot_metric(policy, metric):
    recorders = []
    if metric == 'deficit_proportion_PT1' or metric == 'deficit_proportion_Ag':
        agg_func = models[policy][0].recorders['deficit PT1'].agg_func
    else:
        agg_func = models[policy][0].recorders[metric].agg_func
    func = func_list[agg_func]
    for contracts_purchased in contracts_purchased_list:
        recorders.append(func(metrics[policy][contracts_purchased][metric]))
    plt.plot(contracts_purchased_list, recorders, marker='o')
    plt.title('{} for {}'.format(metric, policy))
    plt.ylim(0, 1.2 * np.max(recorders))
    plt.show()


policy_list = ['contracts_only']#, 'demand_only', "contract_and_demand", 'no_policy']
metric_list = ['reliability_PT1', 'reliability_Ag',
               'Maximum Deficit PT1', 'Maximum Deficit Ag',
               'deficit_proportion_PT1', 'deficit_proportion_Ag',
               'TotalCost']

for metric in metric_list:
    for policy in policy_list:
        plot_metric(policy, metric)


#%% Get times when contracts are activated

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\SRI3.csv")
num_weeks = 1560  # number of time steps in timestepper
num_scenarios = 15
scenario_names = data.columns[1:]
is_activated = np.zeros((num_weeks, num_scenarios))
decision_weeks = np.zeros(num_weeks)
is_currently_activated = np.zeros(num_scenarios)
for week in range(num_weeks):
    if week % 26 == 1:
        decision_weeks[week] = 1
        for scenario in range(num_scenarios):
            is_currently_activated[scenario] = (data.loc[week, scenario_names[scenario]] <= threshold)
    is_activated[week, :] = is_currently_activated


#%% Test to see when contracts are activated and how it affects flow

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\SRI3.csv")
num_weeks = 1560  # number of time steps in timestepper
scenario_names = data.columns[1:]

def flow_over_time_plot(output, contracts_purchased, scenario):
    if output == 'PT1':
        plt.plot(
            range(num_weeks),
            models['contracts_only'][0].recorders['Agricultural demand recorder'].data[:, 0]
        )
        # for scenario in range(15):
        for contracts_purchased in contracts_purchased_list:
            plt.plot(
                range(num_weeks),
                models['contracts_only'][contracts_purchased].recorders['PT1 flow'].data[:, scenario]
            )

    elif output == 'Ag':
        plt.plot(
            range(num_weeks),
            models['contracts_only'][0].recorders['Agricultural demand recorder'].data[:, 0],
            label="Demanded agricultural flow"
        )
        # for scenario in range(1):
            # for contracts_purchased in contracts_purchased_list:
        plt.plot(
            range(num_weeks),
            models['contracts_only'][contracts_purchased].recorders['Agriculture flow'].data[:, scenario],
            linewidth=1,
            label="Agricultural flow"
        )

        plt.fill(is_activated[:, 0]*20, alpha=0.2, label='Times contracts have been purchased')
        # plt.plot(decision_weeks)

        plt.ylim([0, 20])
        plt.xlabel("Week")
        plt.ylabel("Flow")

        plt.title("Agricultural flow over time with {} contracts purchased\nScenario: RCP 6.0, 50% precipitation, 50% temperature".format(contracts_purchased))
        plt.legend()
        plt.show()

# print(len(models['contracts_only'][0].recorders['PT1 flow'].data[:, 14]))
# print(len(models['contracts_only'][0].recorders['Agriculture flow'].data[:, 14]))


for contracts_purchased in contracts_purchased_list:
    flow_over_time_plot('Ag', contracts_purchased, 7)

# flow_over_time_plot('Ag', 0, 7)
# plt.fill(is_activated[:, 0]*20, alpha=0.2)
# plt.show()


#%% [NOT RELEVANT TO CURRENT FILE] Plot catchment inflow for each scenario

catchment_inflow = models['contracts_only'][0].recorders['Total reservoir inflow'].data

def moving_avg(input, n=390):
    cumsum = np.cumsum(input, axis=0)
    cumsum[n:, :] = cumsum[n:, :] - cumsum[:-n, :]
    return cumsum[n - 1:, :] / n

catchment_inflow = moving_avg(catchment_inflow)

ax = plt.axes()
ax.set_facecolor('lightgray')

for scenario in range(15):
    plt.plot(catchment_inflow[:, scenario], color='#4F7899')#, linewidth=0.2)

plt.ylabel('Water inflow')
plt.title('Catchment inflow vs Time')

# plt.tick_params(axis='x', which='both', bottom=False,
#                 top=False, labelbottom=False)

plt.xticks(ticks=np.arange(np.ceil(len(catchment_inflow)/(52.178 * 5))) * 52.178 * 5, labels=[2010 + 5 * x for x in range(int(np.ceil(len(catchment_inflow)/(52.178 * 5))))])  # divide by average number of weeks per year
plt.xlabel('Year')

plt.rcParams['axes.linewidth'] = 0

plt.show()


#%% Checking how derechos_sobrantes_contrato changes over time


derechos_sobrantes_contrato = models['contracts_only'][0].recorders['derechos_sobrantes_contrato_recorder'].data

# def moving_avg(input, n=390):
#     cumsum = np.cumsum(input, axis=0)
#     cumsum[n:, :] = cumsum[n:, :] - cumsum[:-n, :]
#     return cumsum[n - 1:, :] / n
#
# catchment_inflow = moving_avg(catchment_inflow)

ax = plt.axes()
ax.set_facecolor('lightgray')

for scenario in range(1):
    plt.plot(derechos_sobrantes_contrato[:, scenario], color='#4F7899')#, linewidth=0.2)

plt.ylabel('derechos_sobrantes_contrato_recorder')
plt.title('derechos_sobrantes_contrato_recorder flow vs Time')

# plt.tick_params(axis='x', which='both', bottom=False,
#                 top=False, labelbottom=False)

# plt.xticks(ticks=np.arange(np.ceil(len(catchment_inflow)/(52.178 * 5))) * 52.178 * 5, labels=[2010 + 5 * x for x in range(int(np.ceil(len(catchment_inflow)/(52.178 * 5))))])  # divide by average number of weeks per year
# plt.xlabel('Year')

plt.rcParams['axes.linewidth'] = 0

plt.show()
