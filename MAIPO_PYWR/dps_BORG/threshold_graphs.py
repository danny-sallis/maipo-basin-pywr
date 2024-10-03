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


#%% Run model for scenarios being considered

# Will make dictionary with outputs of each model. Called outputs. Has ff_PT1, ff_Ag, deficit_PT1, deficit_Ag, cost

num_k = 1  # number of levels in policy tree
num_DP = 7  # number of decision periods
months_in_year = 12
contracts_purchased = 30
demand_restriction_factor = 0.9

thresholds = [0, -0.28, -0.56, -0.84, -1.12, -1.40]

models = {}
for policy in ['contracts_only', 'demand_only', 'contract_and_demand', 'no_policy']:
    models[policy] = {}

for threshold in thresholds:
    policies = {
        "contracts_only": {
            "contract_threshold_vals": threshold * np.ones(num_DP),  # threshold at which Chile's policies kick in
            "contract_action_vals": contracts_purchased * np.ones(num_DP)
        },
        "demand_only": {
            "demand_threshold_vals": [threshold * np.ones(months_in_year)],
            "demand_action_vals": [np.ones(months_in_year), demand_restriction_factor * np.ones(months_in_year)]
        },
        "contract_and_demand": {
            "contract_threshold_vals": threshold * np.ones(num_DP),  # threshold at which Chile's policies kick in
            "contract_action_vals": contracts_purchased * np.ones(num_DP),
            "demand_threshold_vals": [threshold * np.ones(months_in_year)],
            "demand_action_vals": [np.ones(months_in_year), demand_restriction_factor * np.ones(months_in_year)]
        },
        "no_policy": {}
    }
    for policy, params in policies.items():

        cur_model = make_model(**params)
        cur_model.check()
        cur_model.check_graph()
        cur_model.find_orphaned_parameters()
        cur_model.run()

        models[policy][threshold] = cur_model


#%% Make dictionary of metrics for each model
metrics = {}
for policy, policy_thresholds in models.items():
    metrics[policy] = {}
    for threshold, model in policy_thresholds.items():
        metrics[policy][threshold] = {
            "ff_PT1": np.asarray(model.recorders['failure_frequency_PT1'].values()),
            "ff_Ag": np.asarray(model.recorders['failure_frequency_Ag'].values()),
            "deficit_PT1": np.asarray(model.recorders['deficit PT1'].values()),
            "deficit_Ag": np.asarray(model.recorders['deficit Ag'].values()),
            "max_deficit_PT1": np.asarray(model.recorders['Maximum Deficit PT1'].values()),
            "max_deficit_Ag": np.asarray(model.recorders['Maximum Deficit Ag'].values()),
            "cost": np.asarray(model.recorders['TotalCost'].values())
        }

#%% Check SRI3 values at the first week of each April/October

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\SRI3.csv")
plt.plot(range(len(data['Timestamp'])), data['RCP_45_Pp_50_Temp_50'])
# plt.plot(range(len(data['Timestamp'])), data[data.columns.difference(['Timestamp'])])

plt.show()


#%% Get total urban and ag demand

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
urban_demand = data["PT1"]
num_weeks = len(urban_demand)
total_urban_demand = urban_demand.sum()

agricultural_demand_profile = np.array([18.01638039, 17.94856465, 17.84307351, 17.8053981, 17.7225122, 17.74511745,
                                        17.76772269, 17.77525777, 17.78279286, 17.91088925, 17.97870498, 18.15201186])
total_ag_demand = np.mean(agricultural_demand_profile) * num_weeks  # technically slightly off, but very close

#%% Preliminary plots of metrics

func_list = {
    'mean': np.mean,
    'max': np.max,
    'min': np.min
}


def plot_metric(policy, metric):
    recorders = []
    agg_func = models[0][policy].recorders[metric].agg_func
    func = func_list[agg_func]
    for threshold in thresholds:
        recorders.append(func(models[threshold][policy].recorders[metric].values()))
    #ff_PT1_recorders = [models[threshold]['contracts_only'].recorders['failure_frequency_PT1'].agg_func for threshold in thresholds]
    # print(thresholds)
    # print(recorders)
    if metric == 'deficit PT1':
        recorders /= total_urban_demand
    if metric == 'deficit Ag':
        recorders /= total_ag_demand
    plt.plot(thresholds, recorders)
    plt.title('{} for {}'.format(metric, policy))
    plt.ylim(0, 1.2 * np.max(recorders))
    plt.show()


policy_list = ['contracts_only', 'demand_only', "contract_and_demand", 'no_policy']
metric_list = ['failure_frequency_PT1', 'failure_frequency_Ag',
               'Maximum Deficit PT1', 'Maximum Deficit Ag',
               'deficit PT1', 'deficit Ag']

for metric in metric_list:
    for policy in policy_list:
        plot_metric(policy, metric)


#%% Plot urban-ag failure frequency difference across the thresholds

def threshold_ff_gap_plot(policy):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(thresholds), 15))
    Ag_vals = np.zeros((len(thresholds), 15))
    for idx, threshold in enumerate(thresholds):
        PT1_vals[idx, :] = metrics[policy][threshold]['ff_PT1']
        Ag_vals[idx, :] = metrics[policy][threshold]['ff_Ag']


    plt.gca().set_prop_cycle(plt.cycler('color', ['limegreen', 'turquoise', 'deeppink',
                                                  'forestgreen', 'darkslateblue', 'mediumvioletred']))
    ax.plot(np.flip((PT1_vals - Ag_vals)[:, [0, 2, 4, 10, 12, 14]], axis=0))
    # ax.plot(Ag_vals.T, '--')
    ax.set_xlabel('Threshold')
    ax.set_ylabel("Failure frequency for PT1 minus Ag")
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels(np.flip(thresholds), rotation=45)
    ax.set_title('Failure frequency vs threshold (with {})'.format(policy))
    ax.legend([1, 3, 5, 11, 13, 15])
    # First graph has these ylim values, using same for second for easier comparision
    ax.set_ylim(-0.02621794871794872, 0.08519230769230765)

    fig.show()


threshold_ff_gap_plot("contracts_only")
threshold_ff_gap_plot("demand_only")


#%% Plot cost with each threshold in contracts_only case

def threshold_ff_gap_plot(policy):
    fig, ax = plt.subplots()
    cost_vals = np.zeros((len(thresholds), 15))
    for idx, threshold in enumerate(thresholds):
        cost_vals[idx, :] = metrics[policy][threshold]['cost']

    plt.gca().set_prop_cycle(plt.cycler('color', ['limegreen', 'turquoise', 'deeppink',
                                                  'forestgreen', 'darkslateblue', 'mediumvioletred']))
    ax.plot(np.flip(cost_vals[:, [0, 2, 4, 10, 12, 14]], axis=0))
    # ax.plot(Ag_vals.T, '--')
    ax.set_xlabel('Threshold')
    ax.set_ylabel("Cost values")
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels(np.flip(thresholds), rotation=45)
    ax.set_title('Cost vs threshold (with {})'.format(policy))
    ax.legend([1, 3, 5, 11, 13, 15])
    # First graph has these ylim values, using same for second for easier comparision
    #ax.set_ylim(-0.02621794871794872, 0.08519230769230765)

    fig.show()

threshold_ff_gap_plot('contracts_only')

#%% Box/violin plot of metrics for each scenario

thresholds = [0, -0.28, -0.56, -0.84, -1.12, -1.40]
def threshold_gap_violin_plot(policy, metric_PT1, metric_Ag, metric_title):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(thresholds), 15))
    Ag_vals = np.zeros((len(thresholds), 15))
    for idx, threshold in enumerate(thresholds):
        PT1_vals[idx, :] = metrics[policy][threshold][metric_PT1]
        Ag_vals[idx, :] = metrics[policy][threshold][metric_Ag]
    violins = ax.violinplot(
        np.flip(PT1_vals - Ag_vals).T,
        showextrema=False
        # showmedians=True,
        # quantiles=[[0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]
    )

    # keys = [*violins]
    # print(violins)
    for violin in violins['bodies']:
        violin.set_color('lightgray')
        violin.set_alpha(1)
        # violin.set_edgecolor('black')
    # violins['cmaxes'].set_color('gray')
    # violins['cmins'].set_color('gray')
    # violins['cbars'].set_color('gray')
    # violins['cmedians'].set_color('white')
    # medians =

    boxplots = ax.boxplot(
        np.flip(PT1_vals - Ag_vals).T,
        widths=[0.2] * 6,
    )
    for median in boxplots['medians']:
        median.set_color('black')

    # inds = np.arange(1, len(medians) + 1)
    # # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax.set_xlabel('Threshold')
    ax.set_ylabel("{} values".format(metric_title))
    ax.set_xticks(np.arange(len(thresholds)) + 1)
    ax.set_xticklabels(np.flip(thresholds), rotation=45)
    ax.set_title('{} vs threshold (with {})'.format(metric_title, policy))

    plt.savefig('violins.png', format='png', transparent=True)
    fig.show()


threshold_gap_violin_plot(
    policy='demand_only',
    metric_PT1='ff_PT1',
    metric_Ag='ff_Ag',
    metric_title="Failure frequency gap"
)
