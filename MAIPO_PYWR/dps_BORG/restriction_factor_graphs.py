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
threshold = -0.84
# demand_restriction_factor = 0.9
restriction_factor_list = [1, 0.975, 0.95, 0.925, 0.9, 0.875, 0.85, 0.825, 0.8]

models = {}
for policy in ['contracts_only', 'demand_only', 'contract_and_demand', 'no_policy']:
    models[policy] = {}

for restriction_factor in restriction_factor_list:
    policies = {
        "demand_only": {
            "demand_threshold_vals": [threshold * np.ones(months_in_year)],
            "demand_action_vals": [np.ones(months_in_year), restriction_factor * np.ones(months_in_year)]
        }
    }
    for policy, params in policies.items():

        cur_model = make_model(**params)
        cur_model.check()
        cur_model.check_graph()
        cur_model.find_orphaned_parameters()
        cur_model.run()

        models[policy][restriction_factor] = cur_model


#%% Make dictionary of metrics for each model

metrics = {}
for policy, policy_restrictions in models.items():
    metrics[policy] = {}
    for restriction_factor, model in policy_restrictions.items():
        metrics[policy][restriction_factor] = {
            "failure_frequency_PT1": np.asarray(model.recorders['failure_frequency_PT1'].values()),
            "failure_frequency_Ag": np.asarray(model.recorders['failure_frequency_Ag'].values()),
            "deficit PT1": np.asarray(model.recorders['deficit PT1'].values()),
            "deficit Ag": np.asarray(model.recorders['deficit Ag'].values()),
            "Maximum Deficit PT1": np.asarray(model.recorders['Maximum Deficit PT1'].values()),
            "Maximum Deficit Ag": np.asarray(model.recorders['Maximum Deficit Ag'].values()),
            "TotalCost": np.asarray(model.recorders['TotalCost'].values())
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

for policy, policy_restrictions in models.items():
    for restriction_factor, model in policy_restrictions.items():
        metrics[policy][restriction_factor]["deficit_PT1_proportion"] = metrics[policy][restriction_factor]["deficit PT1"] / total_urban_demand
        metrics[policy][restriction_factor]["deficit_Ag_proportion"] = metrics[policy][restriction_factor]["deficit Ag"] / total_ag_demand


#%% Preliminary plots of metrics

func_list = {
    'mean': np.mean,
    'max': np.max,
    'min': np.min
}


def plot_metric(policy, metric):
    recorders = []
    if metric == 'deficit_PT1_proportion' or metric == 'deficit_Ag_proportion':
        agg_func = models[policy][1].recorders['deficit PT1'].agg_func
    else:
        agg_func = models[policy][1].recorders[metric].agg_func
    func = func_list[agg_func]
    for restriction_factor in restriction_factor_list:
        recorders.append(func(metrics[policy][restriction_factor][metric]))
    plt.plot(restriction_factor_list, recorders, marker='o')
    plt.title('{} for {}'.format(metric, policy))
    plt.ylim(0, 1.2 * np.max(recorders))
    plt.show()


policy_list = ['demand_only']
metric_list = ['failure_frequency_PT1', 'failure_frequency_Ag',
               'Maximum Deficit PT1', 'Maximum Deficit Ag',
               'deficit_PT1_proportion', 'deficit_Ag_proportion']

for metric in metric_list:
    for policy in policy_list:
        plot_metric(policy, metric)


#%% Plot urban-ag failure frequency difference across the thresholds

def restriction_gap_plot(policy, metric_PT1, metric_Ag, metric_title):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(restriction_factor_list), 15))
    Ag_vals = np.zeros((len(restriction_factor_list), 15))
    for idx, contracts_purchased in enumerate(restriction_factor_list):
        PT1_vals[idx, :] = metrics[policy][contracts_purchased][metric_PT1]
        Ag_vals[idx, :] = metrics[policy][contracts_purchased][metric_Ag]


    plt.gca().set_prop_cycle(plt.cycler('color', ['limegreen', 'turquoise', 'deeppink',
                                                  'forestgreen', 'darkslateblue', 'mediumvioletred']))
    ax.plot((PT1_vals - Ag_vals)[:, [0, 2, 4, 10, 12, 14]])
    # ax.plot(Ag_vals.T, '--')
    ax.set_xlabel('Contracts purchased')
    ax.set_ylabel("Failure frequency for PT1 minus Ag")
    ax.set_xticks(np.arange(len(restriction_factor_list)))
    ax.set_xticklabels(restriction_factor_list, rotation=45)
    ax.set_title('{} vs demand restricion factor (with {})'.format(metric_title, policy))
    ax.legend([1, 3, 5, 11, 13, 15])
    # First graph has these ylim values, using same for second for easier comparision

    fig.show()


restriction_gap_plot("demand_only",
                     "deficit_PT1_proportion",
                     "deficit_Ag_proportion",
                     "Deficit proportion gap")


#%% Box/violin plot of metrics for each scenario

def restriction_gap_violin_plot(policy, metric_PT1, metric_Ag, metric_title):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(restriction_factor_list), 15))
    Ag_vals = np.zeros((len(restriction_factor_list), 15))
    for idx, contracts_purchased in enumerate(restriction_factor_list):
        PT1_vals[idx, :] = metrics[policy][contracts_purchased][metric_PT1]
        Ag_vals[idx, :] = metrics[policy][contracts_purchased][metric_Ag]
    violins = ax.violinplot(
        (PT1_vals - Ag_vals).T,
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
        (PT1_vals - Ag_vals).T,
        widths=[0.2] * len(restriction_factor_list),
    )
    for median in boxplots['medians']:
        median.set_color('black')

    # inds = np.arange(1, len(medians) + 1)
    # # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax.set_xlabel('Restriction factor')
    ax.set_ylabel(metric_title)
    ax.set_xticks(np.arange(len(restriction_factor_list)) + 1)
    ax.set_xticklabels(restriction_factor_list, rotation=45)
    ax.set_title('{} vs. restriction factor'.format(metric_title))

    plt.savefig('violins.png', format='png', transparent=True)
    fig.show()


restriction_gap_violin_plot(
    policy='demand_only',
    metric_PT1='deficit_PT1_proportion',
    metric_Ag='deficit_Ag_proportion',
    metric_title="Deficit proportion gap"
)

#%% Box/violin plot of number of days restricted for each scenario
SRI3 = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\SRI3.csv")
SRI3 = SRI3.drop('Timestamp', axis=1)
days_restricted = SRI3 < threshold
num_days_restricted = days_restricted.sum().reset_index()[0].values
num_days = len(SRI3)
proportion_days_restricted = num_days_restricted / num_days

fig, ax = plt.subplots()
violins = ax.violinplot(
    proportion_days_restricted,
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
    proportion_days_restricted,
    # widths=[0.2]
)
for median in boxplots['medians']:
    median.set_color('black')

# inds = np.arange(1, len(medians) + 1)
# # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
# ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
# ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_ylim([0, None])
ax.set_title("Proportion of days restricted")

plt.savefig('violins.png', format='png', transparent=True)
fig.show()

# print(len(SRI3))
# print(len(filter(lambda x: x < threshold), SRI3))


# CAN LOOK AT # DAYS RESTRICTED. CAN CALCULATE JUST BY SEEING HOW OFTEN SRI3 DIPS BELOW -0.84.
# IMPORTANT THING TO NOTE IS THAT WITH SRI3, EVEN IF WATER HAS BEEN MANAGED WELL ENOUGH TO NOT BE IN DROUGHT,
# THE DEMAND RESTRICTION WILL CONTINUE TO BE TRIGGERED -- MAYBE A REASON TO TRY DIFFERENT INDICATORS THAT TAKE
# CURRENT STATE INTO ACCOUNT IN THE FUTURE