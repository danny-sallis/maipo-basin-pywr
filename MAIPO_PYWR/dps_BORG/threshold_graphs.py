#%% Imports

# general package imports
import datetime
import pandas as pd
import tables
import pandas
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
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
import kaleido

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
contracts_purchased = 500
demand_restriction_factor = 0.9

# Usually activated to never activated
thresholds = [0, -0.5, -1, -1.5, -2, -999]

models = {}
for policy in ['contracts_only', 'demand_only']:#, 'contract_and_demand', 'no_policy']:
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
        # "contract_and_demand": {
        #     "contract_threshold_vals": threshold * np.ones(num_DP),  # threshold at which Chile's policies kick in
        #     "contract_action_vals": contracts_purchased * np.ones(num_DP),
        #     "demand_threshold_vals": [threshold * np.ones(months_in_year)],
        #     "demand_action_vals": [np.ones(months_in_year), demand_restriction_factor * np.ones(months_in_year)]
        # },
        # "no_policy": {}
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
            "reliability_PT1": np.asarray(model.recorders['reliability_PT1'].values()),
            "reliability_Ag": np.asarray(model.recorders['reliability_Ag'].values()),
            "deficit_PT1": np.asarray(model.recorders['deficit PT1'].values()),
            "deficit_Ag": np.asarray(model.recorders['deficit Ag'].values()),
            "max_deficit_PT1": np.asarray(model.recorders['Maximum Deficit PT1'].values()),
            "max_deficit_Ag": np.asarray(model.recorders['Maximum Deficit Ag'].values()),
            "cost": np.asarray(model.recorders['TotalCost'].values())
        }

#%% Check SRI3 values at the first week of each April/October [NOT USED IN FINAL PLOTS]

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\SRI3.csv")
plt.plot(range(len(data['Timestamp'])), data['RCP_45_Pp_50_Temp_50'])
# plt.plot(range(len(data['Timestamp'])), data[data.columns.difference(['Timestamp'])])

plt.show()


#%% Update deficit_PT1 and deficit_Ag to be ratios and not raw numbers

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
urban_demand = data["PT1"]
num_weeks = len(urban_demand)
total_urban_demand = urban_demand.sum()

agricultural_demand_profile = np.array([18.01638039, 17.94856465, 17.84307351, 17.8053981, 17.7225122, 17.74511745,
                                        17.76772269, 17.77525777, 17.78279286, 17.91088925, 17.97870498, 18.15201186])
total_ag_demand = np.mean(agricultural_demand_profile) * num_weeks  # technically slightly off, but very close

for policy, policy_thresholds in models.items():
    for threshold, model in policy_thresholds.items():
        metrics[policy][threshold]["deficit_PT1_proportion"] = metrics[policy][threshold]["deficit_PT1"] / total_urban_demand
        metrics[policy][threshold]["deficit_Ag_proportion"] = metrics[policy][threshold]["deficit_Ag"] / total_ag_demand


#%% Preliminary plots of metrics [NOT USED IN FINAL PLOTS]

func_list = {
    'mean': np.mean,
    'max': np.max,
    'min': np.min
}


def plot_metric(policy, metric):
    recorders = []
    agg_func = models[policy][0].recorders[metric].agg_func
    func = func_list[agg_func]
    for threshold in thresholds:
        recorders.append(func(models[policy][threshold].recorders[metric].values()))
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


policy_list = ['contracts_only', 'demand_only']#, "contract_and_demand", 'no_policy']
metric_list = ['failure_frequency_PT1', 'failure_frequency_Ag',
               'Maximum Deficit PT1', 'Maximum Deficit Ag',
               'deficit PT1', 'deficit Ag']

for metric in metric_list:
    for policy in policy_list:
        plot_metric(policy, metric)


#%% Plot urban-ag failure frequency difference across the thresholds [NOT USED IN FINAL PLOTS]

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


#%% Plot cost with each threshold in contracts_only case [NOT USED IN FINAL PLOTS]

def threshold_ff_gap_plot(policy):
    fig, ax = plt.subplots()
    cost_vals = np.zeros((len(thresholds), 15))
    for idx, threshold in enumerate(thresholds):
        cost_vals[idx, :] = metrics[policy][threshold]['ff_PT1']# - metrics[policy][threshold]['ff_Ag']

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

#%% Box/violin plot of metrics for each scenario [NOT USED IN FINAL PLOTS]

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
        positions=range(len(thresholds)),
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
        positions=np.arange(len(thresholds)),
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
    ax.set_xticks(np.arange(len(thresholds)))
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


#%% Boxplots for ag and PT1 separately with connecting lines [NOT USED IN FINAL PLOTS]

def box_plots_with_connectors(policy, metric_PT1, metric_Ag, xlabel, ylabel, graph_title):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(thresholds), 15))
    Ag_vals = np.zeros((len(thresholds), 15))
    for idx, threshold in enumerate(thresholds):
        PT1_vals[idx, :] = metrics[policy][threshold][metric_PT1]
        Ag_vals[idx, :] = metrics[policy][threshold][metric_Ag]

    width = 0.3
    offset = 0.2

    violins_PT1 = ax.violinplot(
        PT1_vals.T,
        positions=np.arange(len(thresholds)) - offset,
        # showextrema=False,
        widths=[width] * len(thresholds)
        # showmedians=True,
        # quantiles=[[0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]
    )

    # keys = [*violins]
    # print(violins)
    for violin in violins_PT1['bodies']:
        violin.set_color('lightgray')
        violin.set_edgecolor('black')
        violin.set_linewidth(0.5)
        violin.set_alpha(1)
        # violin.set_edgecolor('black')
    violins_PT1['cmaxes'].set_color('gray')
    violins_PT1['cmins'].set_color('gray')
    violins_PT1['cbars'].set_color('gray')
    violins_PT1['cbars'].set_linewidth(1.5)
    # violins['cmedians'].set_color('white')
    # medians =

    violins_Ag = ax.violinplot(
        Ag_vals.T,
        positions=np.arange(len(thresholds)) + offset,
        showextrema=False,
        widths=[width] * len(thresholds)
        # showmedians=True,
        # quantiles=[[0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75], [0.25, 0.75]]
    )

    # keys = [*violins]
    # print(violins)
    for violin in violins_Ag['bodies']:
        violin.set_color('lightgray')
        violin.set_edgecolor('black')
        violin.set_linewidth(0.5)
        violin.set_alpha(1)
        violin.set_edgecolor('black')
    # violins['cmaxes'].set_color('gray')
    # violins['cmins'].set_color('gray')
    # violins['cbars'].set_color('gray')
    # violins['cmedians'].set_color('white')
    # medians =

    boxplots_PT1 = ax.boxplot(
        np.flip(PT1_vals).T,
        positions=np.arange(len(thresholds)) - offset,
        widths=[width] * len(thresholds),
        label="Urban"
    )
    PT1_color = 'steelblue'
    for whisker in boxplots_PT1['whiskers']:
        whisker.set_color(PT1_color)
    for cap in boxplots_PT1['caps']:
        cap.set_color(PT1_color)
    for box in boxplots_PT1['boxes']:
        box.set_color(PT1_color)
    for median in boxplots_PT1['medians']:
        median.set_color(PT1_color)
    for flier in boxplots_PT1['fliers']:
        flier.set_color(PT1_color)


    boxplots_Ag = ax.boxplot(
        np.flip(Ag_vals).T,
        positions=np.arange(len(thresholds)) + offset,
        widths=[width] * len(thresholds),
        label="Agricultural"
    )
    Ag_color = 'green'
    for whisker in boxplots_Ag['whiskers']:
        whisker.set_color(Ag_color)
    for cap in boxplots_Ag['caps']:
        cap.set_color(Ag_color)
    for box in boxplots_Ag['boxes']:
        box.set_color(Ag_color)
    for median in boxplots_Ag['medians']:
        median.set_color(Ag_color)
    for flier in boxplots_Ag['fliers']:
        flier.set_color(Ag_color)


    # For each threshold, add line between PT1 and Ag vals
    num_scenarios = len(metrics[policy][thresholds[0]][metric_PT1])
    for threshold_idx, threshold in enumerate(thresholds):
        for scenario in range(num_scenarios):
            plt.plot([threshold_idx - offset, threshold_idx + offset],
                     [np.flip(PT1_vals)[threshold_idx, scenario], np.flip(Ag_vals)[threshold_idx, scenario]],
                     '--', color='black', marker='.', markerfacecolor='black',
                     markeredgewidth=0, linewidth=0.5)
            # FIX THIS!!!


    # inds = np.arange(1, len(medians) + 1)
    # # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_ylabel("{} values".format(metric_title))
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels(np.flip(thresholds), rotation=45)
    ax.set_title(graph_title)
    legend = ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    fig.savefig('boxplots.png', format='png', transparent=False, bbox_extra_artists=[legend], bbox_inches='tight')

    # plt.savefig('boxplots.png', format='png', transparent=True)
    fig.show()


box_plots_with_connectors(
    policy='demand_only',
    metric_PT1='ff_PT1',
    metric_Ag='ff_Ag',
    xlabel="SRI3 Indicator Threshold",
    ylabel="Failure frequency for urban and agricultural consumers",
    graph_title="Failure frequencies vs thresholds for demand restriction"
)

#%% Boxplot for any metric

def box_plot(policy, metric, xlabel, ylabel, graph_title, ylim=None):
    fig, ax = plt.subplots()
    metric_vals = np.zeros((len(thresholds), 15))
    for idx, threshold in enumerate(thresholds):
        metric_vals[idx, :] = metrics[policy][threshold][metric]

    # width = 0.2

    boxplots = ax.boxplot(
        np.flip(metric_vals).T,
        positions=np.arange(len(thresholds)),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels(["no policy", *np.flip(thresholds[:-1])], rotation=45)
    ax.set_ylim(ylim)
    ax.set_title(graph_title)
    legend = ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    fig.show()


box_plot(
    policy='contracts_only',
    metric='cost',
    xlabel="Contracts purchased",
    ylabel="Cost of purchasing contracts",
    graph_title="Total cost vs contracts purchased",
    # ylim=[0, 1]
)


#%% Boxplots for ag and PT1 separately (without connecting lines)

def box_plots_without_connectors(policy, metric_PT1, metric_Ag, xlabel, ylabel, graph_title, ylim=None):
    fig, ax = plt.subplots(figsize=(12, 6))
    PT1_vals = np.zeros((len(thresholds), 15))
    Ag_vals = np.zeros((len(thresholds), 15))
    for idx, threshold in enumerate(thresholds):
        if idx == len(thresholds):
            break
        PT1_vals[idx, :] = metrics[policy][threshold][metric_PT1]
        Ag_vals[idx, :] = metrics[policy][threshold][metric_Ag]

    width = 0.2
    offset = 0.15

    boxplots_PT1 = ax.boxplot(
        100*np.flip(PT1_vals).T,
        positions=np.arange(len(thresholds)) - offset,
        widths=[width] * (len(thresholds)),
        patch_artist=True,
        label="Urban"
    )
    PT1_color = 'steelblue'
    for whisker in boxplots_PT1['whiskers']:
        whisker.set_color(PT1_color)
    for cap in boxplots_PT1['caps']:
        cap.set_color(PT1_color)
    for box in boxplots_PT1['boxes']:
        box.set_color(PT1_color)
    for median in boxplots_PT1['medians']:
        median.set_color('white')
        # median.set_linewidth(1.5)
    for flier in boxplots_PT1['fliers']:
        flier.set_color(PT1_color)


    boxplots_Ag = ax.boxplot(
        100*np.flip(Ag_vals).T,
        positions=np.arange(len(thresholds)) + offset,
        widths=[width] * len(thresholds),
        patch_artist=True,
        label="Agricultural"
    )
    Ag_color = 'green'
    for whisker in boxplots_Ag['whiskers']:
        whisker.set_color(Ag_color)
    for cap in boxplots_Ag['caps']:
        cap.set_color(Ag_color)
    for box in boxplots_Ag['boxes']:
        box.set_color(Ag_color)
    for median in boxplots_Ag['medians']:
        median.set_color('white')
        # median.set_linewidth(1.5)
    for flier in boxplots_Ag['fliers']:
        flier.set_color(Ag_color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels(["Baseline", *np.flip(thresholds[:-1])], rotation=45)
    ax.set_ylim(ylim)
    ax.set_title(graph_title)
    legend = ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')

    ax.set_facecolor("#ebf5f6")
    fig.patch.set_facecolor("#cde5e9")

    plt.subplots_adjust(bottom=0.2)

    # fig.savefig('boxplots.png', format='png', transparent=False, bbox_extra_artists=[legend], bbox_inches='tight')

    plt.savefig('{}.png'.format(graph_title), format='png', transparent=False)
    fig.show()


box_plots_without_connectors(
    policy='contracts_only',
    metric_PT1='reliability_PT1',
    metric_Ag='reliability_Ag',
    xlabel="Threshold for purchasing contracts",
    ylabel="Water supply reliability (%)",
    graph_title="Reliability vs. Drought Policy Threshold with Water Option Contracts",
    ylim=[70, 100]
)

# box_plots_without_connectors(
#     policy='contracts_only',
#     metric_PT1='max_deficit_PT1',
#     metric_Ag='max_deficit_Ag',
#     xlabel="Threshold for purchasing contracts",
#     ylabel="Maximum deficit for urban and agricultural consumers",
#     graph_title="Maximum deficit vs threshold for purchasing contracts",
#     ylim=[0, 7.5]
# )
#
# box_plots_without_connectors(
#     policy='contracts_only',
#     metric_PT1='deficit_PT1_proportion',
#     metric_Ag='deficit_Ag_proportion',
#     xlabel="Threshold for purchasing contracts",
#     ylabel="Proportion of demand unmet for urban and agricultural consumers",
#     graph_title="Proportion of demand unmet vs threshold for purchasing contracts",
#     ylim=[0, 0.07]
# )


box_plots_without_connectors(
    policy='demand_only',
    metric_PT1='reliability_PT1',
    metric_Ag='reliability_Ag',
    xlabel="Threshold for demand restriction",
    ylabel="Water supply reliability (%)",
    graph_title="Reliability vs. Drought Policy Threshold with Demand Restriction",
    ylim=[70, 100]
)

# box_plots_without_connectors(
#     policy='demand_only',
#     metric_PT1='max_deficit_PT1',
#     metric_Ag='max_deficit_Ag',
#     xlabel="Threshold for demand restriction",
#     ylabel="Maximum deficit for urban and agricultural consumers",
#     graph_title="Maximum deficit vs threshold for demand restriction",
#     ylim=[0, 7.5]
# )
#
# box_plots_without_connectors(
#     policy='demand_only',
#     metric_PT1='deficit_PT1_proportion',
#     metric_Ag='deficit_Ag_proportion',
#     xlabel="Threshold for demand restriction",
#     ylabel="Proportion of demand unmet for urban and agricultural consumers",
#     graph_title="Proportion of demand unmet vs threshold for demand restriction",
#     ylim=[0, 0.08]
# )


#%% Parallel coordinates plots with metrics at most and least extreme values

num_scenarios = 15
scenarios = pd.read_csv('../data/COLORADO.csv').columns[1::]


def parse_scenario_title(scenario_name):
    tokens = scenario_name.split('_')
    rcp_val = int(tokens[1])
    pp = int(tokens[3])
    temp = int(tokens[5])

    return rcp_val, pp, temp


metric_list = ['reliability_PT1', 'reliability_Ag', 'deficit_PT1_proportion', 'deficit_Ag_proportion',
           'max_deficit_PT1', 'max_deficit_Ag', 'cost']

par_coord_data = {}
extreme_thresholds = [-0.5, -999]
for policy in ["contracts_only", "demand_only"]:
    par_coord_policy_data = []
    for scenario_idx in range(len(scenarios)):
        new_row = [scenario_idx]
        new_row += [*parse_scenario_title(scenarios[scenario_idx])]
        for metric in metric_list:
            for threshold in extreme_thresholds:
                new_row.append(metrics[policy][threshold][metric][scenario_idx])
        par_coord_policy_data.append(new_row)
    par_coord_data[policy] = par_coord_policy_data

metric_threshold_pairs = []
for metric in metric_list:
    for threshold in extreme_thresholds:
        metric_threshold_pairs.append("{}, {}".format(metric, threshold))

par_coord_df = {}
for policy in par_coord_data.keys():
    par_coord_df[policy] = pd.DataFrame(
        par_coord_data[policy],
        columns=[
                    'scenario index',
                    'rcp val',
                    'pp',
                    'temp'
                ] + metric_threshold_pairs
    )
# print(par_coord_df.head())


# def extreme_par_coords(policy, title, metrics, labels, ranges=None, colorscale='scenario index'):
#     fig = go.Figure(
#         data=go.Parcoords(
#             line=dict(color=par_coord_df[policy][colorscale],
#                       showscale=True),  # COLORSCALE CAN BE ADDED OPTIONALLY
#             dimensions=list([
#                 dict(
#                     range=ranges[i],
#                     label=labels[i],
#                     values=par_coord_df[policy][metrics[i]],
#                 ) for i in range(len(metrics))
#             ])
#         )
#     )
#
#     fig.update_layout(
#         title=title
#     )
#
#     # fig.write_image("Extreme par coords: {}.png".format(title))
#     fig.show(renderer="browser")


def extreme_par_coords_matplotlib(policy, title, metrics, labels, ylim=None):
    metrics_by_scenario = par_coord_df[policy][metrics].T
    cmap = plt.get_cmap('viridis', 5)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for i in range(num_scenarios):
        ax.plot(labels, 100*metrics_by_scenario[i], c=cmap(i % 5))

    plt.title(title)
    plt.ylim(ylim)

    sm = plt.cm.ScalarMappable(cmap=cmap)#, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(
        sm,
        ticks=[0.1, 0.3, 0.5, 0.7, 0.9],
        cax=ax.inset_axes([1.02, 0.1, 0.05, 0.8]),
    )
    cbar.set_label('Precipitation percentile change')
    cbar.ax.set_yticklabels(["5%", "25%", "50%", "75%", "95%"])

    ax.set_facecolor("#ebf5f6")
    fig.patch.set_facecolor("#cde5e9")

    plt.subplots_adjust(right=0.85)

    plt.savefig('{} ({}).png'.format(title, policy), format='png', transparent=False)
    plt.show()


extreme_par_coords_matplotlib(
    "contracts_only",
    "Reliability for Purchasing Contracts with Threshold -0.5",
    [
        "reliability_PT1, -0.5",
        "reliability_Ag, -0.5"
    ],
    [
        "Urban reliability (%)",
        "Agricultural reliability (%)",
    ],
    ylim=[70, 100],
)

extreme_par_coords_matplotlib(
    "contracts_only",
    "Baseline Reliability",
    [
        "reliability_PT1, -999",
        "reliability_Ag, -999"
    ],
    [
        "Urban reliability (%)",
        "Agricultural reliability (%)",
    ],
    ylim=[70, 100]
)

# extreme_par_coords_matplotlib(
#     "contracts_only",
#     "Max deficit for purchasing contracts with threshold 0",
#     [
#         "max_deficit_PT1, 0",
#         "max_deficit_Ag, 0"
#     ],
#     [
#         "Urban max deficit (threshold 0)",
#         "Agricultural max deficit (threshold 0)",
#     ],
#     ylim=[0, 8]
# )
#
# extreme_par_coords_matplotlib(
#     "contracts_only",
#     "Max deficit with no policy",
#     [
#         "max_deficit_PT1, -999",
#         "max_deficit_Ag, -999"
#     ],
#     [
#         "Urban max deficit (-999 contracts purchased)",
#         "Agricultural max deficit (-999 contracts purchased)",
#     ],
#     ylim=[0, 8]
# )
#
# extreme_par_coords_matplotlib(
#     "contracts_only",
#     "Unmet demand proportion for purchasing contracts with threshold 0",
#     [
#         "deficit_PT1_proportion, 0",
#         "deficit_Ag_proportion, 0"
#     ],
#     [
#         "Urban unmet demand proportion (threshold 0)",
#         "Agricultural unmet demand proportion (threshold 0)",
#     ],
#     ylim=[0, 0.1]
# )
#
# extreme_par_coords_matplotlib(
#     "contracts_only",
#     "Unmet demand proportion with no policy",
#     [
#         "deficit_PT1_proportion, -999",
#         "deficit_Ag_proportion, -999"
#     ],
#     [
#         "Urban unmet demand proportion (no policy)",
#         "Agricultural unmet demand proportion (no policy)",
#     ],
#     ylim=[0, 0.07]
# )


extreme_par_coords_matplotlib(
    "demand_only",
    "Reliability for Demand Restriction with Threshold -0.5",
    [
        "reliability_PT1, -0.5",
        "reliability_Ag, -0.5"
    ],
    [
        "Urban reliability (%)",
        "Agricultural reliability (%)",
    ],
    ylim=[70, 100],
)

extreme_par_coords_matplotlib(
    "demand_only",
    "Baseline Reliability",
    [
        "reliability_PT1, -999",
        "reliability_Ag, -999"
    ],
    [
        "Urban reliability (%)",
        "Agricultural reliability (%)",
    ],
    ylim=[70, 100]
)

# extreme_par_coords_matplotlib(
#     "demand_only",
#     "Max deficit for demand restriction with threshold 0",
#     [
#         "max_deficit_PT1, 0",
#         "max_deficit_Ag, 0"
#     ],
#     [
#         "Urban max deficit (threshold 0)",
#         "Agricultural max deficit (threshold 0)",
#     ],
#     ylim=[0, 8]
# )
#
# extreme_par_coords_matplotlib(
#     "demand_only",
#     "Max deficit with no policy",
#     [
#         "max_deficit_PT1, -999",
#         "max_deficit_Ag, -999"
#     ],
#     [
#         "Urban max deficit (-999 contracts purchased)",
#         "Agricultural max deficit (-999 contracts purchased)",
#     ],
#     ylim=[0, 8]
# )
#
# extreme_par_coords_matplotlib(
#     "demand_only",
#     "Unmet demand proportion for demand restriction with threshold 0",
#     [
#         "deficit_PT1_proportion, 0",
#         "deficit_Ag_proportion, 0"
#     ],
#     [
#         "Urban unmet demand proportion (threshold 0)",
#         "Agricultural unmet demand proportion (threshold 0)",
#     ],
#     ylim=[0, 0.1]
# )
#
# extreme_par_coords_matplotlib(
#     "demand_only",
#     "Unmet demand proportion with no policy",
#     [
#         "deficit_PT1_proportion, -999",
#         "deficit_Ag_proportion, -999"
#     ],
#     [
#         "Urban unmet demand proportion (no policy)",
#         "Agricultural unmet demand proportion (no policy)",
#     ],
#     ylim=[0, 0.07]
# )


#%% Parallel coordinates plots with metrics at most and least extreme values [NOT USED IN FINAL PLOTS]

# num_scenarios = 15
# scenarios = pd.read_csv('../data/COLORADO.csv').columns[1::]
#
#
# def parse_scenario_title(scenario_name):
#     tokens = scenario_name.split('_')
#     rcp_val = int(tokens[1])
#     pp = int(tokens[3])
#     temp = int(tokens[5])
#
#     return rcp_val, pp, temp
#
#
# metric_list = ['reliability_PT1', 'reliability_Ag', 'deficit_PT1_proportion', 'deficit_Ag_proportion',
#            'max_deficit_PT1', 'max_deficit_Ag', 'cost']
#
# par_cord_data = []
# extreme_thresholds = [-999, 0]
# policy = "contracts_only"
# for scenario_idx in range(len(scenarios)):
#     new_row = [scenario_idx]
#     new_row += [*parse_scenario_title(scenarios[scenario_idx])]
#     for metric in metric_list:
#         for contracts_purchased in extreme_thresholds:
#             new_row.append(metrics[policy][contracts_purchased][metric][scenario_idx])
#     par_cord_data.append(new_row)
#
# metric_contract_pairs = []
# for metric in metric_list:
#     for contracts_purchased in extreme_thresholds:
#         metric_contract_pairs.append("{}, {}".format(metric, contracts_purchased))
#
# par_cord_df = pd.DataFrame(
#     par_cord_data,
#     columns=[
#                 'scenario index',
#                 'rcp val',
#                 'pp',
#                 'temp'
#             ] + metric_contract_pairs
# )
# print(par_cord_df.head())
#
#
# def extreme_par_coords(title, metrics, labels, ranges=None, colorscale='scenario index'):
#     fig = go.Figure(
#         data=go.Parcoords(
#             line=dict(color=par_cord_df[colorscale],
#                       showscale=True),  # COLORSCALE CAN BE ADDED OPTIONALLY
#             dimensions=list([
#                 dict(
#                     range=ranges[i],
#                     label=labels[i],
#                     values=par_cord_df[metrics[i]],
#                 ) for i in range(len(metrics))
#             ])
#         )
#     )
#
#     fig.update_layout(
#         title=title
#     )
#
#     fig.show(renderer="browser")
#
#
# extreme_par_coords(
#     "Reliability for no purchcases and threshold 0, colored by precipitation change",
#     [
#         'reliability_PT1, -999',
#         'reliability_PT1, 0',
#         'reliability_Ag, -999',
#         'reliability_Ag, 0',
#     ],
#     [
#         "Urban reliability (no purchases)",
#         "Urban reliability (threshold 0)",
#         "Agricultural reliability (no purchases)",
#         "Agricultural reliability (threshold 0)",
#     ],
#     ranges=[
#         [0.7, 1],
#         [0.7, 1],
#         [0.7, 1],
#         [0.7, 1]
#     ],
#     colorscale='pp'
# )
#
# extreme_par_coords(
#     "Maximum deficit for no purchases and threshold 0, colored by precipitation change",
#     [
#         'max_deficit_PT1, -999',
#         'max_deficit_PT1, 0',
#         'max_deficit_Ag, -999',
#         'max_deficit_Ag, 0',
#     ],
#     [
#         "Urban max deficit (no purchases)",
#         "Urban max deficit (threshold 0)",
#         "Agricultural max deficit (no purchases)",
#         "Agricultural max deficit (threshold 0)",
#     ],
#     ranges=[
#         [0, 8],
#         [0, 8],
#         [0, 8],
#         [0, 8]
#     ],
#     colorscale='pp'
# )
#
# extreme_par_coords(
#     "Unmet demand proportion for no purchases and threshold 0, colored by precipitation change",
#     [
#         'deficit_PT1_proportion, -999',
#         'deficit_PT1_proportion, 0',
#         'deficit_Ag_proportion, -999',
#         'deficit_Ag_proportion, 0',
#     ],
#     [
#         "Urban unmet demand proportion (no purchases)",
#         "Urban unmet demand proportion (threshold 0)",
#         "Agricultural unmet demand proportion (no purchases)",
#         "Agricultural unmet demand proportion (threshold 0)",
#     ],
#     ranges=[
#         [0, 0.08],
#         [0, 0.08],
#         [0, 0.08],
#         [0, 0.08]
#     ],
#     colorscale='pp'
# )


#%% Parallel coordinate plots to show order of metrics for scenarios in urban/ag [NOT USED IN FINAL PLOTS]

def parallel_coordinates_thresholds(policy, metric_PT1, metric_Ag, xlabel, ylabel, graph_title, thresholds=thresholds):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(thresholds), 15))
    Ag_vals = np.zeros((len(thresholds), 15))
    for idx, threshold in enumerate(thresholds):
        PT1_vals[idx, :] = metrics[policy][threshold][metric_PT1]
        Ag_vals[idx, :] = metrics[policy][threshold][metric_Ag]

    color_list = ['#9b6871', '#383c3d', '#c014e2', '#1a4554', '#cc954d',
                  '#d60e4e', '#020201', '#262825', '#b21358', '#59ba73',
                  '#2f3030', '#707a87', '#4b5b5b', '#f26546', '#60423f']
    # For each threshold, add line between PT1 and Ag vals
    num_scenarios = len(metrics[policy][thresholds[0]][metric_PT1])
    for threshold_idx, threshold in enumerate(thresholds):
        for scenario_idx in range(num_scenarios):
            plt.plot([0, 1],
                     [np.flip(PT1_vals)[threshold_idx, scenario_idx], np.flip(Ag_vals)[threshold_idx, scenario_idx]],
                     '-', color=color_list[threshold_idx], marker='.', markerfacecolor='black',
                     markeredgewidth=0, linewidth=0.5)
            # FIX THIS!!!

    # inds = np.arange(1, len(medians) + 1)
    # # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax.set_ylabel(ylabel)
    # ax.set_ylabel("{} values".format(metric_title))
    ax.set_xticklabels(np.flip(thresholds), rotation=45)
    ax.set_title(graph_title)
    legend = ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    # fig.savefig('boxplots.png', format='png', transparent=False, bbox_extra_artists=[legend], bbox_inches='tight')

    # plt.savefig('boxplots.png', format='png', transparent=True)
    fig.show()

parallel_coordinates_thresholds(
    policy='demand_only',
    metric_PT1='ff_PT1',
    metric_Ag='ff_Ag',
    xlabel="SRI3 Indicator Threshold",
    ylabel="Failure frequency for urban and agricultural consumers",
    graph_title="Failure frequencies vs thresholds for demand restriction",
    thresholds=[-0.28, -1.4]
)


#%% Urban order vs ag order [NOT USED IN FINAL PLOTS]

def urban_ag_line(policy, metric_PT1, metric_Ag, xlabel, ylabel, graph_title):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(thresholds), 15))
    Ag_vals = np.zeros((len(thresholds), 15))

    for threshold_idx, threshold in enumerate(thresholds):
        PT1_vals[threshold_idx, :] = metrics[policy][threshold][metric_PT1]
        Ag_vals[threshold_idx, :] = metrics[policy][threshold][metric_Ag]

    color_list = ['#9b6871', '#383c3d', '#c014e2', '#1a4554', '#cc954d',
                  '#d60e4e', '#020201', '#262825', '#b21358', '#59ba73',
                  '#2f3030', '#707a87', '#4b5b5b', '#f26546', '#60423f']
    # For each threshold, add line between PT1 and Ag vals
    num_scenarios = len(metrics[policy][thresholds[0]][metric_PT1])
    for threshold_idx, threshold in enumerate(thresholds):
        sorted_PT1 = np.sort(PT1_vals[threshold_idx])
        sorted_Ag = np.sort(Ag_vals[threshold_idx])
        order_points = np.zeros((num_scenarios, 2))
        PT1_ordering = PT1_vals[threshold_idx, :].argsort()
        # order_points[0, :] = PT1_ordering
        # order_points[1, :] = Ag_vals[PT1_ordering]
        plt.plot(range(15), Ag_vals[threshold_idx, PT1_ordering].argsort())
        # PT1_zero_count = 0
        # Ag_zero_count = 0
        # for scenario_idx in range(num_scenarios):
        #     sorted_PT1_idx = np.where(sorted_PT1 == PT1_vals[threshold_idx][scenario_idx])[0][0]
        #     sorted_Ag_idx = np.where(sorted_Ag == Ag_vals[threshold_idx][scenario_idx])[0][0]
        #     # print(sorted_PT1_idx, sorted_Ag_idx)
        #     if np.isclose(PT1_vals[threshold_idx, scenario_idx], 0):
        #         order_points[scenario_idx, 0] = PT1_zero_count
        #         PT1_zero_count += 1
        #     else:
        #         order_points[scenario_idx, 0] = sorted_PT1_idx
        #
        #     if np.isclose(Ag_vals[threshold_idx, scenario_idx], 0, atol=1e-2):
        #         order_points[scenario_idx, 1] = Ag_zero_count
        #         Ag_zero_count += 1
        #     else:
        #         order_points[scenario_idx, 1] = sorted_Ag_idx
        # order = order_points[:, 0].argsort()
        # order_points = order_points[order, :]
        # print(order_points.T)
        # print(sorted_Ag)
        # plt.plot(range(15), order_points[:, 1], color=color_list[threshold_idx])

            # plt.plot([0, 1],
            #          [np.flip(PT1_vals)[threshold_idx, scenario_idx], np.flip(Ag_vals)[threshold_idx, scenario_idx]],
            #          '-', color=color_list[threshold_idx], marker='.', markerfacecolor='black',
            #          markeredgewidth=0, linewidth=0.5)
            # FIX THIS!!!

    # inds = np.arange(1, len(medians) + 1)
    # # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_ylabel("{} values".format(metric_title))
    ax.set_title(graph_title)
    # legend = ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    # fig.savefig('boxplots.png', format='png', transparent=False, bbox_extra_artists=[legend], bbox_inches='tight')

    # plt.savefig('boxplots.png', format='png', transparent=True)
    fig.show()

urban_ag_line(
    policy='demand_only',
    metric_PT1='ff_PT1',
    metric_Ag='ff_Ag',
    xlabel="Urban order",
    ylabel="Agricultural order",
    graph_title="Relative orders of failure frequencies for urban and agricultural users"
)


#%% urban-ag changes for high and low thresholds [NOT USED IN FINAL PLOTS]

def urban_ag_side_by_side(policy, metric_PT1, metric_Ag, xlabel, ylabel, graph_title, thresholds=thresholds):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(thresholds), 15))
    Ag_vals = np.zeros((len(thresholds), 15))
    for idx, threshold in enumerate(thresholds):
        PT1_vals[idx, :] = metrics[policy][threshold][metric_PT1]
        Ag_vals[idx, :] = metrics[policy][threshold][metric_Ag]

    width = 0.3
    offset = 0.2

    boxplots_PT1 = ax.boxplot(
        np.flip(PT1_vals).T,
        positions=np.arange(len(thresholds)) - offset,
        widths=[width] * len(thresholds),
        label="Urban"
    )
    PT1_color = 'steelblue'
    for whisker in boxplots_PT1['whiskers']:
        whisker.set_color(PT1_color)
    for cap in boxplots_PT1['caps']:
        cap.set_color(PT1_color)
    for box in boxplots_PT1['boxes']:
        box.set_color(PT1_color)
    for median in boxplots_PT1['medians']:
        median.set_color(PT1_color)
    for flier in boxplots_PT1['fliers']:
        flier.set_color(PT1_color)


    boxplots_Ag = ax.boxplot(
        np.flip(Ag_vals).T,
        positions=np.arange(len(thresholds)) + offset,
        widths=[width] * len(thresholds),
        label="Agricultural"
    )
    Ag_color = 'green'
    for whisker in boxplots_Ag['whiskers']:
        whisker.set_color(Ag_color)
    for cap in boxplots_Ag['caps']:
        cap.set_color(Ag_color)
    for box in boxplots_Ag['boxes']:
        box.set_color(Ag_color)
    for median in boxplots_Ag['medians']:
        median.set_color(Ag_color)
    for flier in boxplots_Ag['fliers']:
        flier.set_color(Ag_color)


    # For each threshold, add line between PT1 and Ag vals
    num_scenarios = len(metrics[policy][thresholds[0]][metric_PT1])
    for threshold_idx, threshold in enumerate(thresholds):
        for scenario in range(num_scenarios):
            plt.plot([threshold_idx - offset, threshold_idx + offset],
                     [np.flip(PT1_vals)[threshold_idx, scenario], np.flip(Ag_vals)[threshold_idx, scenario]],
                     '--', color='black', marker='.', markerfacecolor='black',
                     markeredgewidth=0, linewidth=0.5)
            # FIX THIS!!!


    # inds = np.arange(1, len(medians) + 1)
    # # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.set_ylabel("{} values".format(metric_title))
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_xticklabels(np.flip(thresholds), rotation=45)
    ax.set_title(graph_title)
    legend = ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    fig.savefig('boxplots.png', format='png', transparent=False, bbox_extra_artists=[legend], bbox_inches='tight')

    # plt.savefig('boxplots.png', format='png', transparent=True)
    fig.show()


urban_ag_side_by_side(
    policy='demand_only',
    metric_PT1='reliability_PT1',
    metric_Ag='reliability_Ag',
    xlabel="SRI3 Indicator Threshold",
    ylabel="Failure frequency for urban and agricultural consumers",
    graph_title="Failure frequencies vs thresholds for demand restriction",
    thresholds=[-1, -0]
)

