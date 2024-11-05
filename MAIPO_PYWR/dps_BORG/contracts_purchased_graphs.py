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
# demand_restriction_factor = 0.9

contracts_purchased_list = [0, 200, 400, 600, 800, 1000, 1200]#[0, 30, 60, 90, 120, 150]

models = {}
for policy in ['contracts_only', 'demand_only', 'contract_and_demand', 'no_policy']:
    models[policy] = {}

for contracts_purchased in contracts_purchased_list:
    policies = {
        "contracts_only": {
            "contract_threshold_vals": -0.84 * np.ones(num_DP),  # threshold at which Chile's policies kick in
            "contract_action_vals": contracts_purchased * np.ones(num_DP)
        },
        # "demand_only": {
        #     "demand_threshold_vals": [threshold * np.ones(months_in_year)],
        #     "demand_action_vals": [np.ones(months_in_year), demand_restriction_factor * np.ones(months_in_year)]
        # },
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

        models[policy][contracts_purchased] = cur_model


#%% Make dictionary of metrics for each model
metrics = {}
for policy, policy_purchases in models.items():
    metrics[policy] = {}
    for contracts_purchased, model in policy_purchases.items():
        metrics[policy][contracts_purchased] = {
            "ff_PT1": np.asarray(model.recorders['failure_frequency_PT1'].values()),
            "ff_Ag": np.asarray(model.recorders['failure_frequency_Ag'].values()),
            "reliability_PT1": np.asarray(model.recorders['reliability_PT1'].values()),
            "reliability_Ag": np.asarray(model.recorders['reliability_Ag'].values()),
            "deficit_PT1": np.asarray(model.recorders['deficit PT1'].values()),
            "deficit_Ag": np.asarray(model.recorders['deficit Ag'].values()),
            "max_deficit_PT1": np.asarray(model.recorders['Maximum Deficit PT1'].values()),
            "max_deficit_Ag": np.asarray(model.recorders['Maximum Deficit Ag'].values()),
            "cost": np.asarray(model.recorders['TotalCost'].values())
        }


#%% Update deficit_PT1 and deficit_Ag to be ratios and not raw numbers

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
urban_demand = data["PT1"]
num_weeks = len(urban_demand)
total_urban_demand = urban_demand.sum()

agricultural_demand = 16.978
total_ag_demand = agricultural_demand * num_weeks  # technically slightly off, but very close

for policy, policy_purchases in models.items():
    for contracts_purchased, model in policy_purchases.items():
        metrics[policy][contracts_purchased]["deficit_PT1_proportion"] = metrics[policy][contracts_purchased]["deficit_PT1"] / total_urban_demand
        metrics[policy][contracts_purchased]["deficit_Ag_proportion"] = metrics[policy][contracts_purchased]["deficit_Ag"] / total_ag_demand
        # metrics[policy][contracts_purchased]["deficit_Ag"] /= total_ag_demand


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
    for contracts_purchased in contracts_purchased_list:
        recorders.append(func(models[policy][contracts_purchased].recorders[metric].values()))
    plt.plot(contracts_purchased_list, recorders, marker='o')
    plt.title('{} for {}'.format(metric, policy))
    plt.ylim(0, 1.2 * np.max(recorders))
    plt.show()


policy_list = ['contracts_only']#, 'demand_only', "contract_and_demand", 'no_policy']
metric_list = ['failure_frequency_PT1', 'failure_frequency_Ag',
               'Maximum Deficit PT1', 'Maximum Deficit Ag',
               'deficit PT1', 'deficit Ag']

for metric in metric_list:
    for policy in policy_list:
        plot_metric(policy, metric)


#%% Plot urban-ag failure frequency difference across the thresholds [NOT USED IN FINAL PLOTS]

def purchases_ff_gap_plot(policy):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(contracts_purchased_list), 15))
    Ag_vals = np.zeros((len(contracts_purchased_list), 15))
    for idx, contracts_purchased in enumerate(contracts_purchased_list):
        PT1_vals[idx, :] = metrics[policy][contracts_purchased]['ff_PT1']
        Ag_vals[idx, :] = metrics[policy][contracts_purchased]['ff_Ag']


    plt.gca().set_prop_cycle(plt.cycler('color', ['limegreen', 'turquoise', 'deeppink',
                                                  'forestgreen', 'darkslateblue', 'mediumvioletred']))
    ax.plot((PT1_vals - Ag_vals)[:, [0, 2, 4, 10, 12, 14]])
    # ax.plot(Ag_vals.T, '--')
    ax.set_xlabel('Contracts purchased')
    ax.set_ylabel("Failure frequency for PT1 minus Ag")
    ax.set_xticks(np.arange(len(contracts_purchased_list)))
    ax.set_xticklabels(contracts_purchased_list, rotation=45)
    ax.set_title('Failure frequency vs contracts purchased (with {})'.format(policy))
    ax.legend([1, 3, 5, 11, 13, 15])
    # First graph has these ylim values, using same for second for easier comparision

    fig.show()


purchases_ff_gap_plot("contracts_only")


#%% Box/violin plot of metrics for each scenario [NOT USED IN FINAL PLOTS]

def purchases_gap_violin_plot(policy, metric_PT1, metric_Ag, metric_title):
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(contracts_purchased_list), 15))
    Ag_vals = np.zeros((len(contracts_purchased_list), 15))
    for idx, contracts_purchased in enumerate(contracts_purchased_list):
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
        widths=[0.2] * len(contracts_purchased_list),
    )
    for median in boxplots['medians']:
        median.set_color('black')

    # inds = np.arange(1, len(medians) + 1)
    # # ax2.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    # ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    # ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

    ax.set_xlabel('Contracts purchased')
    ax.set_ylabel("Relative unmet demand distribution across scenarios (urban-agriculture)")
    ax.set_xticks(np.arange(len(contracts_purchased_list)) + 1)
    ax.set_xticklabels(contracts_purchased_list, rotation=45)
    ax.set_title('Unmet demand gap vs. contracts purchased for buying contracts')

    plt.savefig('violins.png', format='png', transparent=True)
    fig.show()


purchases_gap_violin_plot(
    policy='contracts_only',
    metric_PT1='deficit_PT1_proportion',
    metric_Ag='deficit_Ag_proportion',
    metric_title="Cost"
)


#%% Scenario title parser

num_scenarios = 15
scenarios = pd.read_csv('../data/COLORADO.csv').columns[1::]


def parse_scenario_title(scenario):
    tokens = scenario.split('_')
    rcp_val = int(tokens[1])
    pp = int(tokens[3])
    temp = int(tokens[5])

    return rcp_val, pp, temp


def get_label(scenario):
    rcp_val, pp, temp = parse_scenario_title(scenario)
    # REVISIT BELOW LINE WHEN I FIND OUT WHAT THESE MEAN
    return "Forcing: {}. Precipitation: {}%. Temperature: {}%".format(rcp_val, pp, temp)


#%% Parallel coordinates plots to see how performace and cost interact [NOT USED IN FINAL PLOTS]


metric_list = ['ff_PT1', 'ff_Ag', 'deficit_PT1_proportion', 'deficit_Ag_proportion',
           'max_deficit_PT1', 'max_deficit_Ag', 'cost']

par_coord_data = []
for policy in policy_list:
    for contracts_purchased in contracts_purchased_list:
        for scenario_idx in range(len(scenarios)):
            new_row = [policy, scenario_idx, contracts_purchased]
            for metric in metric_list:
                new_row.append(metrics[policy][contracts_purchased][metric][scenario_idx])
            par_coord_data.append(new_row)


par_coord_df = pd.DataFrame(par_coord_data, columns=['policy', 'scenario', 'contracts_purchased'] + metric_list)
print(par_coord_df.head())


fig = go.Figure(
    data=go.Parcoords(
        line=dict(color=par_coord_df["contracts_purchased"],
                  showscale=True),  # COLORSCALE CAN BE ADDED OPTIONALLY
        dimensions=list([
            dict(
                range=[0, 0.3],
                label="Urban failure frequency",
                values=par_coord_df["ff_PT1"]
            ),
            dict(
                range=[0, 0.3],
                label="Agricultural failure frequency",
                values=par_coord_df["ff_Ag"]
            ),
            dict(
                range=[0, 8],
                label="Urban max deficit",
                values=par_coord_df["max_deficit_PT1"]
            ),
            dict(
                range=[0, 8],
                label="Agricultural max deficit",
                values=par_coord_df["max_deficit_Ag"]
            ),
            dict(
                range=[0, 4e9],
                label="Cost",
                values=par_coord_df["cost"]
            )
        ])
    )
)

fig.show(renderer="browser")


#%% Boxplot for any metric

def box_plot(policy, metric, xlabel, ylabel, graph_title, ylim=None):
    fig, ax = plt.subplots()
    metric_vals = np.zeros((len(contracts_purchased_list), 15))
    for idx, contracts_purchased in enumerate(contracts_purchased_list):
        if idx == len(contracts_purchased_list):
            break
        metric_vals[idx, :] = metrics[policy][contracts_purchased][metric]

    # width = 0.2

    boxplots = ax.boxplot(
        metric_vals.T,
        positions=np.arange(len(contracts_purchased_list)),
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(contracts_purchased_list)))
    ax.set_xticklabels(contracts_purchased_list, rotation=45)
    ax.set_ylim(ylim)
    ax.set_title(graph_title)
    legend = ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')
    # fig.savefig('boxplots.png', format='png', transparent=False, bbox_extra_artists=[legend], bbox_inches='tight')

    plt.savefig('C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\outputs\\Urban Ag results\\Boxplot: contracts purchased, {}, {}.png'.format(policy, metric), format='png', transparent=True)
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
    fig, ax = plt.subplots()
    PT1_vals = np.zeros((len(contracts_purchased_list), 15))
    Ag_vals = np.zeros((len(contracts_purchased_list), 15))
    for idx, contracts_purchased in enumerate(contracts_purchased_list):
        if idx == len(contracts_purchased_list):
            break
        PT1_vals[idx, :] = metrics[policy][contracts_purchased][metric_PT1]
        Ag_vals[idx, :] = metrics[policy][contracts_purchased][metric_Ag]

    width = 0.2
    offset = 0.15

    boxplots_PT1 = ax.boxplot(
        100*PT1_vals.T,
        positions=np.arange(len(contracts_purchased_list)) - offset,
        widths=[width] * (len(contracts_purchased_list)),
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
        median.set_color("white")
    for flier in boxplots_PT1['fliers']:
        flier.set_color(PT1_color)


    boxplots_Ag = ax.boxplot(
        100*Ag_vals.T,
        positions=np.arange(len(contracts_purchased_list)) + offset,
        widths=[width] * (len(contracts_purchased_list)),
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
        median.set_color("white")
    for flier in boxplots_Ag['fliers']:
        flier.set_color(Ag_color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(contracts_purchased_list)))
    ax.set_xticklabels(["Baseline", *contracts_purchased_list[1:]], rotation=45)
    ax.set_ylim(ylim)
    ax.set_title(graph_title)
    legend = ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper left')

    ax.set_facecolor("#ebf5f6")
    fig.patch.set_facecolor("#cde5e9")

    plt.subplots_adjust(bottom=0.2)

    # fig.savefig('Boxplot pairs: contracts purchased, {}, {}.png'.format(policy, metric), format='png', transparent=False, bbox_extra_artists=[legend], bbox_inches='tight')

    # plt.savefig('boxplots.png', format='png', transparent=True)
    plt.savefig('{}.png'.format(graph_title), format='png', transparent=False)
    fig.show()


box_plots_without_connectors(
    policy='contracts_only',
    metric_PT1='reliability_PT1',
    metric_Ag='reliability_Ag',
    xlabel="Number of contracts purchased",
    ylabel="Water supply reliability (%)",
    graph_title="Reliability vs Water Contracts Purchased",
    ylim=[70, 100]#[0, 1]
)

# box_plots_without_connectors(
#     policy='contracts_only',
#     metric_PT1='max_deficit_PT1',
#     metric_Ag='max_deficit_Ag',
#     xlabel="Contracts purchased",
#     ylabel="Maximum deficit for urban and agricultural consumers",
#     graph_title="Maximum deficit vs contracts purchased",
#     ylim=None#[0, 8]
# )
#
# box_plots_without_connectors(
#     policy='contracts_only',
#     metric_PT1='deficit_PT1_proportion',
#     metric_Ag='deficit_Ag_proportion',
#     xlabel="Contracts purchased",
#     ylabel="Proportion of demand unmet for urban and agricultural consumers",
#     graph_title="Proportion of demand unmet vs contracts purchased",
#     ylim=None#[0, 0.07]
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
extreme_contracts = [200, 1200]
for policy in ["contracts_only"]:
    par_coord_policy_data = []
    for scenario_idx in range(len(scenarios)):
        new_row = [scenario_idx]
        new_row += [*parse_scenario_title(scenarios[scenario_idx])]
        for metric in metric_list:
            for contracts_purchased in extreme_contracts:
                print(policy, contracts_purchased, metric, scenario_idx)
                new_row.append(metrics[policy][contracts_purchased][metric][scenario_idx])
        par_coord_policy_data.append(new_row)
    par_coord_data[policy] = par_coord_policy_data

metric_contract_pairs = []
for metric in metric_list:
    for contracts_purchased in extreme_contracts:
        metric_contract_pairs.append("{}, {}".format(metric, contracts_purchased))

par_coord_df = {}
for policy in par_coord_data.keys():
    par_coord_df[policy] = pd.DataFrame(
        par_coord_data[policy],
        columns=[
                    'scenario index',
                    'rcp val',
                    'pp',
                    'temp'
                ] + metric_contract_pairs
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
        cax=ax.inset_axes([1.02, 0.1, 0.05, 0.8])
    )
    cbar.set_label('Precipitation percentile change')
    cbar.ax.set_yticklabels(["5%", "25%", "50%", "75%", "95%"])

    ax.set_facecolor("#ebf5f6")
    fig.patch.set_facecolor("#cde5e9")

    plt.subplots_adjust(right=0.85)

    # plt.ylabel("Water supply reliability")


    plt.savefig('{}.png'.format(title), format='png', transparent=False)
    plt.show()


extreme_par_coords_matplotlib(
    "contracts_only",
    "Reliability with 200 Water Contracts Purchased",
    [
        "reliability_PT1, 200",
        "reliability_Ag, 200"
    ],
    [
        "Urban reliability (%)",
        "Agricultural reliability (%)",
    ],
    ylim=[70, 100],
)

extreme_par_coords_matplotlib(
    "contracts_only",
    "Reliability with 1200 Water Contracts Purchased",
    [
        "reliability_PT1, 1200",
        "reliability_Ag, 1200"
    ],
    [
        "Urban reliability (%)",
        "Agricultural reliability (%)",
    ],
    ylim=[70, 100]
)

#%%

# extreme_par_coords_matplotlib(
#     "contracts_only",
#     "Max deficit for purchasing 1200 contracts",
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


# extreme_par_coords_matplotlib(
#     "demand_only",
#     "Reliability with no policy",
#     [
#         "reliability_PT1, 0",
#         "reliability_Ag, 0"
#     ],
#     [
#         "Urban reliability",
#         "Agricultural reliability",
#     ],
#     ylim=[0.7, 1],
# )
#
# extreme_par_coords_matplotlib(
#     "demand_only",
#     "Reliability with 1200 contracts purchased",
#     [
#         "reliability_PT1, 1200",
#         "reliability_Ag, 1200"
#     ],
#     [
#         "Urban reliability",
#         "Agricultural reliability",
#     ],
#     ylim=[0.7, 1]
# )

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



# #%% Parallel coordinates plots with metrics at most and least extreme values
#
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
# par_coord_data = []
# extreme_contracts = [0, 1200]
# policy = "contracts_only"
# for scenario_idx in range(len(scenarios)):
#     new_row = [scenario_idx]
#     new_row += [*parse_scenario_title(scenarios[scenario_idx])]
#     for metric in metric_list:
#         for contracts_purchased in extreme_contracts:
#             new_row.append(metrics[policy][contracts_purchased][metric][scenario_idx])
#     par_coord_data.append(new_row)
#
# metric_contract_pairs = []
# for metric in metric_list:
#     for contracts_purchased in extreme_contracts:
#         metric_contract_pairs.append("{}, {}".format(metric, contracts_purchased))
#
# par_coord_df = pd.DataFrame(
#     par_coord_data,
#     columns=[
#                 'scenario index',
#                 'rcp val',
#                 'pp',
#                 'temp'
#             ] + metric_contract_pairs
# )
# # print(par_coord_df.head())
#
#
# def extreme_par_coords(title, metrics, labels, ranges=None, colorscale='scenario index'):
#     fig = go.Figure(
#         data=go.Parcoords(
#             line=dict(color=par_coord_df[colorscale],
#                       showscale=True),  # COLORSCALE CAN BE ADDED OPTIONALLY
#             dimensions=list([
#                 dict(
#                     range=ranges[i],
#                     label=labels[i],
#                     values=par_coord_df[metrics[i]],
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
#
#
# def extreme_par_coords_matplotlib(title, metrics, labels, ylim=None):
#     metrics_by_scenario = par_coord_df[metrics].T
#     cmap = plt.get_cmap('viridis', 5)
#
#     fig, ax = plt.subplots(1, 1, figsize=(8, 6))
#     for i in range(num_scenarios):
#         ax.plot(labels, metrics_by_scenario[i], c=cmap(i % 5))
#
#     plt.title(title)
#     plt.ylim(ylim)
#
#     sm = plt.cm.ScalarMappable(cmap=cmap)#, norm=norm)
#     sm.set_array([])
#
#     np.linspace(0, 2, num_scenarios + 1)
#     plt.colorbar(sm, ticks=[], cax=ax.inset_axes([1.02, 0.1, 0.05, 0.8]), label='Precipitation change')
#     plt.show()
#
#
# extreme_par_coords_matplotlib(
#     "Reliability for 0 contracts purchased, colored by precipitation change",
#     [
#         "reliability_PT1, 0",
#         "reliability_Ag, 0"
#     ],
#     [
#         "Urban reliability (0 contracts purchased)",
#         "Agricultural reliability (0 contracts purchased)",
#     ],
#     ylim=[0.7, 1],
# )
#
# extreme_par_coords_matplotlib(
#     "Reliability for 1200 contracts purchased, colored by precipitation change",
#     [
#         "reliability_PT1, 1200",
#         "reliability_Ag, 1200"
#     ],
#     [
#         "Urban reliability (1200 contracts purchased)",
#         "Agricultural reliability (1200 contracts purchased)",
#     ],
#     ylim=[0.7, 1]
# )
#
#
# extreme_par_coords_matplotlib(
#     "Max deficit for 0 contracts purchased, colored by precipitation change",
#     [
#         "max_deficit_PT1, 0",
#         "max_deficit_Ag, 0"
#     ],
#     [
#         "Urban max deficit (0 contracts purchased)",
#         "Agricultural max deficit (0 contracts purchased)",
#     ],
#     ylim=[0, 8]
# )
#
# extreme_par_coords_matplotlib(
#     "Max deficit for 1200 contracts purchased, colored by precipitation change",
#     [
#         "max_deficit_PT1, 1200",
#         "max_deficit_Ag, 1200"
#     ],
#     [
#         "Urban max deficit (1200 contracts purchased)",
#         "Agricultural max deficit (1200 contracts purchased)",
#     ],
#     ylim=[0, 8]
# )
#
#
# extreme_par_coords_matplotlib(
#     "Unmet demand proportion for 0 contracts purchased, colored by precipitation change",
#     [
#         "deficit_PT1_proportion, 0",
#         "deficit_Ag_proportion, 0"
#     ],
#     [
#         "Urban unmet demand proportion (0 contracts purchased)",
#         "Agricultural unmet demand proportion (0 contracts purchased)",
#     ],
#     ylim=[0, 0.1]
# )
#
# extreme_par_coords_matplotlib(
#     "Unmet demand proportion for 1200 contracts purchased, colored by precipitation change",
#     [
#         "deficit_PT1_proportion, 1200",
#         "deficit_Ag_proportion, 1200"
#     ],
#     [
#         "Urban unmet demand proportion (1200 contracts purchased)",
#         "Agricultural unmet demand proportion (1200 contracts purchased)",
#     ],
#     ylim=[0, 0.07]
# )


#%%

# extreme_par_coords(
#     "Reliability for 0 contracts purchased, colored by precipitation change",
#     [
#         "reliability_PT1, 0",
#         "reliability_Ag, 0"
#     ],
#     [
#         "Urban reliability (0 contracts purchased)",
#         "Agricultural reliability (0 contracts purchased)",
#     ],
#     ranges=[
#         [0.7, 1],
#         [0.7, 1]
#     ],
#     colorscale='pp'
# )
#
# extreme_par_coords(
#     "Reliability for 1200 contracts purchased, colored by precipitation change",
#     [
#         "reliability_PT1, 1200",
#         "reliability_Ag, 1200"
#     ],
#     [
#         "Urban reliability (1200 contracts purchased)",
#         "Agricultural reliability (1200 contracts purchased)",
#     ],
#     ranges=[
#         [0.7, 1],
#         [0.7, 1]
#     ],
#     colorscale='pp'
# )
#
#
# extreme_par_coords(
#     "Max deficit for 0 contracts purchased, colored by precipitation change",
#     [
#         "max_deficit_PT1, 0",
#         "max_deficit_Ag, 0"
#     ],
#     [
#         "Urban max deficit (0 contracts purchased)",
#         "Agricultural max deficit (0 contracts purchased)",
#     ],
#     ranges=[
#         [0, 8],
#         [0, 8]
#     ],
#     colorscale='pp'
# )
#
# extreme_par_coords(
#     "Max deficit for 1200 contracts purchased, colored by precipitation change",
#     [
#         "max_deficit_PT1, 1200",
#         "max_deficit_Ag, 1200"
#     ],
#     [
#         "Urban max deficit (1200 contracts purchased)",
#         "Agricultural max deficit (1200 contracts purchased)",
#     ],
#     ranges=[
#         [0, 8],
#         [0, 8]
#     ],
#     colorscale='pp'
# )
#
#
# extreme_par_coords(
#     "Unmet demand proportion for 0 contracts purchased, colored by precipitation change",
#     [
#         "deficit_PT1_proportion, 0",
#         "deficit_Ag_proportion, 0"
#     ],
#     [
#         "Urban unmet demand proportion (0 contracts purchased)",
#         "Agricultural unmet demand proportion (0 contracts purchased)",
#     ],
#     ranges=[
#         [0.7, 1],
#         [0.7, 1]
#     ],
#     colorscale='pp'
# )
#
# extreme_par_coords(
#     "Unmet demand proportion for 1200 contracts purchased, colored by precipitation change",
#     [
#         "reliability_PT1, 1200",
#         "reliability_Ag, 1200"
#     ],
#     [
#         "Urban reliability (1200 contracts purchased)",
#         "Agricultural reliability (1200 contracts purchased)",
#     ],
#     ranges=[
#         [0.7, 1],
#         [0.7, 1]
#     ],
#     colorscale='pp'
# )





# extreme_par_coords(
#     "Reliability for 0 and 1200 contracts purchased, colored by precipitation change",
#     [
#         'reliability_PT1, 0',
#         'reliability_PT1, 1200',
#         'reliability_Ag, 0',
#         'reliability_Ag, 1200',
#     ],
#     [
#         "Urban reliability (no contracts purchased)",
#         "Urban reliability (120 contracts purchased)",
#         "Agricultural reliability (no contracts purchased)",
#         "Agricultural reliability (120 contracts purchased)",
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
#     "Maximum deficit for 0 and 1200 contracts purchased, colored by precipitation change",
#     [
#         'max_deficit_PT1, 0',
#         'max_deficit_PT1, 1200',
#         'max_deficit_Ag, 0',
#         'max_deficit_Ag, 1200',
#     ],
#     [
#         "Urban max deficit (no contracts purchased)",
#         "Urban max deficit (1200 contracts purchased)",
#         "Agricultural max deficit (no contracts purchased)",
#         "Agricultural max deficit (1200 contracts purchased)",
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
#     "Unmet demand proportion for 0 and 1200 contracts purchased, colored by precipitation change",
#     [
#         'deficit_PT1_proportion, 0',
#         'deficit_PT1_proportion, 1200',
#         'deficit_Ag_proportion, 0',
#         'deficit_Ag_proportion, 1200',
#     ],
#     [
#         "Urban unmet demand proportion (no contracts purchased)",
#         "Urban unmet demand proportion (1200 contracts purchased)",
#         "Agricultural unmet demand proportion (no contracts purchased)",
#         "Agricultural unmet demand proportion (1200 contracts purchased)",
#     ],
#     ranges=[
#         [0, 0.07],
#         [0, 0.07],
#         [0, 0.07],
#         [0, 0.07]
#     ],
#     colorscale='pp'
# )


#%% BELOW THIS SECTION IS JUST EXPERIMENTATION!!!


#%% Get times when contracts are activated [NOT USED IN FINAL PLOTS]

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\SRI3.csv")
num_weeks = 1560  # number of time steps in timestepper
num_scenarios = 15
threshold = -0.84
scenario_names = data.columns[1:]
is_activated = np.zeros((num_weeks, num_scenarios))
decision_weeks = np.zeros(num_weeks)
is_currently_activated = np.zeros(num_scenarios)
for week in range(num_weeks):
    if week % 26 == 0:
        decision_weeks[week] = 1
        for scenario in range(num_scenarios):
            is_currently_activated[scenario] = (data.loc[week + 832, scenario_names[scenario]] <= threshold)
    is_activated[week, :] = is_currently_activated


#%% Test to see when contracts are activated and how it affects flow [NOT USED IN FINAL PLOTS]

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\SRI3.csv")
num_weeks = 1560  # number of time steps in timestepper
scenario_names = data.columns[1:]

extra_data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
PT1_demand = extra_data['PT1']

def flow_over_time_plot(output, contracts_purchased, scenario):
    if output == 'PT1':
        plt.plot(
            range(num_weeks),
            PT1_demand[-1560:],
            # models['contracts_only'][0].recorders['PT1 demand recorder'].data[:, 0],
            label="Demanded urban flow"
        )
        # for scenario in range(1):
        # for contracts_purchased in contracts_purchased_list:
        plt.plot(
            range(num_weeks),
            models['contracts_only'][contracts_purchased].recorders['PT1 flow'].data[:, scenario],
            linewidth=1,
            label="PT1 flow"
        )

        plt.plot(
            range(num_weeks),
            models['contracts_only'][contracts_purchased].recorders['Total reservoir inflow'].data[:, scenario],
            linewidth=1,
            label="Total inflow"
        )

        plt.plot(
            range(num_weeks),
            decision_weeks,
            linewidth=1,
            label="Decision weeks"
        )

        scenario_name = scenario_names[scenario]
        plt.plot(
            range(num_weeks),
            data[scenario_name][-1560:],
            # models['contracts_only'][0].recorders['PT1 demand recorder'].data[:, 0],
            linewidth=1,
            label="SRI3"
        )



        plt.fill(is_activated[:, 0] * 30, alpha=0.2, label='Times contracts have been purchased')
        # plt.plot(decision_weeks)

        plt.ylim([-3, 30])
        plt.xlabel("Week")
        plt.ylabel("Flow")

        plt.title(
            "Urban flow over time with {} contracts purchased\nScenario: RCP 6.0, 50% precipitation, 50% temperature".format(
                contracts_purchased))
        plt.legend()
        plt.show()

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
    flow_over_time_plot('PT1', contracts_purchased, 7)

# flow_over_time_plot('Ag', 0, 7)
# plt.fill(is_activated[:, 0]*20, alpha=0.2)
# plt.show()

#%% [NOT USED IN FINAL PLOTS]
num_contracts = 150
plt.plot(models['contracts_only'][num_contracts].recorders['Remaining water rights per week'].data[:, 7])
plt.plot(models['contracts_only'][num_contracts].recorders['Agriculture flow'].data[:, 7])
plt.ylim([0, 150])
plt.show()



#%%
data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
print(data["PT1"].mean())
print((data["PT1"] + data["PT2"]).mean())
