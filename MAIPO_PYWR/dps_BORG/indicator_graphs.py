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
policies = {
    "contracts_only": {
        "contract_threshold_vals": -0.84*np.ones(num_DP),  # threshold at which Chile's policies kick in
        "contract_action_vals": contracts_purchased*np.ones(num_DP)
    },
    "demand_only": {
        "demand_threshold_vals": [-0.84*np.ones(months_in_year)],
        "demand_action_vals": [np.ones(months_in_year), demand_restriction_factor*np.ones(months_in_year)]
    },
    "contract_and_demand": {
        "contract_threshold_vals": -0.84 * np.ones(num_DP),  # threshold at which Chile's policies kick in
        "contract_action_vals": contracts_purchased * np.ones(num_DP),
        "demand_threshold_vals": [-0.84 * np.ones(months_in_year)],
        "demand_action_vals": [np.ones(months_in_year), demand_restriction_factor * np.ones(months_in_year)]
    },
    "no_policy": {}
}

indicators = ['SRI3', 'SRI6', 'SRI12']

models = {}
for policy, params in policies.items():

    indicator_models = {}
    for indicator in indicators:
        cur_model = make_model(**params, indicator=indicator)
        cur_model.check()
        cur_model.check_graph()
        cur_model.find_orphaned_parameters()
        cur_model.run()

        indicator_models[indicator] = cur_model

    models[policy] = indicator_models


#%% Make dictionary of metrics for each model
metrics = {}
for policy, policy_indicators in models.items():
    metrics[policy] = {}
    for indicator, model in policy_indicators.items():
        metrics[policy][indicator] = {
            "ff_PT1": np.asarray(model.recorders['failure_frequency_PT1'].values()),
            "ff_Ag": np.asarray(model.recorders['failure_frequency_Ag'].values()),
            "deficit_PT1": np.asarray(model.recorders['deficit PT1'].values()),
            "deficit_Ag": np.asarray(model.recorders['deficit Ag'].values()),
            "max_deficit_PT1": np.asarray(model.recorders['Maximum Deficit PT1'].values()),
            "max_deficit_Ag": np.asarray(model.recorders['Maximum Deficit Ag'].values()),
            "cost": np.asarray(model.recorders['TotalCost'].values())
        }

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


#%% Parallel coordinates plot for single policy/indicator pair

results_list = {}
for policy, policy_indicators in metrics.items():
    results_list[policy] = {}
    for indicator, indicator_metrics in policy_indicators.items():
        cur_results = pd.DataFrame.from_dict(indicator_metrics, orient='index').T
        cur_results['scenario'] = np.arange(1, len(cur_results.index) + 1)
        cur_results[['rad_forcing', 'precip', 'temp']] = cur_results.apply(
            lambda row: parse_scenario_title(scenarios[round(row.scenario) - 1]), axis='columns', result_type='expand'
        )
        results_list[policy][indicator] = cur_results


def policy_indicator_metrics_graph(policy, indicator, color_var='scenario'):
    cur_results = results_list[policy][indicator]
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=cur_results[color_var],
                      showscale=True),  # COLORSCALE CAN BE ADDED OPTIONALLY
            dimensions=list([
                dict(
                    range=[0, 0.3],
                    label="Urban failure frequency",
                    values=cur_results["ff_PT1"]
                ),
                dict(
                    range=[0, 0.3],
                    label="Agricultural failure frequency",
                    values=cur_results["ff_Ag"]
                ),
                dict(
                    range=[0, 8],
                    label="Urban max deficit",
                    values=cur_results["max_deficit_PT1"]
                ),
                dict(
                    range=[0, 8],
                    label="Agricultural max deficit",
                    values=cur_results["max_deficit_Ag"]
                ),
                dict(
                    range=[0, 3e9],
                    label="Cost",
                    values=cur_results["cost"]
                )
            ])
        )
    )

    fig.update_layout(title="Metrics for policy {}, indicator {}, scenarios grouped by {}"
                      .format(policy, indicator, color_var))

    fig.show(renderer="browser")


color_vars = ['scenario', 'rad_forcing', 'precip', 'temp']

#%% Parallel axes plot of metrics for each policy and indicator
for policy in policies:
    for indicator in indicators:
        policy_indicator_metrics_graph(policy, indicator)


#%% Get total urban and ag demand

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
urban_demand = data["PT1"]
num_weeks = len(urban_demand)
total_urban_demand = urban_demand.sum()

agricultural_demand_profile = np.array([18.01638039, 17.94856465, 17.84307351, 17.8053981, 17.7225122, 17.74511745,
                                        17.76772269, 17.77525777, 17.78279286, 17.91088925, 17.97870498, 18.15201186])
total_ag_demand = np.mean(agricultural_demand_profile) * num_weeks  # technically slightly off, but very close


#%% Graph of metric for all scenarios, SRI time independent variable

def SRI_line_plot(policy, metric, color_var='scenario'):
    fig, ax = plt.subplots()
    vals_to_plot = np.zeros((len(indicators), num_scenarios))
    for i in range(len(indicators)):
        indicator = indicators[i]
        vals_to_plot[i, :] = metrics[policy][indicator][metric]

    # colormap = plt.cm.gist_ncar
    # plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, 15))))
    ax.plot(vals_to_plot, color="darkcyan")
    mean_line = np.mean(vals_to_plot, axis=1)
    ax.plot(mean_line, 'k--')
    ax.set_xlabel('Indicator')
    ax.set_ylabel(metric)
    ax.set_xticks(np.arange(len(indicators)))
    ax.set_xticklabels(indicators, rotation=45)
    ax.set_title('{} each scenario, using {}'.format(metric, policy))
    fig.show()

for policy in policies:
    SRI_line_plot(policy, 'deficit_Ag', color_var='scenario')


#%% Plot line of failure frequency and total deficit fraction depending on precip/temp

indicator = "SRI3"  # Last experiment suggested SRI3 was best performing

precip_vals = [0.05, 0.25, 0.50, 0.75, 0.95]

# Plot PT1 and Ag metric on same plot
def precip_PT1_Ag_plot(policy, metric):
    if metric == "ff":
        PT1_vals = metrics[policy][indicator]['ff_PT1'].reshape(3, 5)
        Ag_vals = metrics[policy][indicator]['ff_Ag'].reshape(3, 5)
    if metric == "deficit_proportion":
        PT1_vals = (metrics[policy][indicator]['deficit_PT1'].reshape(3, 5)
                    / total_urban_demand)
        Ag_vals = (metrics[policy][indicator]['deficit_Ag'].reshape(3, 5)
                    / total_ag_demand)


    fig, ax = plt.subplots()
    plt.gca().set_prop_cycle(plt.cycler('color', ['yellow', 'orange', 'red']))
    ax.plot(PT1_vals.T, '-')
    ax.plot(Ag_vals.T, '--')
    ax.set_xlabel('Precip percent')
    ax.set_ylabel(metric)
    ax.set_xticks(np.arange(len(precip_vals)))
    ax.set_xticklabels(precip_vals, rotation=45)
    ax.set_title('{} for each RCP, as function of precip'.format(metric, policy))

    fig.show()


# Plot difference between PT1 and Ag metric
def precip_gap_plot(policy, metric):
    if metric == "ff":
        PT1_vals = metrics[policy][indicator]['ff_PT1'].reshape(3, 5)
        Ag_vals = metrics[policy][indicator]['ff_Ag'].reshape(3, 5)
    if metric == "deficit_proportion":
        PT1_vals = (metrics[policy][indicator]['deficit_PT1'].reshape(3, 5)
                    / total_urban_demand)
        Ag_vals = (metrics[policy][indicator]['deficit_Ag'].reshape(3, 5)
                    / total_ag_demand)


    fig, ax = plt.subplots()
    plt.gca().set_prop_cycle(plt.cycler('color', ['yellow', 'orange', 'red']))
    ax.plot(PT1_vals.T - Ag_vals.T)
    # ax.plot(Ag_vals.T, '--')
    ax.set_xlabel('Precip percent')
    ax.set_ylabel("{} for PT1 minus Ag".format(metric))
    ax.set_xticks(np.arange(len(precip_vals)))
    ax.set_xticklabels(precip_vals, rotation=45)
    ax.set_title('{} for each RCP, as function of precip'.format(metric, policy))

    fig.show()


precip_gap_plot("contracts_only", "ff")


