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
policies = {
    "contracts_only": {
        "contract_threshold_vals": -0.84*np.ones(num_DP),  # threshold at which Chile's policies kick in
        "contract_action_vals": 30*np.ones(num_DP)
    },
    "demand_only": {
        "demand_threshold_vals": [-0.84*np.ones(months_in_year)],
        "demand_action_vals": [np.ones(months_in_year), 0.9*np.ones(months_in_year)]
    },
    "contract_and_demand": {
        "contract_threshold_vals": -0.84 * np.ones(num_DP),  # threshold at which Chile's policies kick in
        "contract_action_vals": 30 * np.ones(num_DP),
        "demand_threshold_vals": [-0.84 * np.ones(months_in_year)],
        "demand_action_vals": [np.ones(months_in_year), 0.9 * np.ones(months_in_year)]
    },
    "no_policy": {}
}

models = {}
for policy, params in policies.items():
    models[policy] = make_model(**params)

    models[policy].check()
    models[policy].check_graph()
    models[policy].find_orphaned_parameters()

    models[policy].run()

#%% Make dictionary of metrics for each model

metrics = {}
for policy, model in models.items():
    metrics[policy] = {
        "ff_PT1": np.asarray(model.recorders['failure_frequency_PT1'].values()),
        "ff_Ag": np.asarray(model.recorders['failure_frequency_Ag'].values()),
        "max_deficit_PT1": np.asarray(model.recorders['Maximum Deficit PT1'].values()),
        "max_deficit_Ag": np.asarray(model.recorders['Maximum Deficit Ag'].values()),
        "cost": np.asarray(model.recorders['TotalCost'].values())
    }


#%% Parallel coordinates plot for single policy

num_scenarios = 15
scenarios = pd.read_csv('../data/COLORADO.csv').columns[1::]


def parse_scenario_title(scenario):
    tokens = scenario.split('_')
    rcp_val = int(tokens[1]) / 10
    pp = int(tokens[3])
    temp = int(tokens[5])

    return rcp_val, pp, temp


def get_label(scenario):
    rcp_val, pp, temp = parse_scenario_title(scenario)
    # REVISIT BELOW LINE WHEN I FIND OUT WHAT THESE MEAN
    return "Forcing: {}. Precipitation: {}%. Temperature: {}%".format(rcp_val, pp, temp)


results_list = {}
for policy, policy_metrics in metrics.items():
    cur_results = pd.DataFrame.from_dict(policy_metrics, orient='index').T
    cur_results['scenario'] = np.arange(1, len(cur_results.index) + 1)
    cur_results[['rad_forcing', 'precip', 'temp']] = cur_results.apply(
        lambda row: parse_scenario_title(scenarios[round(row.scenario) - 1]), axis='columns', result_type='expand'
    )
    results_list[policy] = cur_results


# Graph policy metrics, with express
def policy_metrics_graph_express(policy, color_var):
    fig = px.parallel_coordinates(
        results_list[policy],
        dimensions=["ff_PT1", "ff_Ag", "max_deficit_PT1", "max_deficit_Ag", "cost"],
        color=color_var,
        title="Metrics for {}, scenarios grouped by {}".format(policy, color_var)
    )

    fig.show(renderer="browser")


# Graph policy metrics, without express
def policy_metrics_graph(policy, color_var='scenario'):
    cur_results = results_list[policy]
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

    fig.update_layout(title="Metrics for {}, scenarios grouped by {}".format(policy, color_var))

    fig.show(renderer="browser")


scenario_characteristics = {"rad_forcing", "precip", "temp"}  # Temp since equiv to precip, include?


urban_ag_results_list = {}
for policy, results in results_list.items():
    PT1_results = results.loc[:, ['ff_PT1', 'max_deficit_PT1', 'cost', 'scenario', 'rad_forcing', 'precip', 'temp']]
    PT1_results.columns = ['ff', 'max_deficit', 'cost', 'scenario', 'rad_forcing', 'precip', 'temp']
    PT1_results['is_PT1'] = 1
    PT1_results['is_Ag'] = 0

    Ag_results = results.loc[:, ['ff_Ag', 'max_deficit_Ag', 'cost', 'scenario', 'rad_forcing', 'precip', 'temp']]
    Ag_results.columns = ['ff', 'max_deficit', 'cost', 'scenario', 'rad_forcing', 'precip', 'temp']
    Ag_results['is_PT1'] = 0
    Ag_results['is_Ag'] = 1

    urban_ag_results_list[policy] = pd.concat([PT1_results, Ag_results])


def urban_ag_metrics_graph(policy, color_var='scenario'):
    cur_results = urban_ag_results_list[policy]
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=cur_results[color_var],
                      showscale=True),  # COLORSCALE CAN BE ADDED OPTIONALLY
            dimensions=list([
                dict(
                    range=[0, 0.3],
                    label="Failure frequency",
                    values=cur_results["ff"]
                ),
                dict(
                    range=[0, 8],
                    label="Max deficit",
                    values=cur_results["max_deficit"]
                ),
                dict(
                    range=[0, 3e9],
                    label="Cost",
                    values=cur_results["cost"]
                )
            ])
        )
    )

    fig.update_layout(title="Metrics for {}, scenarios grouped by {}".format(policy, color_var))

    fig.show(renderer="browser")


# # Using Pandas to
# plt.figure(figsize=(10, 6))
# cur_results = urban_ag_results_list['contracts_only']
# parallel_coordinates(cur_results[cur_results['is_PT1'] == 1], 'rad_forcing', color=('red', 'orange', 'yellow'), linestyle='-')
# parallel_coordinates(cur_results[cur_results['is_PT1'] == 0], 'rad_forcing', color=('red', 'orange', 'yellow'), linestyle='--')
# plt.show()


# Plot separate lines for urban and agricultural demands
# for policy in policies:
    # One axis for failure frequency, one for deficit
    # Maybe still have cost as an axis, will help clarify what's together
    # Make solid line for PT1 and dotted for Ag

# urban_ag_metrics_graph('contracts_only', color_var='precip')

