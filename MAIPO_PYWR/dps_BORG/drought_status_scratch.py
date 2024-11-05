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
contracts_purchased = 90

thresholds = [-0.84]

drought_status_params = ['drought_status_single_day', 'drought_status_min_recent', 'drought_status_second_min_recent',
                         'drought_status_threshold_passes', 'drought_status_discounted_avg']#['drought_status_single_day', 'drought_status_single_day_using_agg']

models = {}
for policy in ['contracts_only']:
    models[policy] = {}

for threshold in thresholds:
    policies = {
        "contracts_only": {
            "contract_threshold_vals": threshold * np.ones(num_DP),  # threshold at which Chile's policies kick in
            "contract_action_vals": contracts_purchased * np.ones(num_DP)
        }
    }
    for policy, params in policies.items():
        models[policy][threshold] = {}
        for drought_status_param in drought_status_params:
            cur_model = make_model(drought_status_agg=drought_status_param, **params)
            cur_model.check()
            cur_model.check_graph()
            print(cur_model.find_orphaned_parameters())
            cur_model.run()

            models[policy][threshold][drought_status_param] = cur_model


#%% Make dictionary of metrics for each model
metrics = {}
for policy, policy_thresholds in models.items():
    metrics[policy] = {}
    for threshold, threshold_drought_status_params in policy_thresholds.items():
        metrics[policy][threshold] = {}
        for drought_status_param, drought_status_params_models in threshold_drought_status_params.items():
            metrics[policy][threshold][drought_status_param] = {
                "failure_frequency_PT1": np.asarray(drought_status_params_models.recorders['failure_frequency_PT1'].values()),
                "reliability_PT1": np.asarray(drought_status_params_models.recorders['reliability_PT1'].values()),
                "failure_frequency_Ag": np.asarray(drought_status_params_models.recorders['failure_frequency_Ag'].values()),
                "reliability_Ag": np.asarray(drought_status_params_models.recorders['reliability_Ag'].values()),
                "deficit PT1": np.asarray(drought_status_params_models.recorders['deficit PT1'].values()),
                "deficit Ag": np.asarray(drought_status_params_models.recorders['deficit Ag'].values()),
                "Maximum Deficit PT1": np.asarray(drought_status_params_models.recorders['Maximum Deficit PT1'].values()),
                "Maximum Deficit Ag": np.asarray(drought_status_params_models.recorders['Maximum Deficit Ag'].values()),
                "TotalCost": np.asarray(drought_status_params_models.recorders['TotalCost'].values())
            }

#%% Get total urban and ag demand

data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
urban_demand = data["PT1"]
num_weeks = len(urban_demand)
total_urban_demand = urban_demand.sum()

agricultural_demand_profile = np.array([18.01638039, 17.94856465, 17.84307351, 17.8053981, 17.7225122, 17.74511745,
                                        17.76772269, 17.77525777, 17.78279286, 17.91088925, 17.97870498, 18.15201186])
total_ag_demand = np.mean(agricultural_demand_profile) * num_weeks  # technically slightly off, but very close


#%% plot failure frequency for each scenario
# print(metrics['contracts_only'][-0.84]['ff_PT1'])
for drought_status_param in drought_status_params:
    plt.plot(metrics['contracts_only'][-0.84][drought_status_param]['failure_frequency_PT1'])

    plt.show()


#%% Parallel coordinates plot for single policy/indicator pair

results_list = {}
for policy, policy_thresholds in metrics.items():
    results_list[policy] = {}
    for threshold, threshold_drought_status_params in policy_thresholds.items():
        results_list[policy][threshold] = {}
        cur_results = {}
        for drought_status_param, drought_status_metrics in threshold_drought_status_params.items():
            # print(drought_status_metrics.keys())
            cur_results[drought_status_param] = {}
            for scenario in range(1, 16):
                cur_results[drought_status_param][scenario] = drought_status_metrics['failure_frequency_Ag'][scenario - 1]

        # cur_results = pd.DataFrame.from_dict(drought_status_metrics, orient='index')
        # cur_results['scenario'] = np.arange(1, len(cur_results.index) + 1)
        cur_results = pd.DataFrame.from_dict(cur_results, orient='index')
        cur_results = cur_results.T
        cur_results['scenario'] = cur_results.index
        print(cur_results)
        # cur_results[['rad_forcing', 'precip', 'temp']] = cur_results.apply(
        #     lambda row: parse_scenario_title(scenarios[round(row.scenario) - 1]), axis='columns', result_type='expand'
        # )
        print(cur_results)
        results_list[policy][threshold] = cur_results


def drought_status_par_coords(policy, threshold, color_var='scenario'):
    cur_results = results_list[policy][threshold]
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=cur_results[color_var],
                      showscale=True),  # COLORSCALE CAN BE ADDED OPTIONALLY
            dimensions=list([
                dict(
                    range=[0, 0.3],
                    label="drought_status_single_day",
                    values=cur_results['drought_status_single_day']
                ),
                dict(
                    range=[0, 0.3],
                    label="drought_status_min_recent",
                    values=cur_results['drought_status_min_recent']
                ),
                dict(
                    range=[0, 0.3],
                    label="drought_status_second_min_recent",
                    values=cur_results['drought_status_second_min_recent']
                ),
                dict(
                    range=[0, 0.3],
                    label="drought_status_threshold_passes",
                    values=cur_results['drought_status_threshold_passes']
                ),
                dict(
                    range=[0, 0.3],
                    label="drought_status_second_min_recent",
                    values=cur_results['drought_status_second_min_recent']
                ),
            ])
        )
    )

    fig.update_layout(title="Metrics for policy {}, threshold {}"
                      .format(policy, threshold))

    fig.show(renderer="browser")

drought_status_par_coords('contracts_only', -0.84)



# #%% Preliminary plots of metrics
#
# func_list = {
#     'mean': np.mean,
#     'max': np.max,
#     'min': np.min
# }
#
#
# def plot_metric(policy, metric):
#     recorders = []
#     agg_func = models[0][policy].recorders[metric].agg_func
#     func = func_list[agg_func]
#     for threshold in thresholds:
#         recorders.append(func(models[threshold][policy].recorders[metric].values()))
#     #ff_PT1_recorders = [models[threshold]['contracts_only'].recorders['failure_frequency_PT1'].agg_func for threshold in thresholds]
#     # print(thresholds)
#     # print(recorders)
#     if metric == 'deficit PT1':
#         recorders /= total_urban_demand
#     if metric == 'deficit Ag':
#         recorders /= total_ag_demand
#     plt.plot(thresholds, recorders)
#     plt.title('{} for {}'.format(metric, policy))
#     plt.ylim(0, 1.2 * np.max(recorders))
#     plt.show()
#
#
# policy_list = ['contracts_only', 'demand_only', "contract_and_demand", 'no_policy']
# metric_list = ['failure_frequency_PT1', 'failure_frequency_Ag',
#                'Maximum Deficit PT1', 'Maximum Deficit Ag',
#                'deficit PT1', 'deficit Ag']
#
# for metric in metric_list:
#     for policy in policy_list:
#         plot_metric(policy, metric)
