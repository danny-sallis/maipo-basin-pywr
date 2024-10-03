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

#%% Plot functions

num_scenarios = 15

scenarios = pd.read_csv('../data/COLORADO.csv').columns[1::]

# Can plot lines with different scenarios in different colors, and same scenario but
# different pp/temp can have different shades
def parse_scenario_title(scenario):
    tokens = scenario.split('_')
    rcp_val = tokens[1]
    pp = int(tokens[3])
    temp = int(tokens[5])

    return rcp_val, pp, temp

def get_label(scenario):
    rcp_val, pp, temp = parse_scenario_title(scenario)
    # REVISIT BELOW LINE WHEN I FIND OUT WHAT THESE MEAN
    return "RCP: {}, pp: {}, temp: {}".format(rcp_val, pp, temp)

def scenario_color(scenario):
    base_color = {
        "45": "Yellow",
        "60": "Orange",
        "85": "Red"
    }

    shade_ratio = {
        5: 1,
        25: 0.9,
        50: 0.8,
        75: 0.7,
        95: 0.6
    }

    rcp_val, pp, _ = parse_scenario_title(scenario)
    rgb_vals = colors.to_rgb(base_color[rcp_val])
    new_color = []
    for val in rgb_vals:
        new_color.append(val*shade_ratio[pp])
    return tuple(new_color)


# LINE CHART WITH ALL SCENARIOS ON ONE PLOT

def line_chart_all_scenarios(var, varying_vals, fixed_val):
    num_k = 1  # number of levels in policy tree
    num_DP = 7  # number of decision periods

    num_vals = len(varying_vals)
    ff_PT1 = np.zeros((num_vals, num_scenarios))
    ff_Ag = np.zeros((num_vals, num_scenarios))
    deficit_PT1 = np.zeros((num_vals, num_scenarios))
    deficit_Ag = np.zeros((num_vals, num_scenarios))
    cost = np.zeros((num_vals, num_scenarios))

    if var == "contract_threshold_vals":
        params = {
            "thresh": [varying_vals[i] * np.ones(num_DP) for i in range(num_vals)],
            "acts": fixed_val * np.ones(num_DP),
            "pairs": [(varying_vals[i] * np.ones(num_DP), fixed_val * np.ones(num_DP)) for i in range(num_vals)]
        }
    elif var == "contract_action_vals":
        params = {
            "thresh": fixed_val * np.ones(num_DP),
            "acts": [varying_vals[i] * np.ones(num_DP) for i in range(num_vals)],
            "pairs": [(fixed_val * np.ones(num_DP), varying_vals[i] * np.ones(num_DP)) for i in range(num_vals)]
        }
    elif var == "demand_threshold_vals":
        params = {
            "thresh": [varying_vals[i] * np.ones(12) for i in range(num_vals)],
            "acts": [np.ones(12), fixed_val * np.ones(12)],
            "pairs": [([varying_vals[i] * np.ones(12)], [np.ones(12), fixed_val * np.ones(12)]) for i in
                      range(num_vals)],
        }
    elif var == "demand_action_vals":
        params = {
            "thresh": [fixed_val * np.ones(12)],
            "acts": [np.ones(12), *[varying_vals[i] * np.ones(12) for i in range(num_vals)]],
            "pairs": [([fixed_val * np.ones(12)], [np.ones(12), *[varying_vals[i] * np.ones(12)]]) for i in
                      range(num_vals)]
        }
    else:
        raise Exception("Please enter one of: contract_threshold_vals, contract_action_vals, "
                        "demand_threshold_vals, demand_action_vals")

    thresh_list = params["thresh"]
    acts_list = params["acts"]
    pair_list = params["pairs"]
    for i in range(num_vals):
        thresh, acts = pair_list[i]
        if var == "contract_threshold_vals" or var == "contract_action_vals":
            m_ = make_model(contract_threshold_vals=thresh, contract_action_vals=acts)  # function in example_sim_opt.py
        elif var == "demand_threshold_vals" or var == "demand_action_vals":
            m_ = make_model(demand_threshold_vals=thresh, demand_action_vals=acts)  # function in example_sim_opt.py
        else:
            raise Exception("Please enter one of: contract_threshold_vals, contract_action_vals, "
                            "demand_threshold_vals, demand_action_vals")

        m_.check()
        m_.check_graph()
        m_.find_orphaned_parameters()

        m_.run()

        print(":)")
        ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
        ff_Ag[i, :] = m_.recorders['failure_frequency_Ag'].values()
        deficit_PT1[i, :] = m_.recorders['Maximum Deficit PT1'].values()
        deficit_Ag[i, :] = m_.recorders['Maximum Deficit Ag'].values()
        cost[i, :] = m_.recorders['TotalCost'].values()

    labels = {
        "contract_threshold_vals": "Contract threshold",
        "contract_action_vals": "Contracts purchased",
        "demand_threshold_vals": "Demand restriction thresholds",
        "demand_action_vals": "Demand fraction required"
    }

    ff_PT1_fig, ff_PT1_ax = plt.subplots(figsize=(15, 12))
    ff_PT1_ax.set_xticks(varying_vals)
    ff_PT1_ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
    ff_PT1_ax.set_xlabel(labels[var])

    ff_Ag_fig, ff_Ag_ax = plt.subplots(figsize=(15, 12))
    ff_Ag_ax.set_xticks(varying_vals)
    ff_Ag_ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
    ff_Ag_ax.set_xlabel(labels[var])

    deficit_PT1_fig, deficit_PT1_ax = plt.subplots(figsize=(15, 12))
    deficit_PT1_ax.set_xticks(varying_vals)
    deficit_PT1_ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
    deficit_PT1_ax.set_xlabel(labels[var])

    deficit_Ag_fig, deficit_Ag_ax = plt.subplots(figsize=(15, 12))
    deficit_Ag_ax.set_xticks(varying_vals)
    deficit_Ag_ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
    deficit_Ag_ax.set_xlabel(labels[var])

    cost_fig, cost_ax = plt.subplots(figsize=(15, 12))
    cost_ax.set_xticks(varying_vals)
    cost_ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
    cost_ax.set_xlabel(labels[var])

    max_ff = np.maximum(np.max(np.max(ff_PT1)), np.max(np.max(ff_Ag)))
    max_deficit = np.maximum(np.max(np.max(deficit_PT1)), np.max(np.max(deficit_Ag)))
    max_cost = np.max(np.max(cost))

    for i in range(num_scenarios):
        ax_list = [ff_PT1_ax, ff_Ag_ax, deficit_PT1_ax, deficit_Ag_ax, cost_ax]
        data_list = [ff_PT1[:, i], ff_Ag[:, i], deficit_PT1[:, i], deficit_Ag[:, i], cost[:, i]]

        for j in range(len(ax_list)):
            ax_list[j].plot(varying_vals, data_list[j], marker='o', label=get_label(scenarios[i]), color=scenario_color(scenarios[i]))

    ff_PT1_ax.legend()
    ff_PT1_fig.suptitle("PT1 Failure Frequency", fontsize=34)
    ff_PT1_fig.show()

    ff_Ag_ax.legend()
    ff_Ag_fig.suptitle("Ag Failure Frequency", fontsize=34)
    ff_Ag_fig.show()

    deficit_PT1_ax.legend()
    deficit_PT1_fig.suptitle("PT1 Max Deficit", fontsize=34)
    deficit_PT1_fig.show()

    deficit_Ag_ax.legend()
    deficit_Ag_fig.suptitle("Ag Max Deficit", fontsize=34)
    deficit_Ag_fig.show()

    cost_ax.legend()
    cost_fig.suptitle("Cost", fontsize=34)
    cost_fig.show()


num_vals = 10
line_chart_all_scenarios("contract_threshold_vals", -0.5*np.arange(num_vals), 30)


#%%

# LINE CHART OF THRESHOLDS FOR EACH SCENARIO

# Make line chart of each metric, varying one lever
# lever: name of variable being changed
# varying_vals: values of variable being changed
# fixed_val: fixed value of corresponding action (for thresholds) or threshold (for actions)
# def line_chart_per_scenario(var, varying_vals, fixed_val):
#     num_k = 1  # number of levels in policy tree
#     num_DP = 7  # number of decision periods
#
#     num_vals = len(varying_vals)
#     ff_PT1 = np.zeros((num_vals, num_scenarios))
#     ff_Ag = np.zeros((num_vals, num_scenarios))
#     deficit_PT1 = np.zeros((num_vals, num_scenarios))
#     deficit_Ag = np.zeros((num_vals, num_scenarios))
#     cost = np.zeros((num_vals, num_scenarios))
#
#     if var == "contract_threshold_vals":
#         params = {
#             "thresh": [varying_vals[i] * np.ones(num_DP) for i in range(num_vals)],
#             "acts": fixed_val * np.ones(num_DP),
#             "pairs": [(varying_vals[i] * np.ones(num_DP), fixed_val * np.ones(num_DP)) for i in range(num_vals)]
#         }
#     elif var == "contract_action_vals":
#         params = {
#             "thresh": fixed_val * np.ones(num_DP),
#             "acts": [varying_vals[i] * np.ones(num_DP) for i in range(num_vals)],
#             "pairs": [(fixed_val * np.ones(num_DP), varying_vals[i] * np.ones(num_DP)) for i in range(num_vals)]
#         }
#     elif var == "demand_threshold_vals":
#         params = {
#             "thresh": [varying_vals[i] * np.ones(12) for i in range(num_vals)],
#             "acts": [np.ones(12), fixed_val * np.ones(12)],
#             "pairs": [([varying_vals[i] * np.ones(12)], [np.ones(12), fixed_val * np.ones(12)]) for i in range(num_vals)],
#         }
#     elif var == "demand_action_vals":
#         params = {
#             "thresh": [fixed_val * np.ones(12)],
#             "acts": [*[varying_vals[i] * np.ones(12) for i in range(num_vals)]],
#             "pairs": [([fixed_val * np.ones(12)], [np.ones(12), varying_vals[i] * np.ones(12)]) for i in range(num_vals)]
#         }
#     else:
#         raise Exception("Please enter one of: contract_threshold_vals, contract_action_vals, "
#                         "demand_threshold_vals, demand_action_vals")
#
#
#     thresh_list = params["thresh"]
#     acts_list = params["acts"]
#     pair_list = params["pairs"]
#     for i in range(num_vals):
#         thresh, acts = pair_list[i]
#         if var == "contract_threshold_vals" or var == "contract_action_vals":
#             m_ = make_model(contract_threshold_vals=thresh, contract_action_vals=acts)  # function in example_sim_opt.py
#         elif var == "demand_threshold_vals" or var == "demand_action_vals":
#             m_ = make_model(demand_threshold_vals=thresh, demand_action_vals=acts)  # function in example_sim_opt.py
#         else:
#             raise Exception("Please enter one of: contract_threshold_vals, contract_action_vals, "
#                   "demand_threshold_vals, demand_action_vals")
#
#         m_.check()
#         m_.check_graph()
#         m_.find_orphaned_parameters()
#
#         m_.run()
#
#         ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#         ff_Ag[i, :] = m_.recorders['failure_frequency_Ag'].values()
#         deficit_PT1[i, :] = m_.recorders['Maximum Deficit PT1'].values()
#         deficit_Ag[i, :] = m_.recorders['Maximum Deficit Ag'].values()
#         cost[i, :] = m_.recorders['TotalCost'].values()
#
#     if num_scenarios == 15:
#         num_rows = 3
#         num_cols = 5
#     elif num_scenarios == 36:
#         num_rows = 6
#         num_cols = 6
#     else:
#         num_rows = 1
#         num_cols = num_scenarios
#
#     ff_fig, ff_ax = plt.subplots(num_rows, num_cols, figsize=(15, 12))
#     deficit_fig, deficit_ax = plt.subplots(num_rows, num_cols, figsize=(15, 12))
#     cost_fig, cost_ax = plt.subplots(num_rows, num_cols, figsize=(15, 12))
#
#     max_ff = np.maximum(np.max(np.max(ff_PT1)), np.max(np.max(ff_Ag)))
#     max_deficit = np.maximum(np.max(np.max(deficit_PT1)), np.max(np.max(deficit_Ag)))
#     max_cost = np.max(np.max(cost))
#
#     labels = {
#         "contract_threshold_vals": "Contract threshold",
#         "contract_action_vals": "Contracts purchased",
#         "demand_threshold_vals": "Demand restriction thresholds",
#         "demand_action_vals": "Demand fraction required"
#     }
#
#     metric_properties = {
#         "ff_PT1": {
#             "ax": ff_ax,
#             "data": ff_PT1,
#             "max_val": max_ff,
#             "label": "PT1 Failure Frequency",
#             "color": "Purple",
#             "x_label": labels[var]
#         },
#
#         "ff_Ag": {
#             "ax": ff_ax,
#             "data": ff_Ag,
#             "max_val": max_ff,
#             "label": "Agriculture Failure Frequency",
#             "color": "Green",
#             "x_label": labels[var]
#         },
#
#         "deficit_PT1": {
#             "ax": deficit_ax,
#             "data": deficit_PT1,
#             "max_val": max_deficit,
#             "label": "PT1 Deficit",
#             "color": "Purple",
#             "x_label": labels[var]
#         },
#
#         "deficit_Ag": {
#             "ax": deficit_ax,
#             "data": deficit_Ag,
#             "max_val": max_deficit,
#             "label": "Agriculture Deficit",
#             "color": "Green",
#             "x_label": labels[var]
#         },
#
#         "cost": {
#             "ax": cost_ax,
#             "data": cost,
#             "max_val": max_cost,
#             "label": "Cost",
#             "color": "Orange",
#             "x_label": labels[var]
#         }
#     }
#
#     def create_line_plot(ax, data, max_val, label, color, x_label):
#         ax.plot(varying_vals, data, marker='o', label=label, color=color)
#         ax.set_ylim(0, 1.2 * max_val)
#         ax.set_xticks(varying_vals)
#         ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
#         ax.set_xlabel(x_label)
#
#     for i in range(15):
#         for metric, properties in metric_properties.items():
#             create_line_plot(
#                 properties["ax"][i // 5, i % 5],
#                 properties["data"][:, i],
#                 properties["max_val"],
#                 properties["label"],
#                 properties["color"],
#                 properties["x_label"]
#             )
#
#     def format_plot(fig, label):
#         lines_labels = [cur_ax.get_legend_handles_labels() for cur_ax in fig.axes[:1]]
#         lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#         fig.legend(lines, labels)
#         fig.suptitle(label, fontsize=34)
#         fig.tight_layout()
#         fig.show()
#
#     format_plot(ff_fig, "Failure Frequency")
#     format_plot(deficit_fig, "Maximum Deficit")
#     format_plot(cost_fig, "Total Cost")


# FUNCTION BELOW NOT USED, BUT PROVIDES TEMPLATES

# def get_inputs(var, num_vals, spacing, fixed):
#     if var == "contract_threshold_vals":
#         return {
#             "var": "demand_action_vals",
#             "varying_vals": -spacing*np.arange(num_vals),
#             "fixed_val": fixed
#         }
#
#     if var == "contract_action_vals":
#         return {
#             "var": "demand_action_vals",
#             "varying_vals": spacing*np.arange(num_vals),
#             "fixed_val": fixed
#         }
#
#     if var == "demand_threshold_vals":
#         return {
#             "var": "demand_threshold_vals",
#             "varying_vals": -spacing*np.arange(num_vals),
#             "fixed_val": fixed
#         }
#
#     if var == "demand_action_vals":
#         return {
#             "var": "demand_action_vals",
#             "varying_vals": 1 - spacing*np.arange(num_vals + 1),
#             "fixed_val": fixed
#         }

# GENERATE LINE PLOTS FOR EACH SCENARIO AND A PLOT WITH LINES FOR EACH SCENARIO

# def all_line_charts(var, varying_vals, fixed_val):
#     num_k = 1  # number of levels in policy tree
#     num_DP = 7  # number of decision periods
#
#     num_vals = len(varying_vals)
#     ff_PT1 = np.zeros((num_vals, num_scenarios))
#     ff_Ag = np.zeros((num_vals, num_scenarios))
#     deficit_PT1 = np.zeros((num_vals, num_scenarios))
#     deficit_Ag = np.zeros((num_vals, num_scenarios))
#     cost = np.zeros((num_vals, num_scenarios))
#
#     if var == "contract_threshold_vals":
#         params = {
#             "thresh": [varying_vals[i] * np.ones(num_DP) for i in range(num_vals)],
#             "acts": fixed_val * np.ones(num_DP),
#             "pairs": [(varying_vals[i] * np.ones(num_DP), fixed_val * np.ones(num_DP)) for i in range(num_vals)]
#         }
#     elif var == "contract_action_vals":
#         params = {
#             "thresh": fixed_val * np.ones(num_DP),
#             "acts": [varying_vals[i] * np.ones(num_DP) for i in range(num_vals)],
#             "pairs": [(fixed_val * np.ones(num_DP), varying_vals[i] * np.ones(num_DP)) for i in range(num_vals)]
#         }
#     elif var == "demand_threshold_vals":
#         params = {
#             "thresh": [varying_vals[i] * np.ones(12) for i in range(num_vals)],
#             "acts": [np.ones(12), fixed_val * np.ones(12)],
#             "pairs": [([varying_vals[i] * np.ones(12)], [np.ones(12), fixed_val * np.ones(12)]) for i in range(num_vals)],
#         }
#     elif var == "demand_action_vals":
#         params = {
#             "thresh": [fixed_val * np.ones(12)],
#             "acts": [np.ones(12), *[varying_vals[i] * np.ones(12) for i in range(num_vals)]],
#             "pairs": [([fixed_val * np.ones(12)], [np.ones(12), *[varying_vals[i] * np.ones(12)]]) for i in range(num_vals)]
#         }
#     else:
#         raise Exception("Please enter one of: contract_threshold_vals, contract_action_vals, "
#                         "demand_threshold_vals, demand_action_vals")
#
#
#     thresh_list = params["thresh"]
#     acts_list = params["acts"]
#     pair_list = params["pairs"]
#     for i in range(num_vals):
#         thresh, acts = pair_list[i]
#         if var == "contract_threshold_vals" or var == "contract_action_vals":
#             m_ = make_model(contract_threshold_vals=thresh, contract_action_vals=acts)  # function in example_sim_opt.py
#         elif var == "demand_threshold_vals" or var == "demand_action_vals":
#             m_ = make_model(demand_threshold_vals=thresh, demand_action_vals=acts)  # function in example_sim_opt.py
#         else:
#             raise Exception("Please enter one of: contract_threshold_vals, contract_action_vals, "
#                   "demand_threshold_vals, demand_action_vals")
#
#         m_.check()
#         m_.check_graph()
#         m_.find_orphaned_parameters()
#
#         m_.run()
#
#         ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#         ff_Ag[i, :] = m_.recorders['failure_frequency_Ag'].values()
#         deficit_PT1[i, :] = m_.recorders['Maximum Deficit PT1'].values()
#         deficit_Ag[i, :] = m_.recorders['Maximum Deficit Ag'].values()
#         cost[i, :] = m_.recorders['TotalCost'].values()
#
#     if num_scenarios == 15:
#         num_rows = 3
#         num_cols = 5
#     elif num_scenarios == 36:
#         num_rows = 6
#         num_cols = 6
#     else:
#         num_rows = 1
#         num_cols = num_scenarios
#
#     ff_fig, ff_ax = plt.subplots(num_rows, num_cols, figsize=(15, 12))
#     deficit_fig, deficit_ax = plt.subplots(num_rows, num_cols, figsize=(15, 12))
#     cost_fig, cost_ax = plt.subplots(num_rows, num_cols, figsize=(15, 12))
#
#     max_ff = np.maximum(np.max(np.max(ff_PT1)), np.max(np.max(ff_Ag)))
#     max_deficit = np.maximum(np.max(np.max(deficit_PT1)), np.max(np.max(deficit_Ag)))
#     max_cost = np.max(np.max(cost))
#
#     labels = {
#         "contract_threshold_vals": "Contract threshold",
#         "contract_action_vals": "Contracts purchased",
#         "demand_threshold_vals": "Demand restriction thresholds",
#         "demand_action_vals": "Demand fraction required"
#     }
#
#     metric_properties = {
#         "ff_PT1": {
#             "ax": ff_ax,
#             "data": ff_PT1,
#             "max_val": max_ff,
#             "label": "PT1 Failure Frequency",
#             "color": "Purple",
#             "x_label": labels[var]
#         },
#
#         "ff_Ag": {
#             "ax": ff_ax,
#             "data": ff_Ag,
#             "max_val": max_ff,
#             "label": "Agriculture Failure Frequency",
#             "color": "Green",
#             "x_label": labels[var]
#         },
#
#         "deficit_PT1": {
#             "ax": deficit_ax,
#             "data": deficit_PT1,
#             "max_val": max_deficit,
#             "label": "PT1 Deficit",
#             "color": "Purple",
#             "x_label": labels[var]
#         },
#
#         "deficit_Ag": {
#             "ax": deficit_ax,
#             "data": deficit_Ag,
#             "max_val": max_deficit,
#             "label": "Agriculture Deficit",
#             "color": "Green",
#             "x_label": labels[var]
#         },
#
#         "cost": {
#             "ax": cost_ax,
#             "data": cost,
#             "max_val": max_cost,
#             "label": "Cost",
#             "color": "Orange",
#             "x_label": labels[var]
#         }
#     }
#
#     def create_line_plot(ax, data, max_val, label, color, x_label):
#         ax.plot(varying_vals, data, marker='o', label=label, color=color)
#         ax.set_ylim(0, 1.2 * max_val)
#         ax.set_xticks(varying_vals)
#         ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
#         ax.set_xlabel(x_label)
#
#     for i in range(num_scenarios):
#         for metric, properties in metric_properties.items():
#             create_line_plot(
#                 properties["ax"][i // num_cols, i % num_cols],
#                 properties["data"][:, i],
#                 properties["max_val"],
#                 properties["label"],
#                 properties["color"],
#                 properties["x_label"]
#             )
#
#     def format_plot(fig, label):
#         lines_labels = [cur_ax.get_legend_handles_labels() for cur_ax in fig.axes[:1]]
#         lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#         fig.legend(lines, labels)
#         fig.suptitle(label, fontsize=34)
#         fig.tight_layout()
#         fig.show()
#
#     format_plot(ff_fig, "Failure Frequency")
#     format_plot(deficit_fig, "Maximum Deficit")
#     format_plot(cost_fig, "Total Cost")
#
#
#     ff_PT1_fig, ff_PT1_ax = plt.subplots(figsize=(15, 12))
#     ff_PT1_ax.set_xticks(varying_vals)
#     ff_PT1_ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
#     ff_PT1_ax.set_xlabel(labels[var])
#
#     ff_Ag_fig, ff_Ag_ax = plt.subplots(figsize=(15, 12))
#     ff_Ag_ax.set_xticks(varying_vals)
#     ff_Ag_ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
#     ff_Ag_ax.set_xlabel(labels[var])
#
#     deficit_PT1_fig, deficit_PT1_ax = plt.subplots(figsize=(15, 12))
#     deficit_PT1_ax.set_xticks(varying_vals)
#     deficit_PT1_ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
#     deficit_PT1_ax.set_xlabel(labels[var])
#
#     deficit_Ag_fig, deficit_Ag_ax = plt.subplots(figsize=(15, 12))
#     deficit_Ag_ax.set_xticks(varying_vals)
#     deficit_Ag_ax.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
#     deficit_Ag_ax.set_xlabel(labels[var])
#
#     cost_fig_condensed, cost_ax_condensed = plt.subplots(figsize=(15, 12))
#     cost_ax_condensed.set_xticks(varying_vals)
#     cost_ax_condensed.set_xticklabels(varying_vals.round(decimals=2), rotation=45)
#     cost_ax_condensed.set_xlabel(labels[var])
#
#     for i in range(num_scenarios):
#         ff_PT1_ax.plot(varying_vals, ff_PT1[:, i], marker='o', color="purple")#, label="PT1, Scenario {}".format(i + 1))
#         ff_Ag_ax.plot(varying_vals, ff_Ag[:, i], marker='o', color="green")#, label="Ag, Scenario {}".format(i + 1))
#         deficit_PT1_ax.plot(varying_vals, deficit_PT1[:, i], marker='o', color="purple")#, label="PT1, Scenario {}".format(i + 1))
#         deficit_Ag_ax.plot(varying_vals, deficit_Ag[:, i], marker='o', color="green")#, label="Ag, Scenario {}".format(i + 1))
#         cost_ax_condensed.plot(varying_vals, cost[:, i], marker='o', color="orange")#, label="Scenario {}".format(i + 1))
#
#     # ff_PT1_ax.legend()
#     ff_PT1_fig.suptitle("PT1 Failure Frequency", fontsize=34)
#     ff_PT1_fig.show()
#
#     # ff_Ag_ax.legend()
#     ff_Ag_fig.suptitle("Ag Failure Frequency", fontsize=34)
#     ff_Ag_fig.show()
#
#     # deficit_PT1_ax.legend()
#     deficit_PT1_fig.suptitle("PT1 Max Deficit", fontsize=34)
#     deficit_PT1_fig.show()
#
#     # deficit_Ag_ax.legend()
#     deficit_Ag_fig.suptitle("Ag Max Deficit", fontsize=34)
#     deficit_Ag_fig.show()
#
#     # cost_ax.legend()
#     cost_fig_condensed.suptitle("Cost", fontsize=34)
#     cost_fig_condensed.show()


# num_vals = 1
# line_chart_per_scenario("contract_threshold_vals", -0.5*np.arange(num_vals), 30)
# line_chart_all_scenarios("contract_threshold_vals", -0.5*np.arange(num_vals), 30)
# line_chart_per_scenario("contract_action_vals", 30*np.arange(num_vals), -0.84)
# line_chart_all_scenarios("contract_action_vals", 30*np.arange(num_vals), -0.84)
# line_chart_per_scenario("demand_threshold_vals", -0.5*np.arange(num_vals), 0.9)
# line_chart_all_scenarios("demand_threshold_vals", -0.5*np.arange(num_vals), 0.9)
# line_chart_per_scenario("demand_action_vals", 1 - 0.05*np.arange(num_vals), -0.84)
# line_chart_all_scenarios("demand_action_vals", 1 - 0.05*np.arange(num_vals), -0.84)


# num_vals = 1
# all_line_charts("contract_threshold_vals", -0.5*np.arange(num_vals), 30)
# all_line_charts("contract_action_vals", 30*np.arange(num_vals), -0.84)
# all_line_charts("demand_threshold_vals", -0.5*np.arange(num_vals), 0.9)
# all_line_charts("demand_action_vals", 1 - 0.05*np.arange(num_vals), -0.84)


#%%

# HEAT MAPS OF METRICS IN EACH SCENARIO

def create_heat_maps(category, thresh_list, acts_list):
    num_k = 1  # number of levels in policy tree
    num_DP = 7  # number of decision periods

    num_thresh = len(thresh_list)
    num_acts = len(acts_list)
    ff_PT1 = np.zeros((num_thresh, num_acts, num_scenarios))
    ff_Ag = np.zeros((num_thresh, num_acts, num_scenarios))
    deficit_PT1 = np.zeros((num_thresh, num_acts, num_scenarios))
    deficit_Ag = np.zeros((num_thresh, num_acts, num_scenarios))
    # Note: Cost might not be affected because contracts and things are already bought?
    cost = np.zeros((num_thresh, num_acts, num_scenarios))
    for i in range(num_thresh):
        for j in range(num_acts):
            thresh = thresh_list[i]
            acts = acts_list[j]
            # thresh = -0.84*np.ones(num_DP)*i #[-0.5*np.ones(12)*i]
            # acts = 30*np.ones(num_DP)*(j + 1)#[np.ones(12), (1 - 0.1*j)*np.ones(12)]

            if category == "contract":
                m_ = make_model(contract_threshold_vals=thresh*np.ones(num_DP),
                                contract_action_vals=acts*np.ones(num_DP))  # function in example_sim_opt.py
            if category == "demand":
                m_ = make_model(demand_threshold_vals=[thresh*np.ones(12)],
                                demand_action_vals=[np.ones(12), acts*np.ones(12)])  # function in example_sim_opt.py
            m_.check()
            m_.check_graph()
            m_.find_orphaned_parameters()

            m_.run()

            ff_PT1[i, j, :] = m_.recorders['failure_frequency_PT1'].values()
            ff_Ag[i, j, :] = m_.recorders['failure_frequency_Ag'].values()
            deficit_PT1[i, j, :] = m_.recorders['Maximum Deficit PT1'].values()
            deficit_Ag[i, j, :] = m_.recorders['Maximum Deficit Ag'].values()
            cost[i, j, :] = m_.recorders['TotalCost'].values()

        if num_scenarios == 15:
            num_rows = 3
            num_cols = 5
        elif num_scenarios == 36:
            num_rows = 6
            num_cols = 6
        else:
            num_rows = 1
            num_cols = num_scenarios

        fig_ff_PT1, ax_ff_PT1 = plt.subplots(num_rows, num_cols, figsize=(15, 12))
        fig_ff_Ag, ax_ff_Ag = plt.subplots(num_rows, num_cols, figsize=(15, 12))
        fig_deficit_PT1, ax_deficit_PT1 = plt.subplots(num_rows, num_cols, figsize=(15, 12))
        fig_deficit_Ag, ax_deficit_Ag = plt.subplots(num_rows, num_cols, figsize=(15, 12))
        fig_cost, ax_cost = plt.subplots(num_rows, num_cols, figsize=(15, 12))
        max_ff_PT1 = np.max(np.max(np.max(ff_PT1)))
        max_ff_Ag = np.max(np.max(np.max(ff_Ag)))
        max_deficit_PT1 = np.max(np.max(np.max(deficit_PT1)))
        max_deficit_Ag = np.max(np.max(np.max(deficit_Ag)))
        max_cost = np.max(np.max(np.max(cost)))

        x = np.arange(num_thresh)
        y = np.arange(num_acts)
        X, Y = np.meshgrid(x, y)

        def create_heat_map(fig, ax, data, max_val, color):
            img = ax.imshow(data, cmap=color, vmin=0, vmax=max_val,
                                                extent=[x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5],
                                                interpolation='nearest', origin='lower')
            ax.set_xticks(np.arange(num_acts))
            ax.set_xticklabels((30 * np.arange(1, num_acts + 1)).round(decimals=2))#1 - 0.1 * np.arange(num_acts), rotation=45)
            ax.set_yticks(np.arange(num_thresh))
            ax.set_yticklabels((-0.84 * np.arange(num_thresh)).round(decimals=2))#-0.5 * np.arange(num_thresh))
            # if category == "contracts":
            #     ax.set_xlabel("Contracts purchased")
            #     ax.set_ylabel("Contract thresholds")
            # if category == "demand":
            #     ax.set_xlabel("Demand fraction required")
            #     ax.set_ylabel("Demand restriction thresholds")

            fig.colorbar(img, ax=ax)

        metric_properties = {
            "ff_PT1": {
                "fig": fig_ff_PT1,
                "ax": ax_ff_PT1,
                "data": ff_PT1,
                "color": "Purples",
                "max_val": max_ff_PT1
            },

            "ff_Ag": {
                "fig": fig_ff_Ag,
                "ax": ax_ff_Ag,
                "data": ff_Ag,
                "color": "Greens",
                "max_val": max_ff_Ag
            },

            "deficit_PT1": {
                "fig": fig_deficit_PT1,
                "ax": ax_deficit_PT1,
                "data": deficit_PT1,
                "color": "Purples",
                "max_val": max_deficit_PT1
            },

            "deficit_Ag": {
                "fig": fig_deficit_Ag,
                "ax": ax_deficit_Ag,
                "data": deficit_Ag,
                "color": "Greens",
                "max_val": max_deficit_Ag
            },

            "cost": {
                "fig": fig_cost,
                "ax": ax_cost,
                "data": cost,
                "color": "Oranges",
                "max_val": max_cost
            }
        }

        for i in range(num_scenarios):
            for metric, properties in metric_properties.items():
                create_heat_map(
                    properties["fig"],
                    properties["ax"][i // 5, i % 5],
                    properties["data"][:, :, i],
                    properties["max_val"],
                    properties["color"]
                )


        fig_ff_PT1.suptitle("PT1 Failure Frequency ({})".format(category), fontsize=34)
        fig_ff_PT1.tight_layout()
        fig_ff_PT1.show()
        fig_ff_Ag.suptitle("Ag Failure Frequency ({})".format(category), fontsize=34)
        fig_ff_Ag.tight_layout()
        fig_ff_Ag.show()
        fig_deficit_PT1.suptitle("PT1 Max Deficit ({})".format(category), fontsize=34)
        fig_deficit_PT1.tight_layout()
        fig_deficit_PT1.show()
        fig_deficit_Ag.suptitle("Ag Max Deficit ({})".format(category), fontsize=34)
        fig_deficit_Ag.tight_layout()
        fig_deficit_Ag.show()
        fig_cost.suptitle("Total Cost ({})".format(category), fontsize=34)
        fig_cost.tight_layout()
        fig_cost.show()

# num_thresh = 5
# num_acts = 5
# create_heat_maps("contract", -0.84*np.arange(num_thresh), 30*np.arange(num_acts))
# create_heat_maps("demand", -0.84*np.arange(num_thresh), 1 - 0.1*np.arange(num_acts))
