'''
Purpose: To initiate the optimization process with BORG, which will iteratively call the Pywr simulation model.
# documentation: https://waterprogramming.wordpress.com/2017/03/06/using-borg-in-parallel-and-serial-with-a-python-wrapper/
'''

import sys
import platform
if platform.system() == 'Windows':
    sys.path.append("C:\\users\\danny\\Pywr projects")  # Windows path
else:
    sys.path.append("/home/danny/FletcherLab/maipo-basin-pywr")  # Linux path

# general package imports
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal, percentileofscore

# pywr imports
from pywr.core import *
from pywr.parameters import *
from pywr.parameters._thresholds import StorageThresholdParameter, ParameterThresholdParameter
from pywr.recorders import DeficitFrequencyNodeRecorder, TotalDeficitNodeRecorder, MeanFlowNodeRecorder, \
    NumpyArrayParameterRecorder, NumpyArrayStorageRecorder, NumpyArrayNodeRecorder, RollingMeanFlowNodeRecorder, \
    AggregatedRecorder
from pywr.dataframe_tools import *

# internal imports
from MAIPO_PYWR.dps_BORG.borg import *
from MAIPO_PYWR.MAIPO_parameters import *
import MAIPO_PYWR.dps_BORG.example_sim_opt_238_test as example_sim_opt_238_test  # Import main optimization module that uses borg python wrapper


#%% Run MLE and Q-learning policies

def run_policy(policy_file):
    model = example_sim_opt_238_test.make_model(
        policy_file=policy_file
    )
    model.check()
    model.check_graph()
    model.find_orphaned_parameters()
    model.run()
    return model


mle_model = run_policy("A MLEs.txt")
q_model = run_policy("A q-learning.txt")


#%% Get metrics

extra_data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
training_results_df = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\preprocessed_training_data.csv")

# Get metrics for model
def get_metrics(model):
    metrics = {
        "Reliability PT1": np.asarray(model.recorders['reliability_PT1'].values()),
        "Reliability Ag": np.asarray(model.recorders['reliability_Ag'].values()),
        "Deficit PT1": np.asarray(model.recorders['deficit PT1'].values()),
        "Deficit Ag": np.asarray(model.recorders['deficit Ag'].values()),
        "Max deficit PT1": np.asarray(model.recorders['Maximum Deficit PT1'].values()),
        "Max deficit Ag": np.asarray(model.recorders['Maximum Deficit Ag'].values()),
        "Daily flow PT1": np.asarray(model.recorders['PT1 flow'].values()),
        "Daily flow Ag": np.asarray(model.recorders['Agriculture flow'].values()),
        "Daily cost": np.asarray(model.recorders['ContractCost'].values())
    }

    urban_demand = np.asarray(extra_data["PT1"])
    urban_flow = metrics["Daily flow PT1"]
    ag_demand = 16.978
    ag_flow = metrics["Daily flow Ag"]

    csv_length = len(urban_demand)
    simulation_length = len(urban_flow)
    relevant_urban_demand = urban_demand[csv_length - simulation_length:].reshape(simulation_length, 1)

    urban_deficit = relevant_urban_demand - urban_flow
    urban_deficit_proportion = urban_deficit / relevant_urban_demand
    ag_deficit = ag_demand - ag_flow
    ag_deficit_proportion = ag_deficit / ag_demand
    penalty = urban_deficit_proportion**2 + ag_deficit_proportion**2

    metrics["Daily deficit PT1"] = urban_deficit
    metrics["Daily deficit proportion PT1"] = urban_deficit_proportion
    metrics["Daily deficit Ag"] = ag_deficit
    metrics["Daily deficit proportion Ag"] = ag_deficit_proportion
    metrics["Deficit penalty"] = penalty

    avg_cost = np.mean(training_results_df["Cost"])
    avg_deficit_penalty = np.mean(training_results_df["Deficit penalty"])
    cost_weighting = 1e-1 / avg_cost  # average cost should be smaller than deficit penalty
    deficit_penalty_weighting = 1 / avg_deficit_penalty

    metrics["Reward"] = -(
            cost_weighting * metrics["Daily cost"] +
            deficit_penalty_weighting * metrics["Deficit penalty"]
    )


#%% Analyze data with plots and graphs
training_scenarios = [1, 12, 8, 9, 10, 4, 6, 13, 3, 14, 2, 15]
testing_scenarios = [5, 7, 11]