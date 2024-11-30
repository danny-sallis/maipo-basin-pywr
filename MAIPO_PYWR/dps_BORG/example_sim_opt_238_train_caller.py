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
import MAIPO_PYWR.dps_BORG.example_sim_opt_238_train as example_sim_opt_238_train  # Import main optimization module that uses borg python wrapper

#%% Tests of run (make sure example_sim_opt_238_train is working)

num_k = 1  # number of levels in policy tree
num_DP = 7  # number of decision periods
model = example_sim_opt_238_train.make_model(
    contracts_purchased=100,
    Embalse_fixed_flow=10
)

model.check()
model.check_graph()
model.find_orphaned_parameters()
model.run()

print(np.asarray(model.recorders['failure_frequency_PT1'].values()))
print(np.asarray(model.recorders['failure_frequency_Ag'].values()))
print(np.asarray(model.recorders['Embalse_fixed_drain flow'].data))

plt.plot(np.asarray(model.recorders['Embalse_fixed_drain flow'].data)[:, 0])
plt.show()
#
# #%% Tests for processing simulation results and data
#
# data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
#
# urban_demand = np.asarray(data["PT1"])
# urban_flow = np.asarray(model.recorders['PT1 flow'].data)
# ag_demand = 16.978
# ag_flow = np.asarray(model.recorders['Agriculture flow'].data)
#
# csv_length = len(urban_demand)
# simulation_length = len(urban_flow)
# print(csv_length, simulation_length)
#
# relevant_urban_demand = urban_demand[csv_length - simulation_length:].reshape(simulation_length, 1)
# urban_deficit = relevant_urban_demand - urban_flow
# urban_deficit_proportion = urban_deficit / relevant_urban_demand
# # print(urban_deficit_proportion)
# ag_deficit = ag_demand - ag_flow
# ag_deficit_proportion = ag_deficit / ag_demand
# penalty = urban_deficit_proportion**2 + ag_deficit_proportion**2
# # print(penalty)
# # print(ag_deficit_proportion)
#
# # print(model.recorders["RollingMeanFlowElManzano"].data.shape)
# print(len(model.recorders["ContractCost"].values()[0, :]))
# plt.plot(model.recorders["ContractCost"].values().T)
# plt.show()


#%% Create dataframe from results of simulation with each reservoir/contract pair, save to CSV

results_columns = [
    "Scenario",  # Permanent, unrelated to state (hopefully)

    # Info about current state (seen before buying; week number, current storage, inflow and indicator previous week)
    "Week of year",  # Deterministic transitions
    "Storage",
    "Inflow",
    "Indicator",

    # Info about next state
    "Next week of year",
    "Next storage",
    "Next inflow",
    "Next indicator",

    # Info for reward
    "Cost",
    "Urban demand",
    "Urban flow",
    "Agricultural demand",
    "Agricultural flow",

    # Action
    "Contracts purchased"
]

contract_val_list = 100*np.arange(11)
print(contract_val_list)
reservoir_val_list = 15 + 10*np.arange(10)
print(reservoir_val_list)
SRI3 = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\SRI3.csv")
extra_data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
PT1_demand = np.asarray(extra_data['PT1'])
Ag_demand = 16.978

results_list = []
for contract_val in contract_val_list:
    for reservoir_val in reservoir_val_list:
        model = example_sim_opt_238_train.make_model(
            contracts_purchased=contract_val,
            Embalse_fixed_flow=reservoir_val - 15
        )

        model.check()
        model.check_graph()
        model.find_orphaned_parameters()
        model.run()

        next_storages = np.asarray(model.recorders["Embalse_fixed_drain flow"].data)
        urban_flow = np.asarray(model.recorders["PT1 flow"].data)
        ag_flow = np.asarray(model.recorders["Agriculture flow"].data)
        costs = np.asarray(model.recorders["ContractCost"].values().T)
        total_inflows = np.asarray(model.recorders["Total catchment inflow"].data)

        for scenario in range(15):
            for timestep_idx in range(1, 1560):
                results_list.append([
                    scenario,

                    # Current state
                    timestep_idx % 52 + 1,
                    reservoir_val,  # Storage before contracts are chosen and model is run
                    total_inflows[timestep_idx - 1][scenario],  # Only know previous inflow and indicator
                    SRI3.iloc[timestep_idx - 1, scenario + 1],

                    # Next state
                    (timestep_idx + 1) % 52 + 1,
                    next_storages[timestep_idx][scenario],  # Storage after running for the week
                    total_inflows[timestep_idx][scenario],  # Inflow and indicator seen after running for the week
                    SRI3.iloc[timestep_idx, scenario + 1],

                    # Information for rewards
                    costs[timestep_idx][scenario],
                    PT1_demand[timestep_idx],
                    urban_flow[timestep_idx][scenario],
                    Ag_demand,
                    ag_flow[timestep_idx][scenario],

                    # Action
                    contract_val
                ])

results_df = pd.DataFrame(data=results_list, columns=results_columns)

results_df.to_csv("offline_training_data.csv")


#%% Read in data from CSV (so we don't have to recalculate each time)

# results_df = pd.read_csv('offline_training_data.csv')


#%% Add cols of column value as a percentile

def get_percentiles(col):
    sorted_col = np.sort(col)
    return np.searchsorted(sorted_col, col) / len(col)


state_cols = [
    # Info about current state
    "Week of year",  # Deterministic transitions
    "Storage",
    "Inflow",
    "Indicator",

    # Info about next state
    "Next week of year",
    "Next storage",
    "Next inflow",
    "Next indicator"
]
for col in state_cols:
    results_df["{} percentiles".format(col)] = get_percentiles(results_df[col])


#%% Add cols of deficits and square deficit function

extra_data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")

results_df["Urban deficit"] = (results_df["Urban demand"] - results_df["Urban flow"])
results_df["Urban deficit proportion"] = results_df["Urban deficit"] / results_df["Urban demand"]
results_df["Agricultural deficit"] = results_df["Agricultural demand"] - results_df["Agricultural flow"]
results_df["Agricultural deficit proportion"] = results_df["Agricultural deficit"] / results_df["Agricultural demand"]
results_df["Deficit penalty"] = (results_df["Urban deficit proportion"]**2
                                 + results_df["Agricultural deficit proportion"]**2)

results_df.head()


#%% Get average positive cost and positive deficit penalty for weighting rewards

avg_cost = np.mean(results_df.loc[results_df["Cost"] > 0, "Cost"])
print(avg_cost)  # 4424.150680679778

avg_deficit_penalty = np.mean(results_df.loc[results_df["Deficit penalty"] > 0, "Deficit penalty"])
print(avg_deficit_penalty)  # 0.013429669306695678


# Combine cost and deficit penalty into single reward
cost_weighting = 1e-1 / avg_cost  # average cost should be smaller than deficit penalty
deficit_penalty_weighting = 1 / avg_deficit_penalty

results_df["Reward"] = -cost_weighting * results_df["Cost"] - deficit_penalty_weighting * results_df["Deficit penalty"]


#%% Get indices of percentiles for each current and next state

percentile_cols = ["{} percentiles".format(col_name) for col_name in state_cols]
for col in percentile_cols:
    results_df["{} index".format(col)] = (10 * results_df[col]).astype(int)

results_df["Contracts purchased index"] = results_df["Contracts purchased"] // 100

print(results_df.head())

#%% Save new columns in new csv to reference later

results_df.to_csv("preprocessed_training_data.csv")


#%% Read in CSV of preprocessed data (to avoid having to restart the whole process)

# results_df = pd.read_csv("preprocessed_training_data.csv")


#%% Choose scenarios to train and test on

np.random.seed(96)
all_scenarios = np.array(range(1, 16))
training_scenarios = np.random.choice(all_scenarios, size=12, replace=False)
print(training_scenarios)
testing_scenarios = np.array([x for x in all_scenarios if x not in training_scenarios])
print(testing_scenarios)

results_df = results_df.loc[results_df["Scenario"].isin(training_scenarios)]


#%% Define Gaussian kernel smoothing pdf

# cov 0.3 makes neighbors 1 away have weight around 1/5 of the true state
gaussian = multivariate_normal(mean=np.zeros(4), cov=np.diag(0.3 * np.ones(4)))

#%% Learn optimal policy with MLEs

# Get percentile of weeks apart in shortest direction
def week_distance(week1, week2):
    max_week = np.max(week1, week2)
    min_week = np.min(week1, week2)

    return np.min(max_week - min_week, 1 + min_week - max_week)

# Get weights of each data point with respect to a state
def gaussian_weight(week_of_year, storage, inflow, indicator, dist, data):
    week_of_year_distances = np.array(np.abs(week_of_year - data["Week of year percentiles"]))
    storage_distances = np.array(np.abs(storage - data["Storage percentiles"]))
    inflow_distances = np.array(np.abs(inflow - data["Inflow percentiles"]))
    indicator_distances = np.array(np.abs(np.floor(norm.cdf(indicator) - norm.cdf(data["Indicator percentiles"]))))

    to_return = dist.pdf([(
            week_of_year_distances[i],
            storage_distances[i],
            inflow_distances[i],
            indicator_distances[i]
        ) for i in range(len(inflow_distances))
    ])
    if np.isscalar(to_return):
        to_return = np.array([to_return])

    return to_return


# Get reward for each state based on Gaussian smoothing with neighboring states
# Note: We weight each row by the state's distance. Rows that appear multiple times
# are weighted more highly. We're only considering close-by points though, so this
# probably isn't a big issue
def reward(
        week_of_year_percentile,
        storage_percentile,
        inflow_percentile,
        indicator_percentile,
        data
):
    weights = gaussian_weight(
        week_of_year_percentile,
        storage_percentile,
        inflow_percentile,
        indicator_percentile,
        gaussian,
        data
    )

    return np.average(data["Reward"], weights=weights)


# Apply value iteration to get optimal actions at each point, with transitions
# and rewards defined by Gaussian-smoothed MLEs. We don't explicitly calculate
# the transitions since the table is very big and hard to store; we calculate
# them online each round (sacrificing lots of speed for lower memory usage)
def get_MLE_utilities_and_actions(discount_factor, data, num_iters=50):
    Q = np.zeros((11, 11, 11, 11, 11))  # Utility for each state/action pair
    U = np.zeros((11, 11, 11, 11))  # best utility for each state pair
    A = -np.ones((11, 11, 11, 11))  # best action for each state
    for iteration in range(num_iters):
        print(iteration)  # Signal how far we are through the run
        for week_of_year_percentile_idx in range(11):
            week_of_year_percentile = 0.1 * week_of_year_percentile_idx
            woy_data = data.loc[np.abs(data["Week of year percentiles"] - week_of_year_percentile) <= 0.1001]
            print("a")  # Signal how far we are through the run
            for storage_percentile_idx in range(11):
                storage_percentile = 0.1 * storage_percentile_idx
                stor_data = woy_data.loc[np.abs(woy_data["Storage percentiles"] - storage_percentile) <= 0.1001]
                for inflow_percentile_idx in range(11):
                    inflow_percentile = 0.1 * inflow_percentile_idx
                    infl_data = stor_data.loc[np.abs(stor_data["Inflow percentiles"] - inflow_percentile) <= 0.1001]
                    for indicator_percentile_idx in range(11):
                        indicator_percentile = 0.1 * indicator_percentile_idx
                        all_relevant_data = infl_data.loc[
                            np.abs(infl_data["Indicator percentiles"] - indicator_percentile) <= 0.1001
                        ]

                        # store best utility and action so far
                        best_utility = -np.infty
                        best_action = 0  # default to no contracts bought

                        # For now, ignore states where no neighbors seen
                        if len(all_relevant_data) == 0:
                            continue

                        for contracts_purchased in contract_val_list:
                            relevant_data = all_relevant_data.loc[
                                all_relevant_data["Contracts purchased"] == contracts_purchased
                            ]

                            # For now, ignore states where no neighbors seen
                            if len(relevant_data) == 0:
                                continue

                            # Reward from initial action
                            initial_reward = reward(
                                week_of_year_percentile,
                                storage_percentile,
                                inflow_percentile,
                                indicator_percentile,
                                relevant_data
                            )

                            # Get weights of all neighbors
                            weights = gaussian_weight(
                                week_of_year_percentile,
                                storage_percentile,
                                inflow_percentile,
                                indicator_percentile,
                                gaussian,
                                relevant_data
                            )

                            # Get indices of states neighbors transition to
                            state_indices = [
                                tuple(indices) for indices in relevant_data[[
                                    "Next week of year percentiles index",
                                    "Next storage percentiles index",
                                    "Next inflow percentiles index",
                                    "Next indicator percentiles index"
                                ]].values
                            ]
                            utilities = [U[state_index] for state_index in state_indices]
                            # Get value of Bellman update for this transition
                            bellman_value = initial_reward + discount_factor * np.average(utilities, weights=weights)

                            if bellman_value > best_utility:
                                best_utility = bellman_value
                                best_action = contracts_purchased // 100

                            Q[
                                week_of_year_percentile_idx,
                                storage_percentile_idx,
                                inflow_percentile_idx,
                                indicator_percentile_idx,
                                contracts_purchased // 100
                            ] = bellman_value

                        indices = (
                            week_of_year_percentile_idx,
                            storage_percentile_idx,
                            inflow_percentile_idx,
                            indicator_percentile_idx
                        )
                        U[indices] = best_utility
                        A[indices] = best_action

    np.savetxt("Q MLEs.txt", Q)
    np.savetxt("U MLEs.txt", U)
    np.savetxt("A MLEs.txt", A)
    return Q, U, A


get_MLE_utilities_and_actions(0.95, results_df, num_iters=100)


#%% Apply Q-learning to get utilities and actions

# Get neighbors of an index (-1 and +1, with cutoff at 0 and 10)
def neighbors(idx):
    return np.array(range(np.max(idx - 1, 0), 1 + np.min(idx + 1, 10)))


# Get neighbors of a state and their weights
def get_nearby_states_and_weights(week_of_year_idx, storage_idx, inflow_idx, indicator_idx):
    neighbors = []
    for week_of_year_neighbor in neighbors(week_of_year_idx):
        for storage_neighbor in neighbors(storage_idx):
            for inflow_neighbor in neighbors(inflow_idx):
                for indicator_neighbor in neighbors(indicator_idx):
                    state = (
                        week_of_year_neighbor,
                        storage_neighbor,
                        inflow_neighbor,
                        indicator_neighbor
                    )

                    weight = gaussian(
                        (week_of_year_idx - week_of_year_neighbor) / 10,
                        (storage_idx - storage_neighbor) / 10,
                        (inflow_idx - inflow_neighbor) / 10,
                        (indicator_idx - indicator_neighbor) / 10,
                    )

                    neighbors.append({
                        "state": state,
                        "weight": weight
                    })

    return neighbors

# Apply to Q-learning algorithm For each iteration, go through every row of data,
# with weight depending on Gaussian distance again
def q_learn_utilities_and_actions(discount_factor, data, num_iters=100):
    # multiplying alpha by Gaussian weights during training
    alpha = 0.04 / gaussian((0, 0, 0, 0))

    Q = np.zeros((11, 11, 11, 11, 11))  # Utility for each state/action pair
    U = np.zeros((11, 11, 11, 11))  # best utility for each state pair
    A = -np.ones((11, 11, 11, 11))  # best action for each state

    for idx, row in pd.iterrows(data):
        week_of_year_idx = row["Week of year percentiles index"]
        storage_idx = row["Storage percentiles index"]
        inflow_idx = row["Inflow percentiles index"]
        indicator_idx = row["Indicator percentiles index"]

        next_week_of_year_idx = row["Next week of year percentiles index"]
        next_storage_idx = row["Next storage percentiles index"]
        next_inflow_idx = row["Next inflow percentiles index"]
        next_indicator_idx = row["Next indicator percentiles index"]

        cur_state = (
            week_of_year_idx,
            storage_idx,
            inflow_idx,
            indicator_idx
        )

        cur_state_neighbors = get_nearby_states_and_weights(cur_state)

        next_state = (
            next_week_of_year_idx,
            next_storage_idx,
            next_inflow_idx,
            next_indicator_idx
        )

        next_state_neighbors = get_nearby_states_and_weights(next_state)

        for cur_state_neighbor in cur_state_neighbors:
            for next_state_neighbor in next_state_neighbors:
                cur_neighbor_state = cur_state_neighbor.state
                next_neighbor_state = next_state_neighbor.state
                cur_weight = cur_state_neighbor.weight
                next_weight = next_state_neighbor.weight
                Q[cur_neighbor_state] = (Q[cur_neighbor_state] + cur_weight * next_weight * alpha * (
                            row["Reward"] + discount_factor * np.max(Q[next_neighbor_state, :]) - Q[cur_neighbor_state]
                        ))

    # Now that we have optimal Q-values, find the optimal utilities and actions
    for week_of_year_idx in range(11):
        for storage_idx in range(11):
            for inflow_idx in range(11):
                for indicator_idx in range(11):
                    state = (
                        week_of_year_idx,
                        storage_idx,
                        inflow_idx,
                        indicator_idx
                    )
                    best_contract_idx = np.argmax(Q[state, :])
                    A[state] = best_contract_idx
                    U[state] = Q[state, best_contract_idx]

    np.savetxt("Q q-learning.txt", Q)
    np.savetxt("U q-learning.txt", U)
    np.savetxt("A q-learning.txt", A)
    return Q, U, A

