'''
This file is not to be run directly. We will store code here that was previously run to generate
plots, test hypotheses, etc
'''

#%% run/simulate the Pywr model specified via the JSON document
#
# warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
#
# OUTPUT_DIR = "../outputs/DPS_results/SRI3/Policy0/2040_2050"
#
# os.makedirs(OUTPUT_DIR, exist_ok=True)
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
# m = Model.load("sc_MAIPO_DPS_sim.json")
# m.run()
#
# # comment out if you do not want to save results as file
# #TablesRecorder(m, OUTPUT_DIR + "/flows.h5", filter_kwds={"complib": "zlib", "complevel": 5 }, parameters= ["contract_value", "DP_index", "purchases_value", "Maipo_max_volume"])
#
# weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
# CustomizedAggregation(m, weights)
#
# # print some sample simulated results
# print('\nTotalCost:')
# print(m.recorders['TotalCost'].aggregated_value())
# print(np.array([m.recorders['TotalCost'].values()[i] for i in range(15)]))
# print('\nfailure_frequency_PT1:')
# print(m.recorders['failure_frequency_PT1'].aggregated_value())
# print(np.array([m.recorders['failure_frequency_PT1'].values()[i] for i in range(15)]))
# print('\nMaximum Deficit:')
# print(m.recorders['Maximum Deficit'].values())
# print('\nRollingMeanFlowElManzano:')
# print(m.recorders['RollingMeanFlowElManzano'].data.shape)  # WHY 1560X15?
# print(m.recorders['RollingMeanFlowElManzano'].data)

#%%
#%% run/simulate the same Pywr model, now specified via the Python API (function make_model()

# num_k = 1  # number of levels in policy tree
# num_DP = 7  # number of decision periods
#
# thresh = np.ones(num_DP) * -0.84
# acts = np.ones(num_DP) * 30
#
# m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
# m_.check()
# m_.check_graph()
# m_.find_orphaned_parameters()
#
# m_.run()

#%% Check for isomorphism of graphs in m and m_
# import networkx as nx
# import networkx.algorithms.isomorphism as iso
# em = iso.categorical_edge_match('weight', 'weight')
# nx.draw(m_.component_graph)
# plt.show()  # commented out so that we don't have to x it out to continue running
# pywr_model_to_d3_json(m_)

#%% Print results of model

# # print some sample simulated results
# print('\nTotalCost:')
# print(m_.recorders['TotalCost'].aggregated_value())
# print(np.array([m_.recorders['TotalCost'].values()[i] for i in range(15)]))
# print('\nfailure_frequency_PT1:')
# print(m_.recorders['failure_frequency_PT1'].aggregated_value())
# print(np.array([m_.recorders['failure_frequency_PT1'].values()[i] for i in range(15)]))
# print('\nMaximum Deficit PT1:')
# print(m_.recorders['Maximum Deficit PT1'].values())
# print('\nRollingMeanFlowElManzano:')
# print(m_.recorders['RollingMeanFlowElManzano'].data)


#%% Check that both models yield same results

# print('\nTotalCost:')
# print(m_.recorders['TotalCost'].aggregated_value())
# print(m.recorders['TotalCost'].aggregated_value())
# print(np.alltrue(np.array([m_.recorders['TotalCost'].values()[i] for i in range(15)]) == np.array([m.recorders['TotalCost'].values()[i] for i in range(15)])))
#
# print('\nfailure_frequency_PT1:')
# print(m_.recorders['failure_frequency_PT1'].aggregated_value() == m.recorders['failure_frequency_PT1'].aggregated_value())
# print(np.alltrue(np.array([m_.recorders['failure_frequency_PT1'].values()[i] for i in range(15)]) == np.array([m.recorders['failure_frequency_PT1'].values()[i] for i in range(15)])))
#
# print('\nMaximum Deficit:')
# print(np.alltrue(m_.recorders['Maximum Deficit'].values() == m.recorders['Maximum Deficit'].values()))
#
# print('\nRollingMeanFlowElManzano:')
# print(np.alltrue(m_.recorders['RollingMeanFlowElManzano'].data == m.recorders['RollingMeanFlowElManzano'].data))


#%% Make graphs of results
#
# def plotProperty(values, label, node=None):
#     # ADD RANGES FOR EACH PROPERTY
#     # COST RANGE: 0-3E9
#     # FAILURE FREQUENCY RANGE: 0-0.08
#     # DEFICIT RANGE: 0-7
#
#     # fig, ax = plt.subplots()
#     plt.figure()
#     plt.bar(list(map(lambda x: str(x), range(1, 16))), values, label=label)
#     # if label == "Cost":
#     #     plt.ylim(0, 3e9)
#     # if label == "Failure Frequency":
#     #     plt.ylim(0, 0.4)
#     # if label == "Maximum Deficit":
#     #     plt.ylim(0, 8)
#     plt.xlabel('Scenario')
#     plt.ylabel(label)
#     if node is None:
#         plt.title(label)
#     else:
#         plt.title(label + " (" + node + ")")
#     plt.axhline(y=np.mean(values), color='red', linestyle='--', label="Avg " + str(label))
#     plt.legend()
#     plt.show()
#
#
# # Plot costs
# totalCosts = m_.recorders['TotalCost'].values()
# plotProperty(totalCosts, label="Cost")
#
# failureFrequencyPT1 = m_.recorders['failure_frequency_PT1'].values()
# plotProperty(failureFrequencyPT1, label="Failure Frequency", node="PT1")
#
# failureFrequencyAg = m_.recorders['failure_frequency_Ag'].values()
# plotProperty(failureFrequencyAg, label="Failure Frequency", node="Ag")
#
# maximumDeficitPT1 = m_.recorders['Maximum Deficit PT1'].values()
# plotProperty(maximumDeficitPT1, label="Maximum Deficit", node="PT1")
#
# maximumDeficitAg = m_.recorders['Maximum Deficit Ag'].values()
# plotProperty(maximumDeficitAg, label="Maximum Deficit", node="Ag")

#%% save results

# data = {}
# values = {}
# for r in m.recorders:
#
#     rdata = {
#         'class': r.__class__.__name__,
#     }
#
#     try:
#         rdata['value'] = r.aggregated_value()
#     except NotImplementedError:
#         pass
#
#     try:
#         rdata['node'] = r.node.name
#     except AttributeError:
#         pass
#
#     try:
#         values[r.name] = list(r.values())
#     except NotImplementedError:
#         pass
#
#     if len(rdata) > 1:
#         data[r.name] = rdata
#
# writer = pandas.ExcelWriter(OUTPUT_DIR + '/metrics.xlsx', engine='xlsxwriter')
# metrics = pandas.DataFrame(data).T
# metrics.to_csv(OUTPUT_DIR + '/metrics.csv')
# metrics.to_excel(writer, 'aggregated')
# scenario_values = pandas.DataFrame(values).T
# scenario_values.to_excel(writer, 'scenarios')
#
# # comment out if you do not want to save results as file
# # writer.save() - note: save has been depreciated. Use close() instead.
#
# writer.close()
#
#
# #%% Load the simulation data
#
# with tables.open_file(os.path.join(OUTPUT_DIR, 'flows.h5')) as h5:
#     tbl = h5.get_node('/time')
#     date_index = pandas.to_datetime({k: tbl.col(k) for k in ('year', 'month', 'day')})
#
#     data = {}
#     for ca in h5.walk_nodes('/', 'CArray'):
#         data[ca._v_name] = pandas.DataFrame(ca.read(), index=date_index, columns=m.scenarios.multiindex)
#
# df = pandas.concat(data, axis=1)
# nrows = len(df.columns.levels[0])
#
# # df.plot(subplots=True, )
# FLOW_UNITS = 'Mm^3/day'

#%% Graphs to validate model

# Graphs checking how ag flow compares to PT1 flow
# extra_data = pd.read_csv("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\Extra data.csv")
# # print((data["PT1"].mean() + data["PT2"].mean())*52)
# print("Wait for it...")
# demand_flow = extra_data["PT1"].to_list()
# PT1_flow = m_.recorders["PT1 flow"].data[:, 10]
# Ag_flow = m_.recorders["Agriculture flow"].data[:, 10]
# for i in range(len(extra_data)):
#     # print(demand_flow[i], true_flow[i])
#     if demand_flow[i] > PT1_flow[i]:
#         print(i, demand_flow[i] - PT1_flow[i], Ag_flow[i])

# Plot of storage over time
# embalse_storage = m_.recorders["Embalse storage"].data[:, 0]
# plt.plot(embalse_storage)
# plt.xlabel('time')
# plt.ylabel('storage')
# plt.ylim(0, 225)
# plt.axhline(y=np.mean(embalse_storage), color='red', linestyle='--', label="Avg storage")
# plt.title('Embalse storage')
# plt.legend()
# plt.show()
#
# embalse_maipo_storage = m_.recorders["Embalse Maipo storage"].data[:, 0]
# plt.plot(embalse_maipo_storage)
# plt.xlabel('time')
# plt.ylabel('storage')
# plt.ylim(0, 225)
# plt.axhline(y=np.mean(embalse_maipo_storage), color='red', linestyle='--', label="Avg storage")
# plt.title('Embalse Maipo storage')
# plt.legend()
# plt.show()

# Plot of water rights remaining for ag after giving to urban
# water_rights_left = m_.recorders["Remaining water rights per week"].data[:, 0]
# ag_demand = m_.recorders["Agricultural demand recorder"].data[:, 0]
# plt.plot(water_rights_left, label="Water rights")
# plt.plot(ag_demand, label="Agricultural demand")
# plt.xlabel('Time')
# plt.ylabel('Water rights')
# plt.ylim(0, 150)
# plt.axhline(y=np.mean(water_rights_left), color='red', linestyle='--', label="Avg water rights")
# plt.title('Remaining water rights')
# plt.legend()
# plt.show()

# Plot of ag flow over time
# ag_flow = m_.recorders["Agriculture flow"].data[:, 0]
# plt.plot(ag_flow)
# plt.xlabel('time')
# plt.ylabel('flow')
# plt.ylim(0, 20)
# plt.axhline(y=np.mean(ag_flow), color='red', linestyle='--', label="Avg flow")
# plt.title('Agriculture flow')
# plt.legend()
# plt.show()
