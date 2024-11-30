# general package imports

import sys
import platform  # helps identify directory locations on different types of OS
# sys.path.append("/home/danny/FletcherLab/maipo-basin-pywr")
if platform.system() == "Windows":
    sys.path.append("C:\\Users\\danny\\Pywr projects")
else:
    sys.path.append("/home/danny/FletcherLab/maipo-basin-pywr")

import numpy as np
import pandas as pd
import sys

# pywr imports
from pywr.core import *
from pywr.parameters import *

from pywr.parameters._thresholds import StorageThresholdParameter, ParameterThresholdParameter
from pywr.recorders import DeficitFrequencyNodeRecorder, TotalDeficitNodeRecorder, MeanFlowNodeRecorder, \
    NumpyArrayParameterRecorder, NumpyArrayStorageRecorder, NumpyArrayNodeRecorder, RollingMeanFlowNodeRecorder, \
    AggregatedRecorder
# from MAIPO_searcher import *
from pywr.dataframe_tools import *

from MAIPO_PYWR.dps_BORG.borg import *

from MAIPO_PYWR.MAIPO_parameters import *
# from MAIPO_PYWR.dps_BORG.MAIPO_DPS import *

import importlib
# BorgMOEA = __import__("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\dps_BORG\\BorgMOEA_master")
# BorgMOEA = importlib.import_module("C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\dps_BORG\\BorgMOEA_master")
# from MAIPO_PYWR.dps_BORG.BorgMOEA_master.plugins.Python.borg import borg as bg  # Import borg wrapper

num_k = 1  # number of levels in policy tree
num_DP = 7  # number of decision periods


def make_model(contracts_purchased,
               demand_threshold_vals=[], demand_action_vals=[np.ones(12)],
               indicator="SRI3", drought_status_agg="drought_status_single_day_using_agg",
               Embalse_fixed_flow=220):
    '''
    Purpose: Creates a Pywr model with the specified number and values for policy thresholds/actions. Intended for use with MOEAs.

    Args:
        threshold_vals: an array of policy thresholds for drought index
        action_vals: an array of policy actions corresponding to policy thresholds

    Returns:
        model: a Pywr Model object
    '''

    # set current working directory
    # os.chdir(os.path.abspath(os.path.dirname(__file__)))

    # create a Pywr model (including an empty network)
    model = Model()

    # create a dictionary object to keep track of key parameters and nodes
    paramIndex = {}
    recorderIndex = {}

    # METADATA
    model.metadata = {
        "title": "Maipo Basin Model",
        "description": "Simulation-only AGU schematic of the model in JSON format for simulated flow used in the WEAP. 15 climate change scenarios between 2020 and 2050. KW",
        "minimum_version": "0.1"
    }

    # TIME STEPPER
    model.timestepper = Timestepper(
        start=pd.to_datetime('2006-07-12'),  # start
        end=pd.to_datetime('2010-10-18'),  # end
        delta=datetime.timedelta(1)  # interval
    )

    # SCENARIOS
    num_scenarios = 15
    Scenario(model, name="climate change", size=num_scenarios)

    # REQUIRED NODES FOR PARAMETERS
    # DP_index
    datestr = ["2006-07-12", "2007-03-29", "2007-12-14", "2008-08-30", "2009-05-17", "2010-02-01", "2010-10-19"]
    FakeYearIndexParameter(
        model,
        name="DP_index",
        dates=[datetime.datetime.strptime(i, '%Y-%m-%d') for i in datestr],
        comment="convert a specific date to integer, from 0 to 6, depending on the 5-year development plan period 2020-2050"
    )
    paramIndex["DP_index"] = model.parameters.__len__() - 1

    # Embalse_fixed inputs constant flow at each time step for training purposes
    Embalse_fixed_flow = ConstantParameter(
        model,
        name="Embalse_fixed_flow",
        value=Embalse_fixed_flow,
        is_variable=False,
        lower_bounds=0,
        upper_bounds=220-15  # max capacity minus min capacity
    )
    paramIndex['Embalse_fixed_flow'] = model.parameters.__len__() - 1

    # # Embalse
    # # Embalse = Storage(
    # #     model,
    # #     name="Embalse",
    # #     min_volume=15,
    # #     max_volume=220,
    # #     initial_volume=220,
    # #     cost=-1000
    # # )
    # Embalse = Storage(
    #     model,
    #     name="Embalse",
    #     min_volume=15,
    #     max_volume=220,
    #     initial_volume=220,
    #     cost=-100000
    # )

    # Maipo_capacity
    ConstantParameter(
        model,
        name="Maipo_capacity",
        value=0,
        is_variable=False,
        lower_bounds=0,
        upper_bounds=300
    )
    paramIndex['Maipo_capacity'] = model.parameters.__len__() - 1

    # Maipo_current_capacity
    ConstantParameter(
        model,
        name="Maipo_current_capacity",
        value=0
    )
    paramIndex['Maipo_current_capacity'] = model.parameters.__len__() - 1

    # Maipo_construction_dp
    ConstantParameter(
        model,
        name="Maipo_construction_dp",
        value=7,
        is_variable=False,
        lower_bounds=2,
        upper_bounds=7,
        comment="Choose any dp from 2025 to 2045. 7 means never constructed"
    )
    paramIndex['Maipo_construction_dp'] = model.parameters.__len__() - 1

    # Maipo_constructed
    ParameterThresholdParameter(
        model,
        param=model.parameters["DP_index"],
        threshold=model.parameters["Maipo_construction_dp"],
        predicate="GE",  # JSON: ">="
        name="Maipo_constructed",
        comment="indicates if the reservoir is active in a specific DP period"
    )
    paramIndex['Maipo_constructed'] = model.parameters.__len__() - 1

    # Maipo_max_volume
    IndexedArrayParameter(
        model,
        index_parameter=model.parameters["Maipo_constructed"],
        params=[
            model.parameters["Maipo_current_capacity"],
            model.parameters["Maipo_capacity"]
        ],
        name="Maipo_max_volume"
    )
    paramIndex['Maipo_max_volume'] = model.parameters.__len__() - 1

    # Embalse_Maipo
    Embalse_Maipo = Storage(
        model,
        name="Embalse Maipo",
        min_volume=0,
        max_volume=model.parameters["Maipo_max_volume"],  # Was 240
        initial_volume=0.0,
        initial_volume_pc=0.0,
        cost=-800
    )

    # requisito_embalse_Maipo
    StorageThresholdParameter(
        model,
        storage=Embalse_Maipo,
        threshold=140,
        predicate="LT",
        values=[1, 0],
        name="requisito_embalse_Maipo"
    )
    paramIndex['requisito_embalse_Maipo'] = model.parameters.__len__() - 1

    # flujo_excedentes_Yeso
    StorageThresholdParameter(
        model,
        storage=Embalse_Maipo,
        threshold=220,
        predicate="LT",
        values=[60.48, 0],
        name="flujo_excedentes_Yeso"
    )
    paramIndex['flujo_excedentes_Yeso'] = model.parameters.__len__() - 1

    # El Manzano
    El_Manzano = River(
        model,
        name="El Manzano"
    )

    # PARAMETERS
    # multiple usable drought_status parameters:
    # drought_status_single_day (uses just the first day of april/october)
    if platform.system() == "Windows":
        data_url_base = "C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\"
    else:
        data_url_base = "/home/danny/FletcherLab/maipo-basin-pywr/MAIPO_PYWR/data/"

    df = {
        'url': '{}{}.csv'.format(data_url_base, indicator),  # 'data/SRI6.csv'
        "parse_dates": True,
        "index_col": "Timestamp",
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        name="drought_status_single_day",
        scenario=model.scenarios.scenarios[0],
    )
    paramIndex["drought_status_single_day"] = model.parameters.__len__() - 1
    model.parameters._objects[
        paramIndex["drought_status_single_day"]].name = "drought_status_single_day"  # add name to parameter

    # drought_status_min_recent (lowest drought status in past month)
    # WILL LIKELY HAVE TO BUILD FROM DATAFRAMEPARAMETER

    # drought_status_percentile_recent (percentile drought status in past month)
    # WILL LIKELY HAVE TO BUILD FROM DATAFRAMEPARAMETER

    # drought_status_threshold_passes (times a given threshold passed in recent time period)
    # WILL LIKELY HAVE TO BUILD FROM DATAFRAMEPARAMETER

    # drought_status_aggregation (aggregations of threshold in recent time period)
    # WILL LIKELY HAVE TO BUILD FROM DATAFRAMEPARAMETER
    # MOST GENERAL -- COULD TRY TO MAKE FIRST!
    df = {
        'url': '{}{}.csv'.format(data_url_base, indicator),  # 'data/SRI6.csv'
        "parse_dates": True,
        "index_col": "Timestamp",
        "dayfirst": True}
    DroughtStatusAggregationParameter(
        model,
        dataframe=read_dataframe(model, df),
        name="drought_status_single_day_using_agg",
        agg_func=lambda x: x[len(x) - 1],
        num_weeks=1,
        scenario=model.scenarios.scenarios[0]
    )
    paramIndex["drought_status_single_day_using_agg"] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex[
        "drought_status_single_day_using_agg"]].name = "drought_status_single_day_using_agg"  # add name to parameter

    # # april_threshold
    # april_thresholds = []
    # for i, k in enumerate(contract_threshold_vals):
    #     april_thresholds.append(
    #         ConstantParameter(model, name=f"april_threshold{i}", value=k, is_variable=False, upper_bounds=0))
    #     paramIndex[f"april_threshold{i}"] = model.parameters.__len__() - 1
    # IndexedArrayParameter(
    #     model,
    #     name="april_threshold",
    #     index_parameter=model.parameters["DP_index"],  # DP_index
    #     params=april_thresholds,
    #     comment="variable parameter that set the drought threshold for contracts in april"
    # )
    # paramIndex["april_threshold"] = model.parameters.__len__() - 1
    #
    # # october_threshold
    # october_thresholds = []
    # for i, k in enumerate(contract_threshold_vals):
    #     october_thresholds.append(
    #         ConstantParameter(model, name=f"october_threshold{i}", value=k, is_variable=False, upper_bounds=0))
    #     paramIndex[f"october_threshold{i}"] = model.parameters.__len__() - 1
    # IndexedArrayParameter(
    #     model,
    #     name="october_threshold",
    #     index_parameter=model.parameters["DP_index"],  # DP_index
    #     params=october_thresholds,
    #     comment="variable parameter that set the drought threshold for contracts in october"
    # )
    # paramIndex["october_threshold"] = model.parameters.__len__() - 1
    #
    #
    #
    #
    # # april_contract
    # april_contracts = []
    # for i, k in enumerate(contract_action_vals):
    #     april_contracts.append(
    #         ConstantParameter(model, name=f"april_contract{i}", value=k, is_variable=False, upper_bounds=1500))
    #     paramIndex[f"april_contract{i}"] = model.parameters.__len__() - 1
    # IndexedArrayParameter(
    #     model,
    #     name="april_contract",
    #     index_parameter=model.parameters["DP_index"],  # DP_index
    #     params=april_contracts,
    #     comment="variable parameter that set the contract shares in april for a determined dp"
    # )
    # paramIndex["april_contract"] = model.parameters.__len__() - 1
    #
    # # october_contract
    # october_contracts = []
    # for i, k in enumerate(contract_action_vals):
    #     october_contracts.append(
    #         ConstantParameter(model, name=f"october_contract{i}", value=k, is_variable=False, upper_bounds=1500))
    #     paramIndex[f"april_contract{i}"] = model.parameters.__len__() - 1
    # IndexedArrayParameter(
    #     model,
    #     name="october_contract",
    #     index_parameter=model.parameters["DP_index"],  # DP_index parameter
    #     params=october_contracts,
    #     comment="variable parameter that set the contract shares in october for a determined dp"
    # )
    # paramIndex["october_contract"] = model.parameters.__len__() - 1
    #
    # # contract_value
    # PolicyTreeTriggerHardCoded(
    #     model,
    #     name="contract_value",
    #     thresholds={
    #         1: model.parameters["april_threshold"],  # april_threshold parameter
    #         27: model.parameters["october_threshold"]  # october_threshold parameter
    #     },
    #     contracts={
    #         1: model.parameters["april_contract"],  # april_threshold parameter
    #         27: model.parameters["october_contract"]  # october_threshold parameter
    #     },
    #     drought_status=model.parameters[drought_status_agg],  # drought_status parameter
    #     comment="Receive two dates where the drought status is evaluated, the contract and the reservoir evaluated, and gives back the amount of shares transferred in that specific week"
    # )
    # paramIndex["contract_value"] = model.parameters.__len__() - 1


    # preset contract
    ConstantParameter(
        model,
        name="contract_value",
        value=contracts_purchased
    )
    paramIndex["contract_value"] = model.parameters.__len__() - 1

    # # purchases_value
    # purchases = []
    # for i in range(len(contract_action_vals)):
    #     purchases.append(ConstantParameter(model, name=f"purchase{i}", value=0, is_variable=False, upper_bounds=813))
    #     paramIndex[f"purchase{i}"] = model.parameters.__len__() - 1
    # AccumulatedIndexedArrayParameter(
    #     model,
    #     name="purchases_value",
    #     index_parameter=model.parameters["DP_index"],  # DP_index parameter
    #     params=purchases,
    #     comment="parameter that set the shares bought at a determined dp, accumulating past purchases"
    # )

    # purchases_value (0 for now)
    ConstantParameter(
        model,
        name="purchases_value",
        value=0
    )
    paramIndex["purchases_value"] = model.parameters.__len__() - 1

    # # thresholds for level1 demand restriction
    # ConstantParameter(
    #     model,
    #     name="level1",
    #     value=-10000
    # )
    # paramIndex["level1"] = model.parameters.__len__() - 1
    #
    # # thresholds for level2 demand restriction
    # ConstantParameter(
    #     model,
    #     name="level2",
    #     value=-20000
    # )
    # paramIndex["level2"] = model.parameters.__len__() - 1

    demand_control_curves = []
    for i in range(len(demand_threshold_vals)):
        # Assume we only pass in monthly profiles
        demand_control_curves.append(
            MonthlyProfileParameter(
                model, name=f"demand_control_curve{i}", values=demand_threshold_vals[i]
            )
        )
        paramIndex[f"demand_control_curve{i}"] = model.parameters.__len__() - 1

    # demand restriction level (done with indicators)
    IndicatorControlCurveIndexParameter(
        model,
        name="demand_restriction_level",
        indicator=model.parameters["drought_status_single_day"],
        control_curves=demand_control_curves
    )
    paramIndex["demand_restriction_level"] = model.parameters.__len__() - 1

    monthly_demand_restrictions = []
    for i in range(len(demand_action_vals)):
        # Assume we only pass in monthly profiles
        monthly_demand_restrictions.append(
            MonthlyProfileParameter(
                model, name=f"monthly_demand_restriction{i}", values=demand_action_vals[i]
            )
        )
        paramIndex[f"monthly_demand_restriction{i}"] = model.parameters.__len__() - 1

    # Demand restriction factor
    IndexedArrayParameter(
        model,
        name="demand_restriction_factor",
        index_parameter=model.parameters["demand_restriction_level"],
        params=monthly_demand_restrictions
    )
    paramIndex["demand_restriction_factor"] = model.parameters.__len__() - 1

    # # demand restriction level
    # ControlCurveIndexParameter(
    #     model,
    #     name="demand_restriction_level",
    #     storage_node=Embalse_Maipo,
    #     control_curves=[
    #         model.parameters["level1"],
    #         model.parameters["level2"]
    #     ]
    # )
    # paramIndex["demand_restriction_level"] = model.parameters.__len__() - 1
    #
    # # Demand restriction factor
    # IndexedArrayParameter(
    #     model,
    #     name="demand_restriction_factor",
    #     index_parameter=model.parameters["demand_restriction_level"],
    #     params=[
    #         ConstantParameter(model, value=1),
    #         MonthlyProfileParameter(
    #             model,
    #             values=[0.95, 0.95, 0.95, 0.95, 0.90, 0.90, 0.90, 0.90, 0.95, 0.95, 0.95, 0.95]
    #         ),
    #         MonthlyProfileParameter(
    #             model,
    #             values=[0.8, 0.8, 0.8, 0.8, 0.75, 0.75, 0.75, 0.75, 0.8, 0.8, 0.8, 0.8]
    #         ),
    #     ]
    # )
    # paramIndex["demand_restriction_factor"] = model.parameters.__len__() - 1

    # flow_Yeso
    df = {
        'url': '{}YESO.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Yeso"
    )
    paramIndex['flow_Yeso'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Yeso']].name = 'flow_Yeso'  # add name to parameter

    # flow_Maipo
    df = {
        'url': '{}/MAIPO.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Maipo"
    )
    paramIndex['flow_Maipo'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Maipo']].name = 'flow_Maipo'  # add name to parameter

    # flow_Colorado
    df = {
        'url': '{}COLORADO.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Colorado"
    )
    paramIndex['flow_Colorado'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Colorado']].name = 'flow_Colorado'  # add name to parameter

    # flow_Volcan
    df = {
        'url': '{}VOLCAN.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Volcan"
    )
    paramIndex['flow_Volcan'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Volcan']].name = 'flow_Volcan'  # add name to parameter

    # flow_Laguna Negra
    df = {
        'url': '{}LAGUNANEGRA.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Laguna Negra"
    )
    paramIndex['flow_Laguna Negra'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Laguna Negra']].name = 'flow_Laguna Negra'  # add name to parameter

    # flow_Maipo extra
    df = {
        'url': '{}MAIPOEXTRA.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        scenario=model.scenarios.scenarios[0],
        name="flow_Maipo extra"
    )
    paramIndex['flow_Maipo extra'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['flow_Maipo extra']].name = 'flow_Maipo extra'  # add name to parameter

    # aux_acueductoln
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Acueducto Laguna Negra']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="aux_acueductoln"
    )
    paramIndex['aux_acueductoln'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['aux_acueductoln']].name = 'aux_acueductoln'  # add name to parameter

    # aux_extraccionln
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Extraccion Laguna Negra']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="aux_extraccionln"
    )
    paramIndex['aux_extraccionln'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['aux_extraccionln']].name = 'aux_extraccionln'  # add name to parameter

    # aux_acueductoyeso
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Acueducto El Yeso']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="aux_acueductoyeso"
    )
    paramIndex['aux_acueductoyeso'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['aux_acueductoyeso']].name = 'aux_acueductoyeso'  # add name to parameter

    # aux_filtraciones
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Filtraciones']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="aux_filtraciones"
    )
    paramIndex['aux_filtraciones'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['aux_filtraciones']].name = 'aux_filtraciones'  # add name to parameter

    # threshold_laobra
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Threshold']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="threshold_laobra"
    )
    paramIndex['threshold_laobra'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['threshold_laobra']].name = 'threshold_laobra'  # add name to parameter

    # discount_rate_factor
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Discount rate factor']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="discount_rate_factor"
    )
    paramIndex['discount_rate_factor'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['discount_rate_factor']].name = 'discount_rate_factor'  # add name to parameter

    # descarga_adicional
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'AdicionalEmbalse']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="descarga_adicional"
    )
    paramIndex['descarga_adicional'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['descarga_adicional']].name = 'descarga_adicional'  # add name to parameter

    # estacionalidad_distribucion
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'Estacionalidad']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="estacionalidad_distribucion"
    )
    paramIndex['estacionalidad_distribucion'] = model.parameters.__len__() - 1
    model.parameters._objects[
        paramIndex['estacionalidad_distribucion']].name = 'estacionalidad_distribucion'  # add name to parameter

    # demanda_PT1
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'PT1']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="demanda_PT1"
    )
    paramIndex['demanda_PT1'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['demanda_PT1']].name = 'demanda_PT1'  # add name to parameter

    # Restricted demand through PT1
    AggregatedParameter(
        model,
        name="demand_max_flow_PT1",
        parameters=[
            model.parameters['demanda_PT1'],
            model.parameters['demand_restriction_factor']
        ],
        agg_func="product"
    )
    paramIndex["demand_max_flow_PT1"] = model.parameters.__len__() - 1

    # # Agricultural demand on monthly cycle
    # MonthlyProfileParameter(
    #     model,
    #     name="agricultural_demand",
    #     # values approximated by scaling water rights profile by ratio of demand to rights
    #     values=[18.01638039, 17.94856465, 17.84307351, 17.8053981, 17.7225122, 17.74511745,
    #             17.76772269, 17.77525777, 17.78279286, 17.91088925, 17.97870498, 18.15201186]
    # )
    # paramIndex['agricultural_demand'] = model.parameters.__len__() - 1


    # Agricultural demand (doesn't vary much, treated as constant for now)
    ConstantParameter(
        model,
        name="agricultural_demand_constant",
        value=16.978
    )
    paramIndex['agricultural_demand_constant'] = model.parameters.__len__() - 1

    # # Agricultural water rights on monthly cycle
    # MonthlyProfileParameter(
    #     model,
    #     name="agricultural_water_rights",
    #     # values taken from DGA paper, changing from m3/s to Mm3/week
    #     values=[144.60768, 144.06336, 143.21664, 142.91424, 142.24896, 142.4304,
    #             142.61184, 142.67232, 142.7328, 143.76096, 144.30528, 145.69632]
    # )
    # paramIndex['agricultural_water_rights'] = model.parameters.__len__() - 1

    # Agricultural shares (doesn't vary much, treated as constant for now)
    ConstantParameter(
        model,
        name="agricultural_shares_constant",
        value=3408.639
    )
    paramIndex['agricultural_water_rights'] = model.parameters.__len__() - 1

    # # Restricted demand through Agriculture
    # AggregatedParameter(
    #     model,
    #     name="demand_max_flow_Ag",
    #     parameters=[
    #         model.parameters['agricultural_demand'],
    #         model.parameters['demand_restriction_factor']
    #     ],
    #     agg_func="product"
    # )
    # paramIndex["demand_max_flow_Ag"] = model.parameters.__len__() - 1

    # demanda_PT2
    df = {
        'url': '{}Extra data.csv'.format(data_url_base),
        "parse_dates": True,
        "index_col": 0,
        "dayfirst": True,
        "usecols": ['Timestamp', 'PT2']
    }
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df).squeeze(),
        name="demanda_PT2"
    )
    paramIndex['demanda_PT2'] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex['demanda_PT2']].name = 'demanda_PT2'  # add name to parameter

    # demanda_PT2_negativa
    NegativeParameter(
        model,
        parameter=model.parameters["demanda_PT2"],
        name="demanda_PT2_negativa"
    )
    paramIndex['demanda_PT2_negativa'] = model.parameters.__len__() - 1

    # requisito_embalse
    ParameterThresholdParameter(
        model,
        param=Embalse_fixed_flow,
        threshold=140 - 15,
        predicate="LT",
        values=[1, 0],
        name="requisito_embalse"
    )
    paramIndex['requisito_embalse'] = model.parameters.__len__() - 1

    # caudal_naturalizado
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["flow_Maipo"],
            model.parameters["flow_Yeso"],
            model.parameters["flow_Colorado"],
            model.parameters["flow_Laguna Negra"],
            model.parameters["flow_Maipo extra"],
            model.parameters["flow_Volcan"]
        ],
        agg_func="sum",
        name="caudal_naturalizado"
    )
    paramIndex['caudal_naturalizado'] = model.parameters.__len__() - 1

    # flow_Volcan+Maipo
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["flow_Maipo"],
            model.parameters["flow_Volcan"]
        ],
        agg_func="sum",
        name="flow_Volcan+Maipo"
    )
    paramIndex['flow_Volcan+Maipo'] = model.parameters.__len__() - 1

    # descarga_embalse
    ParameterThresholdParameter(
        model,
        param=model.parameters["caudal_naturalizado"],
        threshold=60.48,
        predicate="LT",
        values=[0, 1],
        name="descarga_embalse"
    )
    paramIndex['descarga_embalse'] = model.parameters.__len__() - 1

    # descarga_embalse_real
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_embalse"],
            model.parameters["flow_Yeso"]
        ],
        agg_func="product",
        name="descarga_embalse_real"
    )
    paramIndex['descarga_embalse_real'] = model.parameters.__len__() - 1

    # descarga_embalse_real_Maipo
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["flow_Volcan+Maipo"],
            model.parameters["descarga_embalse"]
        ],
        agg_func="product",
        name="descarga_embalse_real_Maipo"
    )
    paramIndex['descarga_embalse_real_Maipo'] = model.parameters.__len__() - 1

    # descarga_adicional2
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_adicional"],
            model.parameters["requisito_embalse"]
        ],
        agg_func="product",
        name="descarga_adicional2"
    )
    paramIndex['descarga_adicional2'] = model.parameters.__len__() - 1

    # descarga_adicional_Maipo
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_adicional"],
            model.parameters["requisito_embalse_Maipo"]
        ],
        agg_func="product",
        name="descarga_adicional_Maipo"
    )
    paramIndex['descarga_adicional_Maipo'] = model.parameters.__len__() - 1

    # descarga_adicional_real
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_adicional2"],
            model.parameters["aux_acueductoyeso"]
        ],
        agg_func="product",
        name="descarga_adicional_real"
    )
    paramIndex['descarga_adicional_real'] = model.parameters.__len__() - 1

    # descarga_regla_Maipo
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["descarga_adicional_Maipo"],
            model.parameters["aux_acueductoyeso"]
        ],
        agg_func="product",
        name="descarga_regla_Maipo"
    )
    paramIndex['descarga_regla_Maipo'] = model.parameters.__len__() - 1

    # AA_total_shares_constant
    ConstantParameter(
        model,
        name="AA_total_shares_constant",
        value=1917
    )
    paramIndex['AA_total_shares_constant'] = model.parameters.__len__() - 1

    # AA_total_shares
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["AA_total_shares_constant"],
            model.parameters["purchases_value"],
            model.parameters["contract_value"],
            model.parameters["agricultural_shares_constant"]
        ],
        # Shares bought can't be more than what ag has to give
        agg_func=lambda x: np.min([x[0] + x[1] + x[2], x[0] + x[3]]),
        name="AA_total_shares",
        comment="expressed as absolute value of total shares"
    )
    paramIndex['AA_total_shares'] = model.parameters.__len__() - 1

    # Agriculture_total_shares
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["agricultural_shares_constant"],
            model.parameters["purchases_value"],
            model.parameters["contract_value"]
        ],
        agg_func=lambda x: np.max([x[0] - x[1] - x[2], 0]),
        name="ag_total_shares",
        comment="expressed as absolute value of total shares"
    )
    paramIndex['ag_total_shares'] = model.parameters.__len__() - 1

    # AA_total_shares_fraction_constant
    ConstantParameter(
        model,
        name="AA_total_shares_fraction_constant",
        value=.0001229558588466740
    )
    paramIndex['AA_total_shares_fraction_constant'] = model.parameters.__len__() - 1

    # AA_total_shares_fraction
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["AA_total_shares"],
            model.parameters["AA_total_shares_fraction_constant"]
        ],
        agg_func="product",
        name="AA_total_shares_fraction",
        comment="expressed as fraction of total shares"
    )
    paramIndex['AA_total_shares_fraction'] = model.parameters.__len__() - 1

    # ag_total_shares_fraction
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["ag_total_shares"],
            model.parameters["AA_total_shares_fraction_constant"]
        ],
        agg_func="product",
        name="ag_total_shares_fraction",
        comment="expressed as fraction of total shares"
    )
    paramIndex['ag_total_shares_fraction'] = model.parameters.__len__() - 1

    # max_flow_perdicez
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["caudal_naturalizado"],
            model.parameters["estacionalidad_distribucion"],
            model.parameters["AA_total_shares_fraction"]
        ],
        agg_func="product",
        name="max_flow_perdicez"
    )
    paramIndex['max_flow_perdicez'] = model.parameters.__len__() - 1

    # derechos_sobrantes
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["max_flow_perdicez"],
            model.parameters["demanda_PT2_negativa"]
        ],
        agg_func="sum",
        name="derechos_sobrantes"
    )
    paramIndex['derechos_sobrantes'] = model.parameters.__len__() - 1

    # contrato
    ConstantParameter(
        model,
        name="contrato",
        value=0
    )
    paramIndex['contrato'] = model.parameters.__len__() - 1

    # derechos_sobrantes_contrato
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["derechos_sobrantes"],
            model.parameters["contrato"]
        ],
        agg_func="sum",
        name="derechos_sobrantes_contrato"
    )
    paramIndex['derechos_sobrantes_contrato'] = model.parameters.__len__() - 1

    # ag max flow
    AggregatedParameter(
        model,
        parameters=[
            model.parameters["caudal_naturalizado"],
            model.parameters["estacionalidad_distribucion"],
            model.parameters["ag_total_shares"]
        ],
        agg_func="product",
        name="ag_max_flow"
    )
    paramIndex['ag_max_flow'] = model.parameters.__len__() - 1

    # # Restricted Ag max flow
    # AggregatedParameter(
    #     model,
    #     parameters=[
    #         model.parameters["agricultural_water_rights"],
    #         model.parameters["contract_value"],
    #         model.parameters["demand_max_flow_Ag"]
    #     ],
    #     agg_func=lambda arr: np.min([np.max([arr[0] - arr[1], 0]), arr[2]]),
    #     name="restricted_ag_max_flow"
    # )
    # paramIndex['restricted_ag_max_flow'] = model.parameters.__len__() - 1

    # # WATER RIGHTS LEFT -- JUST FOR TESTING
    # AggregatedParameter(
    #     model,
    #     parameters=[
    #         model.parameters["agricultural_water_rights"],
    #         model.parameters["contract_value"],
    #     ],
    #     agg_func=lambda arr: np.max([arr[0] - arr[1], 0]),
    #     name="water_rights_left"
    # )

    # REMAINING NODES
    # Yeso
    Yeso = Catchment(
        model,
        name="Yeso",
        flow=model.parameters["flow_Yeso"]
    )

    # Maipo
    Maipo = Catchment(
        model,
        name="Maipo",
        flow=model.parameters["flow_Maipo"]
    )

    # Colorado
    Colorado = Catchment(
        model,
        name="Colorado",
        flow=model.parameters["flow_Colorado"]
    )

    # Volcan
    Volcan = Catchment(
        model,
        name="Volcan",
        flow=model.parameters["flow_Volcan"]
    )

    # Laguna negra
    Laguna_negra = Catchment(
        model,
        name="Laguna negra",
        flow=model.parameters["flow_Laguna Negra"]
    )

    # Maipo extra
    Maipo_extra = Catchment(
        model,
        name="Maipo extra",
        flow=model.parameters["flow_Maipo extra"]
    )

    # Catchment that leads into Embalse_fixed_inflow to replace Embalse in this model
    Embalse_fixed_catchment = Catchment(
        model,
        name="Embalse fixed catchment",
        flow=model.parameters["Embalse_fixed_flow"]
    )

    # Inflow node that the Yeso catchment and Embalse_fixed_catchement lead to
    Embalse_fixed_inflow = River(
        model,
        name="Embalse fixed inflow"
    )

    Embalse_fixed_outflow = River(
        model,
        name="Embalse fixed outflow",
        max_flow=220-15,
        cost=1000
    )

    Embalse_fixed_drain = Output(
        model,
        name="Embalse fixed drain"
    )


    # Regla Embalse Maipo
    Regla_Embalse_Maipo = River(
        model,
        name="Regla Embalse Maipo",
        min_flow=model.parameters["descarga_regla_Maipo"],
        cost=100
    )

    # Rio Yeso Alto
    Rio_Yeso_Alto = River(
        model,
        name="Rio Yeso Alto"
    )

    # Rio Yeso Alto
    Rio_Yeso_Bajo = River(
        model,
        name="Rio Yeso Bajo"
    )

    # Rio Maipo Alto
    Rio_Maipo_Alto = River(
        model,
        name="Rio Maipo Alto"
    )

    # Rio Colorado
    Rio_Colorado = River(
        model,
        name="Rio Colorado"
    )

    # Estero del Manzanito
    Estero_del_Manzanito = River(
        model,
        name="Estero del Manzanito"
    )

    # aux_Maipo Extra
    aux_Maipo_Extra = River(
        model,
        name="aux_Maipo Extra"
    )

    # Acueducto Laguna Negra
    Acueducto_Laguna_Negra = River(
        model,
        name="Acueducto Laguna Negra",
        max_flow=model.parameters["aux_acueductoln"],
        cost=-1000
    )

    # Acueducto El Yeso
    Acueducto_El_Yeso = River(
        model,
        name="Acueducto El Yeso"
    )

    # Acueducto Maipo
    Acueducto_Maipo = River(
        model,
        name="Acueducto Maipo",
        cost=-200
    )

    # Acueducto El Yeso 2
    Acueducto_El_Yeso_2 = River(
        model,
        name="Acueducto El Yeso 2",
        min_flow=model.parameters["descarga_adicional_real"],
        cost=100
    )

    # Retorno al Maipo
    Retorno_al_Maipo = River(
        model,
        name="Retorno al Maipo",
        cost=-100
    )

    # aux_Acueducto Yeso
    aux_Acueducto_Yeso = River(
        model,
        name="aux_Acueducto Yeso"
    )

    # Rio Maipo bajo El Manzano
    Rio_Maipo_bajo_El_Manzano = River(
        model,
        name="Rio Maipo bajo El Manzano"
    )

    # Captacion PT
    Captacion_PT = River(
        model,
        name="Captacion PT"
    )

    # Extraccion a Acueducto Laguna Negra
    Extraccion_a_Acueducto_Laguna_Negra = River(
        model,
        max_flow=model.parameters['aux_extraccionln'],
        name="Extraccion a Acueducto Laguna Negra"
    )

    # Toma a PT1
    Toma_a_PT1 = River(
        model,
        name="Toma a PT1",
        max_flow=model.parameters["derechos_sobrantes_contrato"],
        cost=-300
    )

    # PT1 (unrestricted, lets us find stats with true demand)
    PT1 = Link(
        model,
        name="PT1",
        max_flow=model.parameters["demanda_PT1"]
    )

    # PT1 = RestrictedOutput(
    #     model,
    #     name="PT1",
    #     desired_flow=model.parameters["demanda_PT1"],
    #     restriction_factor=model.parameters["demand_restriction_factor"],
    #     cost=-10000
    # )

    # CAN CREATE A NEW NODE EXTENDING OUTPUT THAT TAKES IN A DESIRED FLOW AND RESTRICTION, LETS US
    # IMPLEMENT DEMAND RESTRICTIONS AND CREATE METRICS WITH DESIRED FLOW MORE CONSISELY

    # PT1 output node representing restricted demand
    PT1_output = Output(
        model,
        name="PT1_output",
        max_flow=model.parameters["demand_max_flow_PT1"],
        cost=-10000
    )

    # PT2
    PT2 = Output(
        model,
        name="PT2",
        max_flow=model.parameters["demanda_PT2"],
        cost=-10000
    )

    # Las Perdicez
    Las_Perdicez = River(
        model,
        name="Las Perdicez"
    )

    # Filtraciones Embalse
    Filtraciones_Embalse = River(
        model,
        name="Filtraciones Embalse",
        max_flow=model.parameters["aux_filtraciones"],
        cost=-9999
    )

    # Filtraciones Embalse Maipo
    Filtraciones_Embalse_Maipo = River(
        model,
        name="Filtraciones Embalse Maipo",
        max_flow=model.parameters["aux_filtraciones"],
        cost=-9999
    )

    # Descarga Embalse
    Descarga_Embalse = River(
        model,
        name="Descarga Embalse",
        min_flow=model.parameters["descarga_embalse_real"],
        cost=-100
    )

    # Descarga Embalse Maipo
    Descarga_Embalse_Maipo = River(
        model,
        name="Descarga Embalse Maipo",
        min_flow=model.parameters["descarga_embalse_real_Maipo"]
    )

    # aux_Salida Maipo
    aux_Salida_Maipo = River(
        model,
        name="aux_Salida Maipo",
        min_flow=model.parameters["flujo_excedentes_Yeso"]
    )

    # aux_PT1
    aux_PT1 = River(
        model,
        name="aux_PT1"
    )

    # Salida_Maipo (leftover leaving the basin)
    Salida_Maipo = Output(
        model,
        name="Salida Maipo",
        cost=-500
    )

    # Agriculture node with true demand
    Agriculture = Link(
        model,
        name="Agriculture",
        max_flow=model.parameters["agricultural_demand_constant"]
    )

    # Agriculture output node with demand restricted by water rights
    Agriculture_output = Output(
        model,
        name="Agriculture_output",
        max_flow=model.parameters["ag_max_flow"],
        cost=-600  # More negative than Salida_Maipo but not enough to take from Embalse
    )

    '''
    TO ADD:
    Negative parameter version of contrato (or derechos_sobrantes_contrato)
    Aggregated sum parameter of Ag water rights and negative contract
    Aggregated max parameter of above param and Ag demand, to be Ag max flow

    TO CONFIRM
    What's the difference between contrato and derechos_sobrantes?
    Also April contract and October contract?
    Which contracts initially belong to water utilities and which are bought from agriculture?
    Did I add agriculture in the right spot?
    Where will water rights be brought from?
    '''

    # EDGES
    # from catchment inflows
    Yeso.connect(Rio_Yeso_Alto)
    Maipo.connect(Rio_Maipo_Alto)
    Colorado.connect(Rio_Colorado)
    Volcan.connect(Rio_Maipo_Alto)
    Laguna_negra.connect(Acueducto_Laguna_Negra)
    Laguna_negra.connect(Estero_del_Manzanito)
    Maipo_extra.connect(aux_Maipo_Extra)
    # from storage nodes
    # Embalse.connect(Acueducto_El_Yeso_2)
    # Embalse.connect(Descarga_Embalse)
    # Embalse.connect(Filtraciones_Embalse)
    Embalse_fixed_catchment.connect(Embalse_fixed_inflow)
    Embalse_fixed_inflow.connect(Embalse_fixed_outflow)
    Embalse_fixed_inflow.connect(Embalse_fixed_drain)
    Embalse_fixed_outflow.connect(Acueducto_El_Yeso_2)
    Embalse_fixed_outflow.connect(Descarga_Embalse)
    Embalse_fixed_outflow.connect(Filtraciones_Embalse)
    Embalse_Maipo.connect(Descarga_Embalse_Maipo)
    Embalse_Maipo.connect(Acueducto_Maipo)
    Embalse_Maipo.connect(Regla_Embalse_Maipo)
    Embalse_Maipo.connect(Filtraciones_Embalse_Maipo)
    # from river nodes
    Regla_Embalse_Maipo.connect(El_Manzano)
    # Rio_Yeso_Alto.connect(Embalse)
    Rio_Yeso_Alto.connect(Embalse_fixed_inflow)
    Rio_Yeso_Bajo.connect(El_Manzano)
    Rio_Maipo_Alto.connect(Embalse_Maipo)
    Rio_Colorado.connect(El_Manzano)
    Estero_del_Manzanito.connect(Rio_Yeso_Bajo)
    aux_Maipo_Extra.connect(El_Manzano)
    Acueducto_Laguna_Negra.connect(aux_PT1)
    Acueducto_El_Yeso.connect(aux_Acueducto_Yeso)
    Acueducto_El_Yeso.connect(Retorno_al_Maipo)
    Acueducto_Maipo.connect(aux_PT1)
    Acueducto_El_Yeso_2.connect(Acueducto_El_Yeso)
    Retorno_al_Maipo.connect(El_Manzano)
    aux_Acueducto_Yeso.connect(aux_PT1)
    El_Manzano.connect(Rio_Maipo_bajo_El_Manzano)
    Rio_Maipo_bajo_El_Manzano.connect(Extraccion_a_Acueducto_Laguna_Negra)
    Rio_Maipo_bajo_El_Manzano.connect(Las_Perdicez)
    Rio_Maipo_bajo_El_Manzano.connect(Captacion_PT)
    Captacion_PT.connect(Toma_a_PT1)
    Captacion_PT.connect(aux_Salida_Maipo)
    Extraccion_a_Acueducto_Laguna_Negra.connect(Acueducto_Laguna_Negra)
    Toma_a_PT1.connect(aux_PT1)
    Las_Perdicez.connect(PT2)
    Filtraciones_Embalse.connect(Rio_Yeso_Bajo)
    Filtraciones_Embalse_Maipo.connect(El_Manzano)
    Descarga_Embalse.connect(Rio_Yeso_Bajo)
    Descarga_Embalse_Maipo.connect(El_Manzano)
    aux_Salida_Maipo.connect(Agriculture)
    aux_Salida_Maipo.connect(Salida_Maipo)
    aux_PT1.connect(PT1)

    PT1.connect(PT1_output)  # add demand restriction as a new node
    Agriculture.connect(Agriculture_output)  # add water rights limit as a new node

    # RECORDERS
    # RollingMeanFlowElManzano
    RollingMeanFlowNodeRecorder(
        model,
        node=model.nodes["El Manzano"],  # El_Manzano
        timesteps=520,
        name="RollingMeanFlowElManzano"
    )
    recorderIndex['RollingMeanFlowElManzano'] = model.recorders.__len__() - 1

    # failure_frequency_PT1 (measured with restricted demand)
    DeficitFrequencyNodeRecorder(
        model,
        node=model.nodes["PT1_output"],  # PT1
        is_objective="minimize",
        comment="Frequency deficit recorded on PT1 output",
        name="failure_frequency_PT1"
    )
    recorderIndex['failure_frequency_PT1'] = model.recorders.__len__() - 1

    def reliability_agg_func(x, axis=0):
        return (1 - np.array(x)).reshape((15,))
    # reliability_PT1 (measured with restricted demand)
    AggregatedRecorder(
        model,
        recorders=[model.recorders['failure_frequency_PT1']],
        recorder_agg_func=reliability_agg_func,
        name="reliability_PT1"
    )
    recorderIndex['reliability_PT1'] = model.recorders.__len__() - 1

    # failure_frequency_Ag
    DeficitFrequencyNodeRecorder(
        model,
        node=model.nodes["Agriculture"],  # Agriculture
        is_objective="minimize",
        comment="Frequency deficit recorded on Agriculture output",
        name="failure_frequency_Ag"
    )
    recorderIndex['failure_frequency_Ag'] = model.recorders.__len__() - 1

    # reliability_Ag
    AggregatedRecorder(
        model,
        recorders=[model.recorders['failure_frequency_Ag']],
        recorder_agg_func=reliability_agg_func,
        name="reliability_Ag"
    )
    recorderIndex['reliability_Ag'] = model.recorders.__len__() - 1

    # ReservoirCost
    ReservoirCostRecorder(
        model,
        capacity=model.parameters["Maipo_capacity"],
        construction_dp=model.parameters["Maipo_construction_dp"],
        discount_rate=0.035,
        unit_costs=[9999, 20, 30, 40, 50, 60, 0],
        fixed_cost=100,
        name="ReservoirCost"
    )
    recorderIndex['ReservoirCost'] = model.recorders.__len__() - 1

    # # PurchasesCost
    # PurchasesCostRecorder(
    #     model,
    #     purchases_value=model.parameters["purchases_value"],
    #     meanflow=model.recorders['RollingMeanFlowElManzano'],
    #     discount_rate=0.035,
    #     coeff=1,
    #     name="PurchasesCost"
    # )
    # recorderIndex['PurchasesCost'] = model.recorders.__len__() - 1

    # # AprilSeasonRollingMeanFlowElManzano
    # SeasonRollingMeanFlowNodeRecorder(
    #     model,
    #     node=El_Manzano,
    #     first_week=1,
    #     last_week=27,
    #     years=5,
    #     name='AprilSeasonRollingMeanFlowElManzano'
    # )
    # recorderIndex['AprilSeasonRollingMeanFlowElManzano'] = model.recorders.__len__() - 1
    #
    # # OctoberSeasonRollingMeanFlowElManzano
    # SeasonRollingMeanFlowNodeRecorder(
    #     model,
    #     node=El_Manzano,
    #     first_week=27,
    #     last_week=53,
    #     years=5,
    #     name='OctoberSeasonRollingMeanFlowElManzano'
    # )
    # recorderIndex['OctoberSeasonRollingMeanFlowElManzano'] = model.recorders.__len__() - 1
    #
    # # PremiumAprilCost
    # PurchasesCostRecorder(
    #     model,
    #     purchases_value=model.parameters["april_contract"],
    #     meanflow=model.recorders['RollingMeanFlowElManzano'],
    #     discount_rate=0.035,
    #     coeff=0.1,
    #     name="PremiumAprilCost"
    # )
    # recorderIndex['PremiumAprilCost'] = model.recorders.__len__() - 1
    #
    # # PremiumOctoberCost
    # PurchasesCostRecorder(
    #     model,
    #     purchases_value=model.parameters["october_contract"],
    #     meanflow=model.recorders["RollingMeanFlowElManzano"],
    #     discount_rate=0.035,
    #     coeff=0.1,
    #     name="PremiumOctoberCost"
    # )
    # recorderIndex['PremiumOctoberCost'] = model.recorders.__len__() - 1

    NumpyArrayContractCostRecorder(
        model,
        contract_value=model.parameters["contract_value"],
        meanflow=model.recorders["RollingMeanFlowElManzano"],
        purchases_value=model.parameters["purchases_value"],
        # discount_rate=0.035,
        max_cost=100,
        gradient=-1,
        coeff=1,
        # week_no=1,
        num_weeks=1560,
        name="ContractCost"
    )
    recorderIndex['ContractCost'] = model.recorders.__len__() - 1

    # # AprilContractCost
    # ContractCostRecorder(
    #     model,
    #     contract_value=model.parameters["april_contract"],
    #     meanflow=model.recorders["AprilSeasonRollingMeanFlowElManzano"],
    #     purchases_value=model.parameters["purchases_value"],
    #     discount_rate=0.035,
    #     max_cost=100,
    #     gradient=-1,
    #     coeff=1,
    #     week_no=1,
    #     name="AprilContractCost"
    # )
    # recorderIndex['AprilContractCost'] = model.recorders.__len__() - 1
    #
    # # OctoberContractCost
    # ContractCostRecorder(
    #     model,
    #     contract_value=model.parameters["october_contract"],
    #     meanflow=model.recorders["OctoberSeasonRollingMeanFlowElManzano"],
    #     purchases_value=model.parameters["purchases_value"],
    #     discount_rate=0.035,
    #     max_cost=100,
    #     gradient=-1,
    #     coeff=1,
    #     week_no=27,
    #     name="OctoberContractCost"
    # )
    # recorderIndex['OctoberContractCost'] = model.recorders.__len__() - 1
    #
    # # TotalCost
    # AggregatedRecorder(
    #     model,
    #     agg_func="mean",
    #     recorder_agg_func="sum",
    #     recorders=[
    #         model.recorders["ReservoirCost"],
    #         model.recorders["PurchasesCost"],
    #         model.recorders["PremiumAprilCost"],
    #         model.recorders["PremiumOctoberCost"],
    #         model.recorders["AprilContractCost"],
    #         model.recorders["OctoberContractCost"]
    #     ],
    #     is_objective="minimize",
    #     name="TotalCost"
    # )
    # recorderIndex['TotalCost'] = model.recorders.__len__() - 1

    # deficit PT1 (measured on original demand)
    TotalDeficitNodeRecorder(
        model,
        node=PT1,
        is_objective="min",
        comment="Total deficit recorded on PT1",
        name="deficit PT1"
    )
    recorderIndex['deficit PT1'] = model.recorders.__len__() - 1

    # deficit Ag
    TotalDeficitNodeRecorder(
        model,
        node=Agriculture,
        is_objective="min",
        comment="Total deficit recorded on Agriculture",
        name="deficit Ag"
    )
    recorderIndex['deficit Ag'] = model.recorders.__len__() - 1

    # Caudal en salida promedio
    MeanFlowNodeRecorder(
        model,
        node=Salida_Maipo,
        is_objective="max",
        comment="Mean flow at system output",
        name="Caudal en salida promedio"
    )
    recorderIndex['Caudal en salida promedio'] = model.recorders.__len__() - 1

    # Maximum Deficit on PT1 (measured with original demand)
    MaximumDeficitNodeRecorder(
        model,
        node=PT1,
        is_objective="min",
        name="Maximum Deficit PT1"
    )
    recorderIndex['Maximum Deficit PT1'] = model.recorders.__len__() - 1

    # Maximum Deficit on Agriculture
    MaximumDeficitNodeRecorder(
        model,
        node=Agriculture,
        is_objective="min",
        name="Maximum Deficit Ag"
    )
    recorderIndex['Maximum Deficit Ag'] = model.recorders.__len__() - 1

    # Total Contracts Made
    NumpyArrayParameterRecorder(
        model,
        param=model.parameters["contract_value"],
        temporal_agg_func="sum",
        agg_func="mean",
        is_objective="min",
        name="Total Contracts Made"
    )
    recorderIndex['Total Contracts Made'] = model.recorders.__len__() - 1

    # InstanstaneousDeficit (measured with original demand)
    InstantaneousDeficictNodeRecorder(
        model,
        node=PT1,
        name="InstanstaneousDeficit"
    )
    recorderIndex['InstanstaneousDeficit'] = model.recorders.__len__() - 1

    # PT1 flow
    NumpyArrayNodeRecorder(
        model,
        node=PT1,
        name="PT1 flow",
        agg_func="SUM"
    )
    recorderIndex['PT1 flow'] = model.recorders.__len__() - 1

    # Agriculture flow
    NumpyArrayNodeRecorder(
        model,
        node=Agriculture,
        name="Agriculture flow",
        agg_func="SUM"
    )
    recorderIndex['Agriculture flow'] = model.recorders.__len__() - 1

    # Salida Maipo flow
    NumpyArrayNodeRecorder(
        model,
        node=Salida_Maipo,
        name="Salida Maipo flow",
        agg_func="SUM"
    )
    recorderIndex['Salida Maipo flow'] = model.recorders.__len__() - 1

    # Embalse_fixed_drain
    NumpyArrayNodeRecorder(
        model,
        node=Embalse_fixed_drain,
        name="Embalse_fixed_drain flow",
        agg_func="SUM"
    )
    recorderIndex['Embalse_fixed_drain flow'] = model.recorders.__len__() - 1

    # # Embalse storage
    # NumpyArrayStorageRecorder(
    #     model,
    #     node=Embalse,
    #     name="Embalse storage"
    # )
    # recorderIndex['Embalse storage'] = model.recorders.__len__() - 1

    # Total inflow from catchments
    NumpyArrayParameterRecorder(
        model,
        param=model.parameters["caudal_naturalizado"],
        name="Total catchment inflow"
    )
    recorderIndex['Total catchment inflow'] = model.recorders.__len__() - 1

    # remaining water rights per week
    NumpyArrayParameterRecorder(
        model,
        param=model.parameters["ag_max_flow"],
        name="Remaining water rights per week"
    )
    recorderIndex['Remaining water rights per week'] = model.recorders.__len__() - 1

    # Agricultural demand
    NumpyArrayParameterRecorder(
        model,
        param=model.parameters["agricultural_demand_constant"],
        name="Agricultural demand recorder",
    )
    recorderIndex['Agricultural demand recorder'] = model.recorders.__len__() - 1

    # check model validity
    model.check_graph()  # check the connectivity of the graph
    model.check()  # check the validity of the model

    return model


# We could call with contract_threshold_vals, contract_action_vals,
# demand_threshold_vals, demand_action_vals
def Simulation_Caller(contract_threshold_val, contract_action_val, demand_threshold_val, demand_action_val):
    '''
    Purpose: Borg calls this function to run the simulation model and return multi-objective performance.

    Note: You could also just put your simulation/function evaluation code here.

    Args:
        vars: A list of decision variable values from Borg

    Returns:
        performance: policy's simulated objective values. A list of objective values, one value each of the objectives.
    '''

    # borg_vars = vars  # Decision variable values from Borg
    #
    # # Reformat decision variable values as necessary (.e.g., cast borg output parameters as array for use in simulation)
    # op_policy_params = np.asarray(borg_vars)
    #
    # # Call/run simulation model, return multi-objective performance:
    # performance = pysedsim.PySedSim(decision_vars=op_policy_params)

    # num_k = 1  # number of levels in policy tree
    # num_DP = 7  # number of decision periods

    # params = {
    #     'contract_threshold_vals': vars[0] * np.ones(num_DP),
    #     'contract_action_vals': vars[1] * np.ones(num_DP),
    #     'demand_threshold_vals': [vars[2] * np.ones(12)],
    #     'demand_action_vals': [vars[3] * np.ones(12), np.ones(12)]
    # }

    # WHAT IN PARTICULAR ARE WE RETURNING HERE?
    # SHOULD PROBABLY CALL CHECK_GRAPH AND STUFF, THEN RUN, THEN GET RECORDER VALS

    contract_threshold_vals = contract_threshold_val * np.ones(num_DP)
    contract_action_vals = contract_action_val * np.ones(num_DP)
    demand_threshold_vals = [demand_threshold_val * np.ones(12)]
    demand_action_vals = [np.ones(12), demand_action_val * np.ones(12)]

    model = make_model(
        contract_threshold_vals=contract_threshold_vals,
        contract_action_vals=contract_action_vals,
        demand_threshold_vals=demand_threshold_vals,
        demand_action_vals=demand_action_vals
    )
    model.check()
    model.check_graph()
    model.find_orphaned_parameters()
    model.run()

    func_list = {
        'mean': np.mean,
        'max': np.max,
        'min': np.min
    }

    def get_performance(recorder):
        # model.recorders[recorder]
        func = func_list[recorder.agg_func]
        values = recorder.values()
        return func(values)

    # I think borg wants to maximize, so adding negatives where needed
    performance = [
        get_performance(model.recorders['reliability_PT1']),
        get_performance(model.recorders['reliability_Ag']),
        -get_performance(model.recorders['TotalCost']),
        -get_performance(model.recorders['deficit PT1']),
        -get_performance(model.recorders['deficit Ag']),
        get_performance(model.recorders['Caudal en salida promedio']),
        -get_performance(model.recorders['Maximum Deficit PT1']),
        -get_performance(model.recorders['Maximum Deficit Ag']),
        -get_performance(model.recorders['Total Contracts Made']),
    ]

    return performance


def Optimization():
    '''

    Purpose: Call this method from command line to initiate simulation-optimization experiment

    Returns:
        --pareto approximate set file (.set) for each random seed
        --Borg runtime file (.runtime) for each random seed

    '''

    # import MAIPO_PYWR.dps_BORG.borg as bg  # Import borg wrapper
    # import MAIPO_PYWR.dps_BORG.BorgMOEA_master.plugins.Python.borg as bg  # Import borg wrapper

    parallel = 0  # 1= master-slave (parallel), 0=serial

    if platform.system() == "Windows":
        data_url_base = "C:\\Users\\danny\\Pywr projects\\MAIPO_PYWR\\data\\"
    else:
        data_url_base = "/home/danny/FletcherLab/maipo-basin-pywr/MAIPO_PYWR/data/"

    # List of objectives:
    # failure_frequency_PT1 (minimize)
    # failure_frequency_Ag (minimize)
    # TotalCost (minimize)
    # deficit PT1 (minimize)
    # deficit Ag (minimize)
    # Caudal en salida promedio (mean flow at system output, maximize)
    # Maximum Deficit PT1 (minimize)
    # Maximum Deficit Ag (minimize)
    # Total Contracts Made (minimize)

    # List of decision variables:
    # contract_threshold_vals
    # contract_action_vals
    # demand_threshold_vals
    # demand_action_vals
    data = pd.read_csv("{}Extra data.csv".format(data_url_base))
    urban_demand = data["PT1"]
    num_weeks = len(urban_demand)
    total_urban_demand = urban_demand.sum()

    agricultural_demand_profile = np.array([18.01638039, 17.94856465, 17.84307351, 17.8053981, 17.7225122, 17.74511745,
                                            17.76772269, 17.77525777, 17.78279286, 17.91088925, 17.97870498,
                                            18.15201186])
    total_ag_demand = np.mean(agricultural_demand_profile) * num_weeks  # technically slightly off, but very close

    # Most below chosen arbitrarily
    nSeeds = 5  # Number of random seeds (Borg MOEA)
    num_dec_vars = 4  # Number of decision variables
    n_objs = 9  # Number of objectives
    n_constrs = 0  # Number of constraints
    num_func_evals = 100  # Number of total simulations to run per random seed. Each simulation may be a monte carlo.
    runtime_freq = 100  # Interval at which to print runtime details for each random seed
    decision_var_range = [[-2.5, 0], [0, 2100], [-2.5, 0], [0, 1]]
    epsilon_list = [0.01, 0.01, 0.05e9, 0.01 * total_urban_demand, 0.01 * total_ag_demand, 5, 0.25, 0.25,
                    1]  # Borg epsilon values for each objective

    # Where to save seed and runtime files
    main_output_file_dir = './MAIPO_PYWR/outputs/first_optimization_attempts'  # Specify location of output files for different seeds
    os_fold = Op_Sys_Folder_Operator()  # Folder operator for operating system
    output_location = main_output_file_dir + os_fold + 'sets'

    # If using master-slave, start MPI. Only do once.
    if parallel == 1:
        Configuration.startMPI()  # start parallelization with MPI

    # Loop through seeds, calling borg.solve (serial) or borg.solveMPI (parallel) each time
    for j in range(nSeeds):
        # Instantiate borg class, then set bounds, epsilon values, and file output locations
        borg = Borg(num_dec_vars, n_objs, n_constrs, Simulation_Caller)
        borg.setBounds(*decision_var_range)  # Set decision variable bounds
        borg.setEpsilons(*epsilon_list)  # Set epsilon values
        # Runtime file path for each seed:
        runtime_filename = bytes(main_output_file_dir + os_fold + 'runtime_file_seed_' + str(j + 1) + '%d.runtime', 'utf-8')
        if parallel == 1:
            # Run parallel Borg
            result = borg.solveMPI(maxEvaluations=num_func_evals, runtime=runtime_filename)#, frequency=runtime_freq)
        if parallel == 0:
            # Run serial Borg
            result = borg.solve({"maxEvaluations": num_func_evals, "runtimeformat": 'borg', "frequency": runtime_freq,
                                 "runtimefile": runtime_filename})

        if result:
            # This particular seed is now finished being run in parallel. The result will only be returned from
            # one node in case running Master-Slave Borg.
            result.display()

            # Create/write objective values and decision variable values to files in folder "sets", 1 file per seed.
            f = open(output_location + os_fold + 'Borg_DPS_PySedSim' + str(j + 1) + '.set', 'w')
            f.write('#Borg Optimization Results\n')
            f.write('#First ' + str(num_dec_vars) + ' are the decision variables, ' + 'last ' + str(n_objs) +
                    ' are the ' + 'objective values\n')
            for solution in result:
                line = ''
                for i in range(len(solution.getVariables())):
                    line = line + (str(solution.getVariables()[i])) + ' '

                for i in range(len(solution.getObjectives())):
                    line = line + (str(solution.getObjectives()[i])) + ' '

                f.write(line[0:-1] + '\n')
            f.write("#")
            f.close()

            # Create/write only objective values to files in folder "sets", 1 file per seed. Purpose is so that
            # the file can be processed in MOEAFramework, where performance metrics may be evaluated across seeds.
            f2 = open(output_location + os_fold + 'Borg_DPS_PySedSim_no_vars' + str(j + 1) + '.set', 'w')
            for solution in result:
                line = ''
                for i in range(len(solution.getObjectives())):
                    line = line + (str(solution.getObjectives()[i])) + ' '

                f2.write(line[0:-1] + '\n')
            f2.write("#")
            f2.close()

            print("Seed {} complete".format(j))

    if parallel == 1:
        Configuration.stopMPI()  # stop parallel function evaluation process


def Op_Sys_Folder_Operator():
    '''
    Function to determine whether operating system is (1) Windows, or (2) Linux

    Returns folder operator for use in specifying directories (file locations) for reading/writing data pre- and
    post-simulation.
    '''

    if platform.system() == 'Windows':
        os_fold_op = '\\'
    elif platform.system() == 'Linux':
        os_fold_op = '/'
    else:
        os_fold_op = '/'  # Assume unix OS if it can't be identified

    return os_fold_op






#%% practice running
# num_k = 1  # number of levels in policy tree
# num_DP = 7  # number of decision periods
# num_DP = 7  # number of decision periods
#
# thresh = np.ones(num_DP) * -0.84
# acts = np.ones(num_DP) * 300
#
# Simulation_Caller({
#     "contract_threshold_vals": -999999 * np.ones(num_DP),
#     "contract_action_vals": np.zeros(num_DP),
#     "demand_threshold_vals": [],
#     "demand_action_vals": [np.ones(12)],
#     "indicator": "SRI3",
#     "drought_status_agg": "drought_status_single_day_using_agg"
# })

# Optimization()