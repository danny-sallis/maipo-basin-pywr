# general package imports
import numpy as np
import pandas as pd
# import pysedsim  # This is your simulation model
import platform  # helps identify directory locations on different types of OS
import sys

# pywr imports
from pywr.core import *
from pywr.parameters import *
from pywr.parameters._thresholds import StorageThresholdParameter, ParameterThresholdParameter
from pywr.recorders import DeficitFrequencyNodeRecorder, TotalDeficitNodeRecorder, MeanFlowNodeRecorder, \
    NumpyArrayParameterRecorder, RollingMeanFlowNodeRecorder, AggregatedRecorder
#from MAIPO_searcher import *
from pywr.dataframe_tools import *

from MAIPO_PYWR.MAIPO_parameters import *
# from pysedsim_main.PySedSim import pysedsim


def make_model(threshold_vals, action_vals):
    '''
    Purpose: Creates a Pywr model with the specified number and values for policy thresholds/actions. Intended for use with MOEAs.

    Args:
        threshold_vals: an array of policy thresholds for drought index
        action_vals: an array of policy actions corresponding to policy thresholds

    Returns:
        model: a Pywr Model object
    '''

    # set current working directory
    #os.chdir(os.path.abspath(os.path.dirname(__file__)))

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
    Scenario(model, name="climate change", size=15)

    # REQUIRED NODES FOR PARAMETERS
    # Embalse
    Embalse = Storage(
        model,
        name="Embalse",
        min_volume=15,
        max_volume=220,
        initial_volume=220,
        cost=-1000
    )

    # Embalse Maipo
    Embalse_Maipo = Storage(
        model,
        name="Embalse Maipo",
        min_volume=0,
        max_volume=220,
        initial_volume=0.0,
        initial_volume_pc=0.0,
        cost=-800
    )

    # El Manzano
    El_Manzano = River(
        model,
        name="El Manzano"
    )

    # PARAMETERS
    # drought_status
    df = {
        'url': '../data/SRI3.csv',  # 'data/SRI3.csv'
        "parse_dates": True,
        "index_col": "Timestamp",
        "dayfirst": True}
    DataFrameParameter(
        model,
        dataframe=read_dataframe(model, df),
        name="drought_status",
        scenario=model.scenarios.scenarios[0],
    )
    paramIndex["drought_status"] = model.parameters.__len__() - 1
    model.parameters._objects[paramIndex["drought_status"]].name = "drought_status"  # add name to parameter

    # DP_index
    datestr = ["2006-07-12", "2007-03-29", "2007-12-14", "2008-08-30", "2009-05-17", "2010-02-01", "2010-10-19"]
    FakeYearIndexParameter(
        model,
        name="DP_index",
        dates=[datetime.datetime.strptime(i, '%Y-%m-%d') for i in datestr],
        comment="convert a specific date to integer, from 0 to 6, depending on the 5-year development plan period 2020-2050"
    )
    paramIndex["DP_index"] = model.parameters.__len__() - 1

    # april_threshold
    april_thresholds = []
    for i, k in enumerate(threshold_vals):
        april_thresholds.append(ConstantParameter(model, name=f"april_threshold{i}", value=k, is_variable=False, upper_bounds=0))
        paramIndex[f"april_threshold{i}"] = model.parameters.__len__() - 1
    IndexedArrayParameter(
        model,
        name="april_threshold",
        index_parameter=model.parameters["DP_index"],  # DP_index
        params=april_thresholds,
        comment="variable parameter that set the drought threshold for contracts in april"
    )
    paramIndex["april_threshold"] = model.parameters.__len__() - 1

    # october_threshold
    october_thresholds = []
    for i, k in enumerate(threshold_vals):
        october_thresholds.append(ConstantParameter(model, name=f"october_threshold{i}", value=k, is_variable=False, upper_bounds=0))
        paramIndex[f"october_threshold{i}"] = model.parameters.__len__() - 1
    IndexedArrayParameter(
        model,
        name="october_threshold",
        index_parameter=model.parameters["DP_index"],  # DP_index
        params=october_thresholds,
        comment="variable parameter that set the drought threshold for contracts in october"
    )
    paramIndex["october_threshold"] = model.parameters.__len__() - 1

    # april_contract
    april_contracts = []
    for i, k in enumerate(action_vals):
        april_contracts.append(ConstantParameter(model, name=f"april_contract{i}", value=k, is_variable=False, upper_bounds=1500))
        paramIndex[f"april_contract{i}"] = model.parameters.__len__() - 1
    IndexedArrayParameter(
        model,
        name="april_contract",
        index_parameter=model.parameters["DP_index"],  # DP_index
        params=april_contracts,
        comment="variable parameter that set the contract shares in april for a determined dp"
    )
    paramIndex["april_contract"] = model.parameters.__len__() - 1

    # october_contract
    october_contracts = []
    for i, k in enumerate(action_vals):
        october_contracts.append(ConstantParameter(model, name=f"october_contract{i}", value=k, is_variable=False, upper_bounds=1500))
        paramIndex[f"april_contract{i}"] = model.parameters.__len__() - 1
    IndexedArrayParameter(
        model,
        name="october_contract",
        index_parameter=model.parameters["DP_index"],  # DP_index parameter
        params=october_contracts,
        comment="variable parameter that set the contract shares in october for a determined dp"
    )
    paramIndex["october_contract"] = model.parameters.__len__() - 1

    # contract_value
    PolicyTreeTriggerHardCoded(
        model,
        name="contract_value",
        thresholds={
            "1": model.parameters["april_threshold"],  # april_threshold parameter
            "27": model.parameters["october_threshold"]  # october_threshold parameter
        },
        contracts={
            "1": model.parameters["april_contract"],  # april_threshold parameter
            "27": model.parameters["october_contract"]  # october_threshold parameter
        },
        drought_status=model.parameters["drought_status"],  # drought_status parameter
        comment="Receive two dates where the drought status is evaluated, the contract and the reservoir evaluated, and gives back the amount of shares transferred in that specific week"
    )
    paramIndex["contract_value"] = model.parameters.__len__() - 1

    # purchases_value
    purchases = []
    for i in range(len(action_vals)):
        purchases.append(ConstantParameter(model, name=f"purchase{i}", value=0, is_variable=False, upper_bounds=813))
        paramIndex[f"purchase{i}"] = model.parameters.__len__() - 1
    AccumulatedIndexedArrayParameter(
        model,
        name="purchases_value",
        index_parameter=model.parameters["DP_index"],  # DP_index parameter
        params=purchases,
        comment="parameter that set the shares bought at a determined dp, accumulating past purchases"
    )
    paramIndex["purchases_value"] = model.parameters.__len__() - 1

    # flow_Yeso
    df = {
        'url': '../data/YESO.csv',
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
        'url': '../data/MAIPO.csv',
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
        'url': '../data/COLORADO.csv',
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
        'url': '../data/VOLCAN.csv',
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
        'url': '../data/LAGUNANEGRA.csv',
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
        'url': '../data/MAIPOEXTRA.csv',
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
        'url': '../data/Extra data.csv',
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
        'url': '../data/Extra data.csv',
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
        'url': '../data/Extra data.csv',
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
        'url': '../data/Extra data.csv',
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
        'url': '../data/Extra data.csv',
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
        'url': '../data/Extra data.csv',
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
        'url': '../data/Extra data.csv',
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
        'url': '../data/Extra data.csv',
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
    model.parameters._objects[paramIndex['estacionalidad_distribucion']].name = 'estacionalidad_distribucion'  # add name to parameter

    # demanda_PT1
    df = {
        'url': '../data/Extra data.csv',
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

    # demanda_PT2
    df = {
        'url': '../data/Extra data.csv',
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
    StorageThresholdParameter(
        model,
        storage=Embalse,
        threshold=140,
        predicate="LT",
        values=[1, 0],
        name="requisito_embalse"
    )
    paramIndex['requisito_embalse'] = model.parameters.__len__() - 1

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
            model.parameters["contract_value"]
        ],
        agg_func="sum",
        name="AA_total_shares",
        comment="expressed as absolute value of total shares"
    )
    paramIndex['AA_total_shares'] = model.parameters.__len__() - 1

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

    # PT1
    PT1 = Output(
        model,
        name="PT1",
        max_flow=model.parameters["demanda_PT1"],
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

    # Salida_Maipo
    Salida_Maipo = Output(
        model,
        name="Salida Maipo",
        cost=-500
    )

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
    Embalse.connect(Acueducto_El_Yeso_2)
    Embalse.connect(Descarga_Embalse)
    Embalse.connect(Filtraciones_Embalse)
    Embalse_Maipo.connect(Descarga_Embalse_Maipo)
    Embalse_Maipo.connect(Acueducto_Maipo)
    Embalse_Maipo.connect(Regla_Embalse_Maipo)
    Embalse_Maipo.connect(Filtraciones_Embalse_Maipo)
    # from river nodes
    Regla_Embalse_Maipo.connect(El_Manzano)
    Rio_Yeso_Alto.connect(Embalse)
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
    aux_Salida_Maipo.connect(Salida_Maipo)
    aux_PT1.connect(PT1)

    # RECORDERS
    # RollingMeanFlowElManzano
    RollingMeanFlowNodeRecorder(
        model,
        node=model.nodes["El Manzano"],  #El_Manzano
        timesteps=520,
        name="RollingMeanFlowElManzano"
    )
    recorderIndex['RollingMeanFlowElManzano'] = model.recorders.__len__() - 1

    # failure_frequency_PT1
    DeficitFrequencyNodeRecorder(
        model,
        node=model.nodes["PT1"],  # PT1
        is_objective="minimize",
        comment="Frequency deficit recorded on PT1",
        name="failure_frequency_PT1"
    )
    recorderIndex['failure_frequency_PT1'] = model.recorders.__len__() - 1

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

    # PurchasesCost
    PurchasesCostRecorder(
        model,
        purchases_value=model.parameters["purchases_value"],
        meanflow=model.recorders['RollingMeanFlowElManzano'],
        discount_rate=0.035,
        coeff=1,
        name="PurchasesCost"
    )
    recorderIndex['PurchasesCost'] = model.recorders.__len__() - 1

    # AprilSeasonRollingMeanFlowElManzano
    SeasonRollingMeanFlowNodeRecorder(
        model,
        node=El_Manzano,
        first_week=1,
        last_week=27,
        years=5,
        name='AprilSeasonRollingMeanFlowElManzano'
    )
    recorderIndex['AprilSeasonRollingMeanFlowElManzano'] = model.recorders.__len__() - 1

    # OctoberSeasonRollingMeanFlowElManzano
    SeasonRollingMeanFlowNodeRecorder(
        model,
        node=El_Manzano,
        first_week=27,
        last_week=53,
        years=5,
        name='OctoberSeasonRollingMeanFlowElManzano'
    )
    recorderIndex['OctoberSeasonRollingMeanFlowElManzano'] = model.recorders.__len__() - 1

    # PremiumAprilCost
    PurchasesCostRecorder(
        model,
        purchases_value=model.parameters["april_contract"],
        meanflow=model.recorders['RollingMeanFlowElManzano'],
        discount_rate=0.035,
        coeff=0.1,
        name="PremiumAprilCost"
    )
    recorderIndex['PremiumAprilCost'] = model.recorders.__len__() - 1

    # PremiumOctoberCost
    PurchasesCostRecorder(
        model,
        purchases_value=model.parameters["october_contract"],
        meanflow=model.recorders["RollingMeanFlowElManzano"],
        discount_rate=0.035,
        coeff=0.1,
        name="PremiumOctoberCost"
    )
    recorderIndex['PremiumOctoberCost'] = model.recorders.__len__() - 1

    # AprilContractCost
    ContractCostRecorder(
        model,
        contract_value=model.parameters["april_contract"],
        meanflow=model.recorders["AprilSeasonRollingMeanFlowElManzano"],
        purchases_value=model.parameters["purchases_value"],
        discount_rate=0.035,
        max_cost=100,
        gradient=-1,
        coeff=1,
        week_no=1,
        name="AprilContractCost"
    )
    recorderIndex['AprilContractCost'] = model.recorders.__len__() - 1

    # OctoberContractCost
    ContractCostRecorder(
        model,
        contract_value=model.parameters["october_contract"],
        meanflow=model.recorders["OctoberSeasonRollingMeanFlowElManzano"],
        purchases_value=model.parameters["purchases_value"],
        discount_rate=0.035,
        max_cost=100,
        gradient=-1,
        coeff=1,
        week_no=27,
        name="OctoberContractCost"
    )
    recorderIndex['OctoberContractCost'] = model.recorders.__len__() - 1

    # TotalCost
    AggregatedRecorder(
        model,
        agg_func="mean",
        recorder_agg_func="sum",
        recorders=[
            model.recorders["ReservoirCost"],
            model.recorders["PurchasesCost"],
            model.recorders["PremiumAprilCost"],
            model.recorders["PremiumOctoberCost"],
            model.recorders["AprilContractCost"],
            model.recorders["OctoberContractCost"]
        ],
        is_objective="minimize",
        name="TotalCost"
    )
    recorderIndex['TotalCost'] = model.recorders.__len__() - 1

    # deficit PT1
    TotalDeficitNodeRecorder(
        model,
        node=PT1,
        is_objective="min",
        comment="Total deficit recorded on PT1",
        name="deficit PT1"
    )
    recorderIndex['deficit PT1'] = model.recorders.__len__() - 1

    # Caudal en salida promedio
    MeanFlowNodeRecorder(
        model,
        node=Salida_Maipo,
        is_objective="max",
        comment="Mean flow at system output",
        name="Caudal en salida promedio"
    )
    recorderIndex['Caudal en salida promedio'] = model.recorders.__len__() - 1

    # Maximum Deficit
    MaximumDeficitNodeRecorder(
        model,
        node=PT1,
        is_objective="min",
        name="Maximum Deficit"
    )
    recorderIndex['Maximum Deficit'] = model.recorders.__len__() - 1

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

    # InstanstaneousDeficit
    InstantaneousDeficictNodeRecorder(
        model,
        node=PT1,
        name="InstanstaneousDeficit"
    )
    recorderIndex['InstanstaneousDeficit'] = model.recorders.__len__() - 1

    # check model validity
    model.check_graph()  # check the connectivity of the graph
    model.check()  # check the validity of the model

    return model



def Simulation_Caller(vars):
    '''
    Purpose: Borg calls this function to run the simulation model and return multi-objective performance.

    Note: You could also just put your simulation/function evaluation code here.

    Args:
        vars: A list of decision variable values from Borg

    Returns:
        performance: policy's simulated objective values. A list of objective values, one value each of the objectives.
    '''

    borg_vars = vars  # Decision variable values from Borg

    # Reformat decision variable values as necessary (.e.g., cast borg output parameters as array for use in simulation)
    op_policy_params = np.asarray(borg_vars)

    # Call/run simulation model, return multi-objective performance:
    performance = pysedsim.PySedSim(decision_vars=op_policy_params)

    return performance


def Optimization():
    '''

    Purpose: Call this method from command line to initiate simulation-optimization experiment

    Returns:
        --pareto approximate set file (.set) for each random seed
        --Borg runtime file (.runtime) for each random seed

    '''

    import borg as bg  # Import borg wrapper

    parallel = 1  # 1= master-slave (parallel), 0=serial

    # The following are just examples of relevant MOEA specifications. Select your own values.
    nSeeds = 25  # Number of random seeds (Borg MOEA)
    num_dec_vars = 10  # Number of decision variables
    n_objs = 6  # Number of objectives
    n_constrs = 0  # Number of constraints
    num_func_evals = 30000  # Number of total simulations to run per random seed. Each simulation may be a monte carlo.
    runtime_freq = 1000  # Interval at which to print runtime details for each random seed
    decision_var_range = [[0, 1], [4, 6], [-1, 4], [1, 2], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]
    epsilon_list = [50000, 1000, 0.025, 10, 13, 4]  # Borg epsilon values for each objective

    # Where to save seed and runtime files
    main_output_file_dir = 'E:\output_directory'  # Specify location of output files for different seeds
    os_fold = Op_Sys_Folder_Operator()  # Folder operator for operating system
    output_location = main_output_file_dir + os_fold + 'sets'

    # If using master-slave, start MPI. Only do once.
    if parallel == 1:
        bg.Configuration.startMPI()  # start parallelization with MPI

    # Loop through seeds, calling borg.solve (serial) or borg.solveMPI (parallel) each time
    for j in range(nSeeds):
        # Instantiate borg class, then set bounds, epsilon values, and file output locations
        borg = bg.Borg(num_dec_vars, n_objs, n_constrs, Simulation_Caller)
        borg.setBounds(*decision_var_range)  # Set decision variable bounds
        borg.setEpsilons(*epsilon_list)  # Set epsilon values
        # Runtime file path for each seed:
        runtime_filename = main_output_file_dir + os_fold + 'runtime_file_seed_' + str(j + 1) + '.runtime'
        if parallel == 1:
            # Run parallel Borg
            result = borg.solveMPI(maxEvaluations='num_func_evals', runtime=runtime_filename, frequency=runtime_freq)

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

            print("Seed %s complete") % j

    if parallel == 1:
        bg.Configuration.stopMPI()  # stop parallel function evaluation process


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
#
# thresh = np.ones(num_DP) * -0.84
# acts = np.ones(num_DP) * 30
#
# m__ = make_model2(threshold_vals=thresh, action_vals=acts)
# m__.run()
