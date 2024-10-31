#%% Tips for running this code
"""
**Description**

Uses a wrapper function to optimize the thresholds and actions in drought management policies via DPS, using Pywr
strictly for simulation of plausible policies! Optimization leverages BorgMOEA.

"""
#%% Imports

# general package imports
import datetime
import pandas as pd
import tables
import pandas
import numpy as np
import matplotlib.pyplot as plt
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
exec(open("example_sim_opt.py").read())
# local imports
#from MAIPO_parameters import *
#from MAIPO_searcher import *
#from example_sim_opt import *

#%% Define custom Pywr parameter classes

class FakeYearIndexParameter(IndexParameter):
    """
        Spanish:
        Clasifica una fecha cualquiera, situandola en alguno de losParameterThresholdParameter 6 periodos de planes de desarrollo
        Entrega un entero entre 0 y 7 donde 0 es antes de la semana 14 del 2020 y de 1 a 7 es segun PD

        English:
        Classify any date, placing it in one of the 6 periods of development plans
        Returns an integer between 0 and 7 where 0 is before week 14 of 2020 and 1 to 7 is according to PD.
    """
    def __init__(self, model, dates, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.dates = dates

    def index(self, timestep, scenario_index):

        for i, date in enumerate(self.dates):
            if timestep.datetime < date:
                return i
        else:
            raise ValueError('Simulation date "{}" is at or beyond final date "{}"'.format(timestep.datetime, self.dates[-1]))

    @classmethod
    def load(cls, model, data):

        dates = [pandas.to_datetime(d) for d in data.pop("dates")]
        return cls(model, dates, **data)


FakeYearIndexParameter.register()


class StickyTrigger(Parameter):
    """
        Spanish:
        Evalua si esta activo el contrato en esos 6 meses especificos. En caso de estarlo, entrega el valor del contrato

        English:
        Evaluate if the contract is active in those 6 specific months. If so, deliver the value of the contract

    """

    def __init__(self, model, storage_node, thresholds, contracts, **kwargs):
            super().__init__(model, **kwargs)
            for param in thresholds.values():
                param.parents.add(self)
            self.thresholds = thresholds

            for param in contracts.values():
                if isinstance(param, Parameter):
                    self.parents.add(param)
            self.contracts = contracts
            self.storage_node = storage_node

          # Internal state
            self._outcome = 0  # default not triggered
            self._last_week_evaluated = None
            self._last_year_evaluated = None

    def value(self, ts, si):
            week_no = ts.index % 52
            week_no += 1
            year_no = ts.index // 52
            year = self.model.timestepper.start.year + year_no

            try:
                threshold_parameter = self.thresholds[week_no]
            except KeyError:
                return self._outcome
            else:
                # There is a threshold for this week
                if self._last_week_evaluated == week_no and self._last_year_evaluated == year:
                    # We've already evaluated this week
                    return self._outcome
                else:
                    # We need to evaluate the trigger for this threshold
                    current_volume = self.storage_node.volume[si.global_id]
                    threshold = threshold_parameter.get_value(si)
                    if current_volume < threshold:
                        contract_parameter = self.contracts[week_no]
                        contract_size = contract_parameter.get_value(si)
                        self._outcome = contract_size
                    else:
                        self._outcome = 0
                    self._last_week_evaluated = week_no
                    self._last_year_evaluated = year
                    return self._outcome

    @classmethod
    def load(cls, model, data):
        thresholds = {int(k): load_parameter(model, v) for k, v in data.pop('thresholds').items()}
        contracts = {int(k): load_parameter(model, v) for k, v in data.pop('contracts').items()}
        storage_node = model._get_node_from_ref(model, data.pop('storage_node'))
        return cls(model, storage_node, thresholds, contracts)


StickyTrigger.register()


class AccumulatedIndexedArrayParameter(Parameter):
    """
        Spanish:
        Guarda la cantidad de derechos comprados en los planes de desarrollo anteriores

        English:
        Save the amount of rights purchased in previous development plans

    """
    def __init__(self, model, index_parameter, params, **kwargs):
        super().__init__(model, **kwargs)
        assert(isinstance(index_parameter, IndexParameter))
        self.index_parameter = index_parameter
        self.children.add(index_parameter)

        self.params = []
        for p in params:
            if not isinstance(p, Parameter):
                from pywr.parameters import ConstantParameter
                p = ConstantParameter(model, p)
            self.params.append(p)

        for param in self.params:
            self.children.add(param)
        self.children.add(index_parameter)

    def value(self, timestep, scenario_index):
        """Returns the value of the Parameter at the current index"""
        index = self.index_parameter.get_index(scenario_index)
        value = 0
        for i in range(index):
            value += self.params[i].get_value(scenario_index)
        return value

    @classmethod
    def load(cls, model, data):
        index_parameter = load_parameter(model, data.pop("index_parameter"))
        try:
            parameters = data.pop("params")
        except KeyError:
            parameters = data.pop("parameters")
        parameters = [load_parameter(model, parameter_data) for parameter_data in parameters]
        return cls(model, index_parameter, parameters, **data)


AccumulatedIndexedArrayParameter.register()


"""
To generalize demand restrictions: will make a generalized version of control curves. Instead of a
storage node, we will be able to pass in any parameter. We'll need a way to clarify exactly what
value or property is being accessed (like ideally, they'd have consistent behavior with a regular
control curve when applied with a storage node's level as an indicator). We could pass in a property
and a getter
"""

class IndicatorParameter(Parameter):
    """
    Parameter that combines others into a single "indicator" to use with indicator control curves
    """
    def __init__(self, model, indicator_list, indicator_value_getter, **kwargs):
        """
        Parameters
        ----------
        indicator_list : list of params and nodes
            The curve we use is based on the values of these indicators
        indicator_value_getter : function, takes in indicator_list, timestep, scenario_index
            Returns the relevant value of the indicators
            Can be a getter method, an aggregate of indicator properties, etc.
        gen_control_curves : iterable of Parameter objects or single Parameter
            The Parameter objects to use as a control curve(s).
        """
        super(IndicatorParameter, self).__init__(model, **kwargs)

        if indicator_list is None:
            raise ValueError("indicator_list is required")

        for obj in indicator_list:
            if isinstance(obj, Parameter):
                self.children.add(obj)

        self._indicator_list = indicator_list
        self._indicator_value_getter = indicator_value_getter

    def value(self, timestep, scenario_index):
        indicator_list = self._indicator_list
        indicator_value_getter = self._indicator_value_getter
        return indicator_value_getter(indicator_list, timestep, scenario_index)

    @classmethod
    def load(cls, model, data):
        """
        TECHNICALLY, WE CAN ALLOW THE USER TO ADD A CODE STRING TO THE JSON FILE AND RUN ANY
        OPERATION THEY WANT ON THE INDICATORS USING "eval" OR "exec". THESE RAISE MAJOR
        SECURITY CONCERNS THOUGH. I WILL NOT IMPLEMENT THIS FUNCTION FOR NOW AND ASSUME THE
        USER WILL ONLY USE THIS PARAMETER WITH make_model AND NOT JSON

        ONE OPTION TO MAKE IT EASIER TO USE AND WORK FOR JSON: CAN EXTEND AGGREGATED PARAMS
        TO TAKE IN STORAGE NODES
        """
        raise AttributeError("IndicatorParameters can only be run with Python, not JSON")

class BaseIndicatorControlCurveParameter(Parameter):
    """
    Generalized base class for all Parameters that rely on the indicator containing a control_curve Parameter
    """
    def __init__(self, model, indicator, control_curves, **kwargs):
        """
        Parameters
        ----------
        indicator : Parameter
            The indicator we use to choose which control curve to use
        control_curves : iterable of Parameter objects or single Parameter
            The Parameter objects to use as a control curve(s).
        """
        super(BaseIndicatorControlCurveParameter, self).__init__(model, **kwargs)
        if indicator is None:
            raise ValueError("indicator is required")
        self.children.add(indicator)
        for control_curve in control_curves:
            self.children.add(control_curve)
        self._indicator = indicator
        self._control_curves = control_curves

    @property
    def control_curves(self):
        return self._control_curves

    @control_curves.setter
    def control_curves(self, control_curves):
        # Accept a single Parameter and convert to a list internally
        if isinstance(control_curves, Parameter):
            control_curves = [control_curves]

        # remove existing control curves (if any)
        if self._control_curves is not None:
            for control_curve in self._control_curves:
                control_curve.parents.remove(self)

        _new_control_curves = []
        for control_curve in control_curves:
            # Accept numeric inputs and convert to `ConstantParameter`
            if isinstance(control_curve, (float, int)):
                control_curve = ConstantParameter(self.model, control_curve)

            control_curve.parents.add(self)
            _new_control_curves.append(control_curve)
        self._control_curves = list(_new_control_curves)

    @property
    def indicator(self):
        return self._indicator

    @indicator.setter
    def indicator(self, value):
        if not isinstance(value, IndicatorParameter):
            raise ValueError("Please use an IndicatorParameter")
        self._indicator = value

    @classmethod
    def _load_control_curves(cls, model: object, data: object) -> object:
        """ Private class method to load gen control curve data from dict. """

        control_curves = []
        if 'control_curve' in data:
            control_curves.append(load_parameter(model, data.pop('control_curve')))
        elif 'control_curves' in data:
            for pdata in data.pop('control_curves'):
                control_curves.append(load_parameter(model, pdata))
        return control_curves

    @classmethod
    def _load_indicator(cls, model, data):
        """ Private class method to load indicator from dict. """

        indicator = data.pop('indicator')
        if not isinstance(indicator, Parameter):
            raise ValueError("Please use a Parameter as the indicator")
        indicator = load_parameter(model, data.pop('indicator'))
        return indicator


BaseIndicatorControlCurveParameter.register()


class IndicatorControlCurveParameter(BaseIndicatorControlCurveParameter):
    """ A generic multi-levelled generalized control curve Parameter.

     This parameter can be used to return different values when an indicator list's current
      value is different relative to predefined control curves.
     By default this parameter returns an integer sequence from zero if the first control curve
      is passed, and incrementing by one for each control curve (or "level") the inidcator value
      is below.

    Parameters
    ----------
    indicator : Parameter
        Used to determine which control curve to use
    control_curves : `float`, `int` or `Parameter` object, or iterable thereof
        The position of the control curves. Internally `float` or `int` types are cast to
        `ConstantParameter`. Multiple values correspond to multiple control curve positions.
        These should be specified in descending order.
    values : array_like or `None`, optional
        The values to return if the indicator object is above the correspond control curve.
        I.e. the first value is returned if the indicator value is above the first control curve,
        and second value if above the second control curve, and so on. The length of `values`
        must be one more than the length of `control_curves`.
    parameters : iterable `Parameter` objects or `None`, optional
        If `values` is `None` then `parameters` can specify a `Parameter` object to use at level
        of the control curves. In the same way as `values` the first `Parameter` is used if
        `Storage` is above the first control curve, and second `Parameter` if above the
        second control curve, and so on.
    variable_indices : iterable of ints, optional
        A list of indices that correspond to items in `values` which are to be considered variables
         when `self.is_variable` is True. This mechanism allows a subset of `values` to be variable.
    lower_bounds, upper_bounds : array_like, optional
        Bounds of the variables. The length must correspond to the length of `variable_indices`, i.e.
         there are bounds for each index to be considered as a variable.

    Notes
    -----
    If `values` and `parameters` are both `None`, the default, then `values` defaults to
     a range of integers, starting at zero, one more than length of `control_curves`.

    See also
    --------
    BaseIndicatorControlCurveParameter
    """
    def __init__(self, model, indicator, control_curves, values=None, parameters=None,
                 variable_indices=None, upper_bounds=None, lower_bounds=None, **kwargs):
        super(IndicatorControlCurveParameter, self).__init__(model, indicator, control_curves, **kwargs)
        # Expected number of values is number of control curves plus one.
        nvalues = len(self.control_curves) + 1
        self.parameters = None
        if values is not None:
            if len(values) != nvalues:
                raise ValueError('Length of values should be one more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self.values = values
        elif parameters is not None:
            if len(parameters) != nvalues:
                raise ValueError('Length of parameters should be one more than the number of '
                                 'control curves ({}).'.format(nvalues))
            self.parameters = list(parameters)
            # Make sure these parameters depend on this parameter to ensure they are evaluated
            # in the correct order.
            for p in self.parameters:
                p.parents.add(self)
        else:
            # No values or parameters given, default to sequence of integers
            self.values = np.arange(nvalues)

        # Default values
        self._upper_bounds = None
        self._lower_bounds = None

        if variable_indices is not None:
            self.variable_indices = variable_indices
            self.double_size = len(variable_indices)
        else:
            self.double_size = 0
        # Bounds for use as a variable (i.e. when self.is_variable = True)
        if upper_bounds is not None:
            if self.values is None or variable_indices is None:
                raise ValueError('Upper bounds can only be specified if `values` and `variable_indices` '
                                 'is not `None`.')
            if len(upper_bounds) != self.double_size:
                raise ValueError('Length of upper_bounds should be equal to the length of `variable_indices` '
                                 '({}).'.format(self.double_size))
            self._upper_bounds = np.array(upper_bounds)

        if lower_bounds is not None:
            if self.values is None or variable_indices is None:
                raise ValueError('Lower bounds can only be specified if `values` and `variable_indices` '
                                 'is not `None`.')
            if len(lower_bounds) != self.double_size:
                raise ValueError('Length of lower_bounds should be equal to the length of `variable_indices` '
                                 '({}).'.format(self.double_size))
            self._lower_bounds = np.array(lower_bounds)

        self.children.add(indicator)
        for control_curve in control_curves:
            self.children.add(control_curve)

    @property
    def values(self):
        return np.asarray(self._values)

    @values.setter
    def values(self, values):
        self._values = np.asarray(values, dtype=np.float64)

    @property
    def variable_indices(self):
        return np.array(self._variable_indices)

    @variable_indices.setter
    def variable_indices(self, values):
            self._variable_indices = np.array(values, dtype=np.int32)

    @classmethod
    def load(cls, model, data):
        control_curves = super(IndicatorControlCurveParameter, cls)._load_control_curves(model, data)
        indicator = super(IndicatorControlCurveParameter, cls)._load_indicator(model, data)

        parameters = None
        values = None
        if 'values' in data:
            values = load_parameter_values(model, data)
        elif 'parameters' in data:
            # Load parameters
            parameters_data = data.pop('parameters')
            parameters = []
            for pdata in parameters_data:
                parameters.append(load_parameter(model, pdata))

        return cls(model, indicator, control_curves, values=values, parameters=parameters, **data)

    def value(self, ts, scenario_index):
        indicator = self.indicator
        indicator_value = indicator.value(ts, scenario_index)

        # Assumes control_curves is sorted highest to lowest
        for j, cc_param in enumerate(self.control_curves):
            cc = cc_param.get_value(scenario_index)
            # If level above control curve then return this level's value
            if indicator_value >= cc:
                if self.parameters is not None:
                    param = self.parameters[j]
                    return param.get_value(scenario_index)
                else:
                    return self._values[j]

        if self.parameters is not None:
            param = self.parameters[-1]
            return param.get_value(scenario_index)
        else:
            return self._values[-1]

    def set_double_variables(self, values):
        if len(values) != len(self.variable_indices):
            raise ValueError('Number of values must be the same as the number of variable_indices.')

        if self.double_size != 0:
            for i, v in zip(self.variable_indices, values):
                self._values[i] = v

    def get_double_variables(self):
        arry = np.empty((len(self.variable_indices), ))
        for i, j in enumerate(self.variable_indices):
            arry[i] = self._values[j]
        return arry

    def get_double_lower_bounds(self):
        return self._lower_bounds

    def get_double_upper_bounds(self):
        return self._upper_bounds


IndicatorControlCurveParameter.register()


class IndicatorControlCurveIndexParameter(IndexParameter):
    """Multiple control curve holder which returns an index not a value

    Parameters
    ----------
    indicator : `Parameter`
    control_curves : iterable of `Parameter` instances or floats
    """
    def __init__(self, model, indicator, control_curves, **kwargs):
        super(IndicatorControlCurveIndexParameter, self).__init__(model, **kwargs)
        for control_curve in control_curves:
            self.children.add(control_curve)
        self.children.add(indicator)
        self.indicator = indicator
        self._control_curves = control_curves

    @property
    def control_curves(self):
        return self._control_curves

    @control_curves.setter
    def control_curves(self, control_curves):
        # Accept a single Parameter and convert to a list internally
        if isinstance(control_curves, Parameter):
            control_curves = [control_curves]

        # remove existing control curves (if any)
        if self._control_curves is not None:
            for control_curve in self._control_curves:
                control_curve.parents.remove(self)

        _new_control_curves = []
        for control_curve in control_curves:
            # Accept numeric inputs and convert to `ConstantParameter`
            if isinstance(control_curve, (float, int)):
                control_curve = ConstantParameter(self.model, control_curve)

            control_curve.parents.add(self)
            _new_control_curves.append(control_curve)
        self._control_curves = list(_new_control_curves)

    def index(self, timestep, scenario_index):
        """Returns the index of the first control curve the storage is above

        The index is zero-based. For example, if only one control curve is
        supplied then the index is either 0 (above) or 1 (below). For two
        curves the index is either 0 (above both), 1 (in between), or 2 (below
        both), and so on.
        """
        current_value = self.indicator.value(timestep, scenario_index)
        index = len(self.control_curves)
        for j, control_curve in enumerate(self.control_curves):
            target_value = control_curve.get_value(scenario_index)
            if current_value >= target_value:
                index = j
                break
        return index

    @classmethod
    def load(cls, model, data):
        indicator = load_parameter(model, data.pop("indicator"))
        control_curves = [load_parameter(model, d) for d in data.pop("control_curves")]
        return cls(model, indicator, control_curves, **data)


IndicatorControlCurveIndexParameter.register()

class DroughtStatusAggregationParameter(Parameter):
    # Aggregates values of a dataframe (typically an indicator) over a time period to get a single value, used to determine
    # whether to apply a policy such as buying contracts
    # num_weeks is the number of weeks in the past we look (1 is most recent week only)

    # DESCRIPTION FROM DATAFRAMEPARAMETER, TO MODIFY:
    """Timeseries parameter with automatic alignment and resampling

    Parameters
    ----------
    model : pywr.model.Model
    dataframe : pandas.DataFrame or pandas.Series
    scenario: pywr._core.Scenario (optional)
    timestep_offset : int (default=0)
        Optional offset to apply to the timestep look-up. This can be used to look forward (positive value) or
        backward (negative value) in the dataset. The offset is applied to dataset after alignment and resampling.
        If the offset takes the indexing out of the data bounds then the parameter will return the first or last
        value available.
    """

    def __init__(self, model, dataframe, agg_func, num_weeks=1, scenario=None, timestep_offset=0, **kwargs):
        super(DroughtStatusAggregationParameter, self).__init__(model, *kwargs)
        self.dataframe = dataframe
        self.agg_func = agg_func
        self.num_weeks = num_weeks
        self.scenario = scenario
        self.timestep_offset = timestep_offset

    def setup(self):
        # i
        super(DroughtStatusAggregationParameter, self).setup()
        # align and resample the dataframe (function from pywr.dataframe_tools.py -- lines up indices with timestepper)
        dataframe_resampled = align_and_resample_dataframe(self.dataframe, self.model.timestepper.datetime_index)
        if dataframe_resampled.ndim == 1:
            dataframe_resampled = pandas.DataFrame(dataframe_resampled)
        # dataframe should now have the correct number of timesteps for the model
        if len(dataframe_resampled) != len(self.model.timestepper):
            raise ValueError("Aligning DataFrame failed with a different length compared with model timesteps.")
        # check that if a 2D DataFrame is given that we also have a scenario assigned with it
        if dataframe_resampled.ndim == 2 and dataframe_resampled.shape[1] > 1:
            if self.scenario is None:
                raise ValueError("Scenario must be given for a DataFrame input with multiple columns.")
            if self.scenario.size != dataframe_resampled.shape[1]:
                raise ValueError("Scenario size ({}) is different to the number of columns ({}) "
                                 "in the DataFrame input.".format(self.scenario.size, dataframe_resampled.shape[1]))

        if self.scenario is not None:
            self._scenario_index = self.model.scenarios.get_scenario_index(self.scenario)
            # if possible, only load the data required
            scenario_indices = None
            # Default to index that is just out of bounds to cause IndexError if something goes wrong
            self._scenario_ids = np.ones(self.scenario.size, dtype=np.int32) * self.scenario.size

            # Calculate the scenario indices to load dependning on how scenario combinations are defined.
            if self.model.scenarios.user_combinations:
                scenario_indices = set()
                for user_combination in self.model.scenarios.user_combinations:
                    scenario_indices.add(user_combination[self._scenario_index])
                scenario_indices = sorted(list(scenario_indices))
            elif self.scenario.slice:
                scenario_indices = range(*self.scenario.slice.indices(self.scenario.slice.stop))
            else:
                # scenario is defined, but all data required
                self._scenario_ids = None
            if scenario_indices is not None:
                # Now load only the required data
                for n, i in enumerate(scenario_indices):
                    self._scenario_ids[i] = n
                dataframe_resampled = dataframe_resampled.iloc[:, scenario_indices]

        self._values = dataframe_resampled.values.astype(np.float64)


    # def minus1onexception(func):
    #     def Inner_Function(*args, **kwargs):
    #         try:
    #             func(*args, **kwargs)
    #         except Exception:
    #             return -1
    #
    #     return Inner_Function
    #
    # @minus1onexception
    def value(self, timestep, scenario_index):
        # value
        first_week = min(max(timestep.index + self.timestep_offset - (self.num_weeks - 1), 0), self._values.shape[0] - 1)
        last_week = min(max(timestep.index + self.timestep_offset, 0), self._values.shape[0] - 1)
        # j

        if self.scenario is not None:
            j = scenario_index.indices[self._scenario_index]
            if self._scenario_ids is not None:
                j = self._scenario_ids[j]
            values = self._values[first_week:last_week + 1, j]
        else:
            values = self._values[first_week:last_week + 1, 0]

        value = self.agg_func(values)
        return value


    @classmethod
    def load(cls, model, data):
        scenario = data.pop('scenario', None)
        if scenario is not None:
            scenario = model.scenarios[scenario]
        timestep_offset = data.pop('timestep_offset', 0)
        # This will consume all keyword arguments silently in pandas. I.e. don't rely on **data passing keywords
        df = load_dataframe(model, data)
        return cls(model, df, scenario=scenario, timestep_offset=timestep_offset, **data)


DroughtStatusAggregationParameter.register()


class DroughtStatusMinAggregationParameter(DroughtStatusAggregationParameter):
    def __init__(self, model, dataframe, num_weeks=1, scenario=None, timestep_offset=0, **kwargs):
        super(DroughtStatusMinAggregationParameter, self).__init__(model, dataframe, lambda x: np.min(x), num_weeks, scenario, timestep_offset, **kwargs)

DroughtStatusMinAggregationParameter.register()


class ReservoirCostRecorder (Recorder):
    """
        Spanish:
        Guarda el costo de construir el embalse, dependiendo de cuando se construye, considerando un periodo de DP antes

        English:
        Saves the cost of building the reservoir, depending on when it is built, considering a DP period before
    """

    def __init__(self, model, construction_dp, capacity, unit_costs, discount_rate, fixed_cost, **kwargs):
        super().__init__(model, **kwargs)
        self.construction_dp = construction_dp
        self.capacity = capacity
        self.children.add(construction_dp)
        self.children.add(capacity)
        self.discount_rate = discount_rate
        self._costs = None

    def finish(self):

        costs = np.zeros(len(self.model.scenarios.combinations))

        capacity = self.capacity.get_all_values()
        construction_dp = np.floor(self.construction_dp.get_all_values())

        for i in range(len(costs)):
            if construction_dp[i] > 6:
                costs[i] = 0
            else:
                decision_dp = int(construction_dp[i]) - 1
                discount_factor = 1 / (1 + self.discount_rate)**((decision_dp-1)*5)
                costs[i] = (335748.31*capacity[i]**0.342)*discount_factor

        self._costs = costs

    def values(self):
        return self._costs


    @classmethod
    def load(cls, model, data):

        construction_dp = load_parameter(model, data.pop("construction_dp"))
        capacity = load_parameter(model, data.pop("capacity"))
        return cls(model, construction_dp, capacity, **data)


ReservoirCostRecorder.register()


class PurchasesCostRecorder (Recorder):
    """
        Spanish:
        Guarda el costo total de todas las compras de derechos

        English:
        Save the total cost of all rights purchases
    """

    def __init__(self, model, purchases_value, meanflow, discount_rate, coeff, **kwargs):
        super().__init__(model, **kwargs)
        self.purchases_value = purchases_value
        self.children.add(purchases_value)
        self.meanflow = meanflow
        self.children.add(meanflow)
        self.coeff = coeff
        self.discount_rate = discount_rate
        self._costs = None

    def finish(self):

        costs = np.zeros(len(self.model.scenarios.combinations))

        start_year = self.model.timestepper.start.year
        mean_flows = self.meanflow.data

        for dp_index, purchase in enumerate(self.purchases_value.params):
            if dp_index == 0:
                continue
            #self.model.timestepper.end < pandas.to_datetime(d) for d in data.pop("dates")

            shares = purchase.get_all_values()
            #timestep_index = (2020 - start_year)*52 + (dp_index - 1) * 52 * 5  # contains an offset index if start year is 2004
            timestep_index = (dp_index - 1) * 52 * 5

            # added to allow for different time periods to be calculated


            discount_factor = 1 / (1 + self.discount_rate)**((dp_index-1)*5)

            for i in range(len(costs)):
                try:
                    mean_flow = mean_flows[timestep_index, i]
                except:
                    costs[i] += 0
                else:
                    if shares[i] > 0:
                        share_cost = 2.71828182846**(12.46382-0.7639182*np.log(shares[i])+1.516715*np.log(mean_flow))
                    else: share_cost = 0
                    costs[i] += shares[i] * share_cost * discount_factor

        self._costs = costs

    def values(self):
        return self._costs

    @classmethod
    def load(cls, model, data):

        purchases_value = load_parameter(model, data.pop("purchases"))
        mean_flow = load_recorder(model, data.pop("mean_flow"))

        return cls(model, purchases_value, mean_flow, **data)


PurchasesCostRecorder.register()


class SeasonRollingMeanFlowNodeRecorder(NodeRecorder):
    """Records the mean flow of a Node for the previous N timesteps for a specific season (april to october or october to april)

    Parameters
    ----------
    model : `pywr.core.Model`
    node : `pywr.core.Node`
        The node to record
    timesteps : int
        The number of timesteps to calculate the mean flow for
    name : str (optional)
        The name of the recorder

    """
    def __init__(self, model, node, first_week, last_week, years, **kwargs):
        super().__init__(model, node, **kwargs)
        self.first_week = first_week
        self.last_week = last_week
        self.years = years
        self.data = None

    def reset(self):
        super().reset()
        self.position = 0
        self.data = np.empty([len(self.model.timestepper), len(self.model.scenarios.combinations)])
        self._memory = np.zeros([len(self.model.scenarios.combinations), (self.last_week-self.first_week)*self.years])
        self.passed_weeks = 0

    def after(self):
        # calculates the week
        ts = self.model.timestepper.current
        week_no = ts.index % 52
        week_no += 1
        year_no = ts.index // 52
        year = self.model.timestepper.start.year + year_no
        if self.first_week <= week_no < self.last_week:
            # save today's flow
            for i in range(0, self._memory.shape[0]):
                self._memory[i, self.position] = self.node.flow[i]
            self.passed_weeks += 1
            # calculate the mean flow
            n = self._memory.shape[1]
            if self.passed_weeks < n:
                n = self.passed_weeks
            # save the mean flow
            mean_flow = np.mean(self._memory[:, 0:n], axis=1)
            self.data[ts.index, :] = mean_flow
            # prepare for the next timestep
            self.position += 1
            if self.position >= self._memory.shape[1]:
                self.position = 0

    @classmethod
    def load(cls, model, data):
        node = model._get_node_from_ref(model, data.pop("node"))
        return cls(model, node, **data)


SeasonRollingMeanFlowNodeRecorder.register()


class ContractCostRecorder (Recorder):
    """
        Spanish:
        Entrega el costo total relativo a los contratos segun fueron activados

        English:
        Delivers the total cost relative to the contracts as they were activated
    """
    def __init__(self, model, contract_value, meanflow, purchases_value, discount_rate, max_cost, gradient, coeff, week_no, total_shares = 8133, **kwargs):
        super().__init__(model, **kwargs)
        self.meanflow = meanflow
        self.children.add(meanflow)
        self.contract_value = contract_value
        self.children.add(contract_value)
        self.purchases_value = purchases_value
        self.children.add(purchases_value)
        self.discount_rate = discount_rate
        self.max_cost = max_cost
        self.gradient = gradient
        self.coeff = coeff
        self.week_no = week_no
        self.total_shares = total_shares
        self._costs = None

    def reset(self):
        self._costs = np.zeros(len(self.model.scenarios.combinations))

    def after(self):
        # calculates the week
        ts = self.model.timestepper.current
        week_no = ts.index % 52
        week_no += 1
        year_no = ts.index // 52
        #year = self.model.timestepper.start.year + year_no

        if week_no != self.week_no:
            return

        costs = self._costs
        shares = self.contract_value.get_all_values()
        purchases = self.purchases_value.get_all_values()
        mean_flows = self.meanflow.data
        m = self.gradient
        n = self.max_cost

        discount_factor = 1 / (1 + self.discount_rate)**(year_no)

        interruptor = 1 #por mientras

        for i in range(len(costs)):
            if interruptor == 1:
                c = 262.273/0.08
                costs[i] += c*discount_factor
            else:
                f = mean_flows[ts.index, i]*26
                K = shares[i]
                p = purchases[i]
                c = (m*K*f/8133 * (f*K/(2*8133) + n/m - (8133-1917-p)*f/8133))
                costs[i] += c*discount_factor


    def values(self):
        return self._costs

    @classmethod
    def load(cls, model, data):
        meanflow = load_recorder(model, data.pop("mean_flow"))
        contract_value = load_parameter(model, data.pop("contract"))
        purchases_value = load_parameter(model, data.pop("purchases"))
        return cls(model, contract_value, meanflow, purchases_value, **data)


ContractCostRecorder.register()

def CustomizedAggregation(model, weights):
# Apply weighted aggregation function
    def weighted_agg_func(values):
        return np.dot(weights, values)
    model.recorders['deficit PT1'].agg_func = weighted_agg_func

    threshold = model.recorders["InstanstaneousDeficit"]
    events = EventRecorder(model, threshold, name = "deficit_event")
    duration = EventDurationRecorder(model, events, name = "Max Deficit Duration", recorder_agg_func = "max", agg_func = "mean", is_objective = "min")


class MaximumDeficitNodeRecorder(NodeRecorder):
    """
    Recorder the maximum difference between modelled flow and max_flow for a Node
    """
    def reset(self):
        self._values = np.zeros(len(self.model.scenarios.combinations))

    def after(self):
        ts = self.model.timestepper.current
        days = self.model.timestepper.current.days
        node = self.node
        values = self._values

        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)
            deficit = (max_flow - node.flow[scenario_index.global_id])*days
            if deficit > values[scenario_index.global_id]:
                values[scenario_index.global_id] = deficit

    def values(self):
        return self._values


MaximumDeficitNodeRecorder.register()


class InstantaneousDeficictNodeRecorder(NodeRecorder):
    """
    Recorder the maximum difference between modelled flow and max_flow for a Node
    """
    def reset(self):
        self._values = np.zeros(len(self.model.scenarios.combinations))

    def after(self):
        ts = self.model.timestepper.current
        days = self.model.timestepper.current.days
        node = self.node
        values = self._values

        for scenario_index in self.model.scenarios.combinations:
            max_flow = node.get_max_flow(scenario_index)
            deficit = (max_flow - node.flow[scenario_index.global_id])*days
            if deficit < 1e-4:
                deficit = 0
            values[scenario_index.global_id] = deficit

    def values(self):
        return self._values


InstantaneousDeficictNodeRecorder.register()


class ShortageCostRecorder(Recorder):
    """
    Saves domestic water shortage costs depending on urban water deficits
    """

    def __init__(self, model, unit_cost, deficits_value, discount_rate, **kwargs):
        super().__init__(model, **kwargs)
        self.unit_cost = unit_cost
        self.deficits_value = deficits_value
        self.children.add(deficits_value)
        self.discount_rate = discount_rate
        self._costs = None

    def reset(self):
        self._costs = np.zeros(len(self.model.scenarios.combinations))

    def after(self):
        ts = self.model.timestepper.current
        year_no = ts.index // 52

        costs = self._costs
        deficits = self.deficits_value.get_all_values()

        discount_factor = 1 / (1 + self.discount_rate)**(year_no)

        for i in range(len(costs)):
            d = deficits[i]
            c = self.unit_cost*d
            costs[i] += c*discount_factor

    def values(self):
        return self._costs


ShortageCostRecorder.register()


class PolicyTrigger(Parameter):
    """
        English:
        Evaluate if the contract is active in those 6 specific months. If so, deliver the value of the contract

    """

    def __init__(self, model, drought_status, thresholds, contracts, **kwargs):
        super().__init__(model, **kwargs)
        for param in thresholds.values():
            param.parents.add(self)
        self.thresholds = thresholds

        for param in contracts.values():
            param.parents.add(self)
        self.contracts = contracts
        self.drought_status = drought_status

        # Internal state
        #self._outcome = 0  # default not triggered
        #self._last_week_evaluated = None
        #self._last_year_evaluated = None

    def setup(self):
        # allocate an array to hold the parameter state
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self._outcome = np.empty([num_scenarios], np.float64)
        self._last_week_evaluated = np.empty([num_scenarios], np.int64)
        self._last_year_evaluated = np.empty([num_scenarios], np.int64)


    def reset(self):
        # reset the amount remaining in all states to the initial value
        self._outcome[...] = 0
        self._last_week_evaluated[...] = -1
        self._last_year_evaluated[...] = -1

    def value(self, timestep, scenario_index):
        gid = scenario_index.global_id
        week_no = timestep.index % 52
        week_no += 1
        year_no = timestep.index // 52
        year = self.model.timestepper.start.year + year_no

        try:
            threshold_parameter = self.thresholds[week_no]
        except KeyError:
            return self._outcome[gid]
        else:
            # There is a threshold for this week
            if self._last_week_evaluated[gid] == week_no and self._last_year_evaluated[gid] == year:
                # We've already evaluated this week
                return self._outcome[gid]
            else:
                # We need to evaluate the trigger for this threshold
                current_drought = self.drought_status.value(timestep, scenario_index)

                # threshold = threshold_parameter.get_value(scenario_index)
                threshold = threshold_parameter.value(timestep, scenario_index)
                if current_drought < threshold:
                    contract_parameter = self.contracts[week_no]
                    contract_size = contract_parameter.get_value(scenario_index)
                    self._outcome[gid] = contract_size
                else:
                    self._outcome[gid] = 0
                self._last_week_evaluated[gid] = week_no
                self._last_year_evaluated[gid] = year
                return self._outcome[gid]

    @classmethod
    def load(cls, model, data):
        thresholds = {int(k): load_parameter(model, v) for k, v in data.pop('thresholds').items()}
        contracts = {int(k): load_parameter(model, v) for k, v in data.pop('contracts').items()}
        drought_status = load_parameter(model, 'drought_index')
        return cls(model, drought_status, thresholds, contracts)


PolicyTrigger.register()


# Use this class for initial model development (next step: non-hard coded policies for optimization)
class PolicyTreeTriggerHardCoded(Parameter):
    """
        English:
        Evaluate if the contract is active in those 6 specific months. If so, deliver the value of the contract.
        We hard code the policy alternatives for simulation results for AGU.

        (a) policy 1 (baseline):
            {if <=-0.84, contracts 30}
            {else 0}
        (b) policy 2:
            {if -0.5, contracts 10}
            {if -0.84, contracts 30}
            {if -1.15, contracts 50}
            {else 0}
        (c) policy 3:
            {if -1.15, contracts 10}
            {if -1.5, contracts 30}
            {if -1.85, contracts 50}
            {else 0}
    """

    def __init__(self, model, drought_status, thresholds, contracts, **kwargs):
        super().__init__(model, **kwargs)
        for param in thresholds.values():
            param.parents.add(self)
        self.thresholds = thresholds

        for param in contracts.values():
            param.parents.add(self)
        self.contracts = contracts
        self.drought_status = drought_status

    def setup(self):
        # allocate an array to hold the parameter state
        super().setup()
        num_scenarios = len(self.model.scenarios.combinations)
        self._outcome = np.empty([num_scenarios], np.float64)
        self._last_week_evaluated = np.empty([num_scenarios], np.int64)
        self._last_year_evaluated = np.empty([num_scenarios], np.int64)

    def reset(self):
        # reset the amount remaining in all states to the initial value
        self._outcome[...] = 0
        self._last_week_evaluated[...] = -1
        self._last_year_evaluated[...] = -1

    def value(self, timestep, scenario_index):
        gid = scenario_index.global_id
        week_no = timestep.index % 52
        week_no += 1
        year_no = timestep.index // 52
        year = self.model.timestepper.start.year + year_no

        # week_no = str(week_no)

        try:
            threshold_parameter = self.thresholds[week_no]
        except KeyError:
            return self._outcome[gid]
        else:
            # There is a threshold for this week
            if self._last_week_evaluated[gid] == week_no and self._last_year_evaluated[gid] == year:
                # We've already evaluated this week
                return self._outcome[gid]
            else:
                # We need to evaluate the trigger for this threshold
                current_drought = self.drought_status.value(timestep, scenario_index)
                threshold = threshold_parameter.value(timestep, scenario_index)

                if current_drought < threshold:
                    contract_parameter = self.contracts[week_no]
                    contract_size = contract_parameter.get_value(scenario_index)
                    self._outcome[gid] = contract_size
                    # print(contract_size)
                # else:
                #     self._outcome[gid] = 0

                # self._outcome[gid] = 0

                # if current_drought <= -1.5:
                #     self._outcome[gid] = 0
                # elif current_drought <= -1.5:
                #     self._outcome[gid] = 0
                # elif current_drought <= -1.5:
                #     self._outcome[gid] = 0
                # else:
                #     self._outcome[gid] = 0

                # Policy 1 (baseline):
                # if current_drought <= -0.84:
                #     self._outcome[gid] = 30
                # else:
                #     self._outcome[gid] = 0

                # Policy 2
                # if current_drought <= -1.15:
                #     self._outcome[gid] = 50
                # elif current_drought <= -0.84:
                #     self._outcome[gid] = 30
                # elif current_drought <= -0.5:
                #     self._outcome[gid] = 10
                # else:
                #     self._outcome[gid] = 0

                # Policy 3
                # if current_drought <= -1.85:
                #     self._outcome[gid] = 50
                # elif current_drought <= -1.5:
                #     self._outcome[gid] = 30
                # elif current_drought <= -0.15:
                #     self._outcome[gid] = 10
                # else:
                #     self._outcome[gid] = 0

                self._last_week_evaluated[gid] = week_no
                self._last_year_evaluated[gid] = year
                return self._outcome[gid]

    @classmethod
    def load(cls, model, data):
        thresholds = {int(k): load_parameter(model, v) for k, v in data.pop('thresholds').items()}
        contracts = {int(k): load_parameter(model, v) for k, v in data.pop('contracts').items()}
        drought_status = load_parameter(model, data.pop('drought_status'))
        return cls(model, drought_status, thresholds, contracts)


PolicyTreeTriggerHardCoded.register()


# Output node that can take in restriction
class RestrictedOutput(Output):
    def __init__(self, model, desired_flow, restriction_factor=1, *args, **kwargs):
        self.desired_flow = desired_flow
        self.restriction_factor = restriction_factor
        max_flow = AggregatedParameter(
            model,
            parameters=[
                desired_flow,
                restriction_factor
            ],
            agg_func="product"
        )
        kwargs["model"] = model
        super(RestrictedOutput, self).__init__(max_flow=max_flow, *args, **kwargs)




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
#     if label == "Failure Frequency":
#         plt.ylim(0, 0.3)
#     if label == "Maximum Deficit":
#         plt.ylim(0, 8)
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
# num_k = 1  # number of levels in policy tree
# num_DP = 7  # number of decision periods
#
# contract_thresh = np.ones(num_DP)
# contract_acts = np.ones(num_DP) * 30
# demand_thresh = [
#     -np.ones(12),
#     -2*np.ones(12)
# ]
# demand_acts = [
#     np.ones(12),
#     [0.85, 0.85, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.85, 0.85],
#     [0.75, 0.75, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.75, 0.75]
# ]
#
# m_ = make_model(contract_threshold_vals=contract_thresh, contract_action_vals=contract_acts,
#                 demand_threshold_vals=demand_thresh, demand_action_vals=demand_acts)  # function in example_sim_opt.py
# m_.check()
# m_.check_graph()
# m_.find_orphaned_parameters()
#
# m_.run()
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


#%% Test demand thresholds and actions

num_k = 1  # number of levels in policy tree
num_DP = 7  # number of decision periods

# LINE CHART OF THRESHOLDS FOR EACH SCENARIO

# num_thresh = 10
# thresh_space = 0.84
# acts_space = 30
# ff_PT1 = np.zeros((num_thresh, 15))
# ff_Ag = np.zeros((num_thresh, 15))
# deficit_PT1 = np.zeros((num_thresh, 15))
# deficit_Ag = np.zeros((num_thresh, 15))
# cost = np.zeros((num_thresh, 15))
# for i in range(num_thresh):
#     thresh = -np.ones(num_DP) * thresh_space * i
#     acts = np.ones(num_DP) * acts_space# * (i + 1)
#
#     m_ = make_model(contract_threshold_vals=thresh, contract_action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#     ff_Ag[i, :] = m_.recorders['failure_frequency_Ag'].values()
#     deficit_PT1[i, :] = m_.recorders['Maximum Deficit PT1'].values()
#     deficit_Ag[i, :] = m_.recorders['Maximum Deficit Ag'].values()
#     cost[i, :] = m_.recorders['TotalCost'].values()
#
# ff_fig, ff_ax = plt.subplots(3, 5, figsize=(15, 12))
# deficit_fig, deficit_ax = plt.subplots(3, 5, figsize=(15, 12))
# cost_fig, cost_ax = plt.subplots(3, 5, figsize=(15, 12))
# x_vals = acts_space*np.arange(1, num_thresh + 1)
#
# max_ff = np.maximum(np.max(np.max(ff_PT1)), np.max(np.max(ff_Ag)))
# max_deficit = np.maximum(np.max(np.max(deficit_PT1)), np.max(np.max(deficit_Ag)))
# max_cost = np.max(np.max(cost))
# metric_properties = {
#     "ff_PT1": {
#         "ax": ff_ax,
#         "data": ff_PT1,
#         "max_val": max_ff,
#         "label": "PT1 Failure Frequency",
#         "color": "Purple",
#     },
#
#     "ff_Ag": {
#         "ax": ff_ax,
#         "data": ff_Ag,
#         "max_val": max_ff,
#         "label": "Agriculture Failure Frequency",
#         "color": "Green"
#     },
#
#     "deficit_PT1": {
#         "ax": deficit_ax,
#         "data": deficit_PT1,
#         "max_val": max_deficit,
#         "label": "PT1 Deficit",
#         "color": "Purple"
#     },
#
#     "deficit_Ag": {
#         "ax": deficit_ax,
#         "data": deficit_Ag,
#         "max_val": max_deficit,
#         "label": "Agriculture Deficit",
#         "color": "Green"
#     },
#
#     "cost": {
#         "ax": cost_ax,
#         "data": cost,
#         "max_val": max_cost,
#         "label": "Cost",
#         "color": "Orange"
#     }
# }
#
# def create_line_plot(ax, data, max_val, label, color):
#     ax.plot(x_vals, data, marker='o', label=label, color=color)
#     ax.set_ylim(0, 1.2*max_val)
#     ax.set_xticks(x_vals)
#     ax.set_xticklabels(x_vals, rotation=45)
#
#
# num_ticks = num_thresh
# for i in range(15):
#     for metric, properties in metric_properties.items():
#         create_line_plot(
#             properties["ax"][i // 5, i % 5],
#             properties["data"][:, i],
#             properties["max_val"],
#             properties["label"],
#             properties["color"]
#         )
#
# def format_plot(fig, label):
#     lines_labels = [cur_ax.get_legend_handles_labels() for cur_ax in fig.axes[:1]]
#     lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#     fig.legend(lines, labels)
#     fig.suptitle(label, fontsize=34)
#     fig.tight_layout()
#     fig.show()
#
# format_plot(ff_fig, "Failure Frequency")
# format_plot(deficit_fig, "Maximum Deficit")
# format_plot(cost_fig, "Total Cost")


# LINE CHART OF ACTIONS FOR EACH SCENARIO

# num_acts = 10
# thresh_space = 0.5
# acts_space = 0.05
# ff_PT1 = np.zeros((num_acts, 15))
# ff_Ag = np.zeros((num_acts, 15))
# deficit_PT1 = np.zeros((num_acts, 15))
# deficit_Ag = np.zeros((num_acts, 15))
# cost = np.zeros((num_acts, 15))
# for i in range(num_acts):
#     # thresh = [-np.ones(12) * thresh_space * i]
#     thresh = [-np.ones(12)]
#     acts = [np.ones(12), np.ones(12) * (1 - (num_acts - 1 - i)*acts_space)]
#
#     m_ = make_model(demand_threshold_vals=thresh, demand_action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#     ff_Ag[i, :] = m_.recorders['failure_frequency_Ag'].values()
#     deficit_PT1[i, :] = m_.recorders['Maximum Deficit PT1'].values()
#     deficit_Ag[i, :] = m_.recorders['Maximum Deficit Ag'].values()
#     cost[i, :] = m_.recorders['TotalCost'].values()
#
# ff_fig, ff_ax = plt.subplots(3, 5, figsize=(15, 12))
# deficit_fig, deficit_ax = plt.subplots(3, 5, figsize=(15, 12))
# cost_fig, cost_ax = plt.subplots(3, 5, figsize=(15, 12))
# x_vals = np.linspace(1 - (num_acts - 1) * acts_space,1, num_acts, endpoint=True)
#
# max_ff = np.maximum(np.max(np.max(ff_PT1)), np.max(np.max(ff_Ag)))
# max_deficit = np.maximum(np.max(np.max(deficit_PT1)), np.max(np.max(deficit_Ag)))
# max_cost = np.max(np.max(cost))
# metric_properties = {
#     "ff_PT1": {
#         "ax": ff_ax,
#         "data": ff_PT1,
#         "max_val": max_ff,
#         "label": "PT1 Failure Frequency",
#         "color": "Purple",
#     },
#
#     "ff_Ag": {
#         "ax": ff_ax,
#         "data": ff_Ag,
#         "max_val": max_ff,
#         "label": "Agriculture Failure Frequency",
#         "color": "Green"
#     },
#
#     "deficit_PT1": {
#         "ax": deficit_ax,
#         "data": deficit_PT1,
#         "max_val": max_deficit,
#         "label": "PT1 Deficit",
#         "color": "Purple"
#     },
#
#     "deficit_Ag": {
#         "ax": deficit_ax,
#         "data": deficit_Ag,
#         "max_val": max_deficit,
#         "label": "Agriculture Deficit",
#         "color": "Green"
#     },
#
#     "cost": {
#         "ax": cost_ax,
#         "data": cost,
#         "max_val": max_cost,
#         "label": "Cost",
#         "color": "Orange"
#     }
# }
#
# def create_line_plot(ax, data, max_val, label, color):
#     ax.plot(x_vals, data, marker='o', label=label, color=color)
#     ax.set_ylim(0, (1.2*max_val).round(decimals=2))
#     ax.set_xticks(x_vals)
#     ax.set_xticklabels(x_vals.round(decimals=2), rotation=45)
#
#
# num_ticks = num_acts
# for i in range(15):
#     for metric, properties in metric_properties.items():
#         create_line_plot(
#             properties["ax"][i // 5, i % 5],
#             properties["data"][:, i],
#             properties["max_val"],
#             properties["label"],
#             properties["color"]
#         )
#
# def format_plot(fig, label):
#     lines_labels = [cur_ax.get_legend_handles_labels() for cur_ax in fig.axes[:1]]
#     lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#     fig.legend(lines, labels)
#     fig.suptitle(label, fontsize=34)
#     fig.tight_layout()
#     fig.show()
#
# format_plot(ff_fig, "Failure Frequency")
# format_plot(deficit_fig, "Maximum Deficit")
# format_plot(cost_fig, "Total Cost")



# num_acts = 10
#
# ff_PT1 = np.zeros((num_acts, 15))
# ff_Ag = np.zeros((num_acts, 15))
# for i in range(num_acts):
#     thresh = np.ones(num_DP) * -0.84
#     acts = np.ones(num_DP) * 30 * i
#
#     m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#     ff_Ag[i, :] = m_.recorders['failure_frequency_Ag'].values()
#
# fig, ax = plt.subplots(3, 5, figsize=(15, 12))
# x_vals = 30*np.arange(num_acts)
# for i in range(15):
#     cur_plot = ax[i // 5, i % 5]
#     cur_plot.plot(x_vals, ff_PT1[:, i], marker='o', label="PT1")
#     cur_plot.plot(x_vals, ff_Ag[:, i], marker='o', label="Ag")
#     cur_plot.set_ylim(0, 0.3)
#
#     num_ticks = num_acts
#     cur_plot.set_xticks(30 * np.arange(num_acts))
#     cur_plot.set_xticklabels(30 * np.arange(num_acts), rotation=45)
#
# lines_labels = [cur_ax.get_legend_handles_labels() for cur_ax in fig.axes[:1]]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)
#
# fig.tight_layout()
# fig.show()


# HEAT MAPS OF METRICS IN EACH SCENARIO

# num_thresh = 5
# num_acts = 5
# ff_PT1 = np.zeros((num_thresh, num_acts, 15))
# ff_Ag = np.zeros((num_thresh, num_acts, 15))
# deficit_PT1 = np.zeros((num_thresh, num_acts, 15))
# deficit_Ag = np.zeros((num_thresh, num_acts, 15))
# # Note: Cost might not be affected because contracts and things are already bought?
# cost = np.zeros((num_thresh, num_acts, 15))
# for i in range(num_thresh):
#     for j in range(num_acts):
#         thresh = -0.84*np.ones(num_DP)*i #[-0.5*np.ones(12)*i]
#         acts = 30*np.ones(num_DP)*(j + 1)#[np.ones(12), (1 - 0.1*j)*np.ones(12)]
#
#         m_ = make_model(contract_threshold_vals=thresh, contract_action_vals=acts)  # function in example_sim_opt.py
#         m_.check()
#         m_.check_graph()
#         m_.find_orphaned_parameters()
#
#         m_.run()
#
#         ff_PT1[i, j, :] = m_.recorders['failure_frequency_PT1'].values()
#         ff_Ag[i, j, :] = m_.recorders['failure_frequency_Ag'].values()
#         deficit_PT1[i, j, :] = m_.recorders['Maximum Deficit PT1'].values()
#         deficit_Ag[i, j, :] = m_.recorders['Maximum Deficit Ag'].values()
#         cost[i, j, :] = m_.recorders['TotalCost'].values()
#
# fig_ff_PT1, ax_ff_PT1 = plt.subplots(3, 5, figsize=(15, 12))
# fig_ff_Ag, ax_ff_Ag = plt.subplots(3, 5, figsize=(15, 12))
# fig_deficit_PT1, ax_deficit_PT1 = plt.subplots(3, 5, figsize=(15, 12))
# fig_deficit_Ag, ax_deficit_Ag = plt.subplots(3, 5, figsize=(15, 12))
# fig_cost, ax_cost = plt.subplots(3, 5, figsize=(15, 12))
# max_ff_PT1 = np.max(np.max(np.max(ff_PT1)))
# max_ff_Ag = np.max(np.max(np.max(ff_Ag)))
# max_deficit_PT1 = np.max(np.max(np.max(deficit_PT1)))
# max_deficit_Ag = np.max(np.max(np.max(deficit_Ag)))
# max_cost = np.max(np.max(np.max(cost)))
#
# x = np.arange(num_thresh)
# y = np.arange(num_acts)
# X, Y = np.meshgrid(x, y)
#
# def create_heat_map(fig, ax, data, max_val, color):
#     img = ax.imshow(data, cmap=color, vmin=0, vmax=max_val,
#                                         extent=[x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5],
#                                         interpolation='nearest', origin='lower')
#     ax.set_xticks(np.arange(num_acts))
#     ax.set_xticklabels(30 * np.arange(1, num_acts + 1))#1 - 0.1 * np.arange(num_acts), rotation=45)
#     ax.set_yticks(np.arange(num_thresh))
#     ax.set_yticklabels(-0.84 * np.arange(num_thresh))#-0.5 * np.arange(num_thresh))
#
#     fig.colorbar(img, ax=ax)
#
# metric_properties = {
#     "ff_PT1": {
#         "fig": fig_ff_PT1,
#         "ax": ax_ff_PT1,
#         "data": ff_PT1,
#         "color": "Purples",
#         "max_val": max_ff_PT1
#     },
#
#     "ff_Ag": {
#         "fig": fig_ff_Ag,
#         "ax": ax_ff_Ag,
#         "data": ff_Ag,
#         "color": "Greens",
#         "max_val": max_ff_Ag
#     },
#
#     "deficit_PT1": {
#         "fig": fig_deficit_PT1,
#         "ax": ax_deficit_PT1,
#         "data": deficit_PT1,
#         "color": "Purples",
#         "max_val": max_deficit_PT1
#     },
#
#     "deficit_Ag": {
#         "fig": fig_deficit_Ag,
#         "ax": ax_deficit_Ag,
#         "data": deficit_Ag,
#         "color": "Greens",
#         "max_val": max_deficit_Ag
#     },
#
#     "cost": {
#         "fig": fig_cost,
#         "ax": ax_cost,
#         "data": cost,
#         "color": "Oranges",
#         "max_val": max_cost
#     }
# }
#
# for i in range(15):
#     for metric, properties in metric_properties.items():
#         create_heat_map(
#             properties["fig"],
#             properties["ax"][i // 5, i % 5],
#             properties["data"][:, :, i],
#             properties["max_val"],
#             properties["color"]
#         )
#
#
# fig_ff_PT1.suptitle("PT1 Failure Frequency", fontsize=34)
# fig_ff_PT1.tight_layout()
# fig_ff_PT1.show()
# fig_ff_Ag.suptitle("Ag Failure Frequency", fontsize=34)
# fig_ff_Ag.tight_layout()
# fig_ff_Ag.show()
# fig_deficit_PT1.suptitle("PT1 Max Deficit", fontsize=34)
# fig_deficit_PT1.tight_layout()
# fig_deficit_PT1.show()
# fig_deficit_Ag.suptitle("Ag Max Deficit", fontsize=34)
# fig_deficit_Ag.tight_layout()
# fig_deficit_Ag.show()
# fig_cost.suptitle("Total Cost", fontsize=34)
# fig_cost.tight_layout()
# fig_cost.show()


# LINE CHART OF FAILURE FREQUENCY FOR DIFFERENT THRESHOLDS WITH LINE FOR EACH SCENARIO

# num_thresh = 5
# ff_PT1 = np.zeros((5, 15))
# for i in range(num_thresh):
#     thresh = np.ones(num_DP) * -0.84 * i
#     acts = np.ones(num_DP) * 30
#
#     m_ = make_model(contract_threshold_vals=thresh, contract_action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#
# fig, ax = plt.subplots(figsize=(12, 15))
# for i in range(15):
#     ax.plot(ff_PT1[:, i], marker='o', label="Scenario {}".format(i))
#     ax.set_xticks(np.arange(num_thresh))
#     ax.set_xticklabels(-0.84 * np.arange(num_thresh), rotation=45)
#
# ax.legend()
# fig.suptitle("PT1 Failure Frequency", fontsize=34)
# fig.show()

#%% Test contract thresholds and actions

num_k= 1  # number of levels in policy tree
num_DP = 7  # number of decision periods

# BARS OF SCENARIOS FOR EACH THRESHOLD

# fig, ax = plt.subplots(5)
#
# for i in range(5):
#     thresh = np.ones(num_DP) * -0.84 * i
#     acts = np.ones(num_DP) * 30
#
#     m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1 = m_.recorders['failure_frequency_PT1'].values()
#     ax[i].bar(list(map(lambda x: str(x), range(1, 16))), ff_PT1)
#     ax[i].set_ylim(0, 0.3)
#     ax[i].axhline(y=np.mean(ff_PT1), color='red', linestyle='--', label="Avg failure frequency PT1")
#
# fig.tight_layout()
# fig.show()


# BARS OF THRESHOLDS FOR EACH SCENARIO

# ff_PT1 = np.zeros((5, 15))
# ff_Ag = np.zeros((5, 15))
# for i in range(5):
#     thresh = np.ones(num_DP) * -0.84 * i
#     acts = np.ones(num_DP) * 30
#
#     m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#     ff_Ag[i, :] = m_.recorders['failure_frequency_Ag'].values()
#
# fig, ax = plt.subplots(3, 5, figsize=(15, 12))
# X_axis = np.arange(5)
# for i in range(15):
#     cur_plot = ax[i // 5, i % 5]
#     cur_plot.bar(-width/2 + X_axis, ff_PT1[:, i], width=width, label="PT1")
#     cur_plot.bar(width/2 + X_axis, ff_Ag[:, i], width=width, label="Ag")
#     cur_plot.set_ylim(0, 0.3)
#
#     cur_plot.set_xticks(X_axis, list(map(lambda x: str(x), -0.84 * np.arange(5))))
#     cur_plot.set_xticklabels(list(map(lambda x: str(x), -0.84 * np.arange(5))), rotation=45)
#     cur_plot.legend()


# LINE CHART OF THRESHOLDS FOR EACH SCENARIO

# num_thresh = 10
# ff_PT1 = np.zeros((num_thresh, 15))
# ff_Ag = np.zeros((num_thresh, 15))
# for i in range(num_thresh):
#     thresh = np.ones(num_DP) * -0.84 * i
#     acts = np.ones(num_DP) * 30
#
#     m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#     ff_Ag[i, :] = m_.recorders['failure_frequency_Ag'].values()
#
# fig, ax = plt.subplots(3, 5, figsize=(15, 12))
# # X_axis = np.arange(5)
# x_vals = -0.84*np.arange(num_thresh)
# for i in range(15):
#     cur_plot = ax[i // 5, i % 5]
#     cur_plot.plot(x_vals, ff_PT1[:, i], marker='o', label="PT1")
#     cur_plot.plot(x_vals, ff_Ag[:, i], marker='o', label="Ag")
#     cur_plot.set_ylim(0, 0.3)
#
#     num_ticks = num_thresh
#     cur_plot.set_xticks(-0.84 * np.arange(num_ticks))
#     cur_plot.set_xticklabels(-0.84 * np.arange(num_ticks), rotation=45)
#     # cur_plot.legend()
#
# lines_labels = [cur_ax.get_legend_handles_labels() for cur_ax in fig.axes[:1]]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)#, loc="upper right")
#
# fig.tight_layout()
# fig.show()


# LINE CHART OF ACTIONS FOR EACH SCENARIO

# num_acts = 10
#
# ff_PT1 = np.zeros((num_acts, 15))
# ff_Ag = np.zeros((num_acts, 15))
# for i in range(num_acts):
#     thresh = np.ones(num_DP) * -0.84
#     acts = np.ones(num_DP) * 30 * i
#
#     m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#     ff_Ag[i, :] = m_.recorders['failure_frequency_Ag'].values()
#
# fig, ax = plt.subplots(3, 5, figsize=(15, 12))
# x_vals = 30*np.arange(num_acts)
# for i in range(15):
#     cur_plot = ax[i // 5, i % 5]
#     cur_plot.plot(x_vals, ff_PT1[:, i], marker='o', label="PT1")
#     cur_plot.plot(x_vals, ff_Ag[:, i], marker='o', label="Ag")
#     cur_plot.set_ylim(0, 0.3)
#
#     num_ticks = num_acts
#     cur_plot.set_xticks(30 * np.arange(num_acts))
#     cur_plot.set_xticklabels(30 * np.arange(num_acts), rotation=45)
#
# lines_labels = [cur_ax.get_legend_handles_labels() for cur_ax in fig.axes[:1]]
# lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
# fig.legend(lines, labels)
#
# fig.tight_layout()
# fig.show()


# HEAT MAPS FAILURE FREQUENCY IN EACH SCENARIO

# num_thresh = 5
# num_acts = 5
# ff_PT1 = np.zeros((num_thresh, num_acts, 15))
# ff_Ag = np.zeros((num_thresh, num_acts, 15))
# for i in range(num_thresh):
#     for j in range(num_acts):
#         thresh = np.ones(num_DP) * -0.84 * (num_thresh - 1 - i)
#         acts = np.ones(num_DP) * 30 * (j + 1)
#
#         m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
#         m_.check()
#         m_.check_graph()
#         m_.find_orphaned_parameters()
#
#         m_.run()
#
#         ff_PT1[i, j, :] = m_.recorders['failure_frequency_PT1'].values()
#         ff_Ag[i, j, :] = m_.recorders['failure_frequency_Ag'].values()
#
#
# fig_PT1, ax_PT1 = plt.subplots(3, 5, figsize=(15, 12))
# fig_Ag, ax_Ag = plt.subplots(3, 5, figsize=(15, 12))
# x = np.arange(num_thresh)
# y = np.arange(num_acts)
# X, Y = np.meshgrid(x, y)
# for i in range(15):
#     cur_plot_PT1 = ax_PT1[i // num_thresh, i % num_thresh]
#     cur_data_PT1 = ff_PT1[:, :, i]
#
#     img_PT1 = cur_plot_PT1.imshow(cur_data_PT1, cmap='Greens', vmin=0, vmax=0.3,
#                    extent=[x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5],
#                    interpolation='nearest', origin='lower')
#     cur_plot_PT1.set_xticks(np.arange(num_acts))
#     cur_plot_PT1.set_xticklabels(30 * np.arange(1, num_acts + 1))
#     cur_plot_PT1.set_yticks(np.arange(num_thresh))
#     cur_plot_PT1.set_yticklabels(-0.84 * np.arange(num_thresh - 1, -1, -1), rotation=45)
#
#     fig_PT1.colorbar(img_PT1, ax=cur_plot_PT1)
#
#
#     cur_plot_Ag = ax_Ag[i // num_thresh, i % num_thresh]
#     cur_data_Ag = ff_Ag[:, :, i]
#
#     img_Ag = cur_plot_Ag.imshow(cur_data_Ag, cmap='Greens', vmin=0, vmax=0.3,
#                    extent=[x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5],
#                    interpolation='nearest', origin='lower')
#     cur_plot_Ag.set_xticks(np.arange(num_acts))
#     cur_plot_Ag.set_xticklabels(30 * np.arange(num_acts))
#     cur_plot_Ag.set_yticks(np.arange(num_thresh))
#     cur_plot_Ag.set_yticklabels(-0.84 * np.arange(num_thresh - 1, -1, -1), rotation=45)
#
#     fig_Ag.colorbar(img_Ag, ax=cur_plot_Ag)
#
# fig_PT1.tight_layout()
# fig_PT1.show()
# fig_Ag.tight_layout()
# fig_Ag.show()

# LINE CHART OF FAILURE FREQUENCY FOR DIFFERENT THRESHOLDS WITH LINE FOR EACH SCENARIO

# ff_PT1 = np.zeros((5, 15))
# for i in range(5):
#     thresh = np.ones(num_DP) * -0.84 * i
#     acts = np.ones(num_DP) * 30
#
#     m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1[i, :] = m_.recorders['failure_frequency_PT1'].values()
#
# fig, ax = plt.subplots(figsize=(12, 15))
# for i in range(15):
#     ax.plot(ff_PT1[:, i], marker='o', label="Scenario {}".format(i))
#
# ax.legend()
# fig.show()


#%% run/simulate the same Pywr model and make Plotly plots

# BUBBLE PLOT OF FAILURE FREQUENCY IN EACH SCENARIO

# ff_PT1 = np.zeros((2, 2, 15))
#
# thresh_list = -0.84 * np.atleast_2d(np.arange(2)).T.repeat(num_DP, axis=1)
# acts_list = 30 * np.atleast_2d(np.arange(1, 3)).T.repeat(num_DP, axis=1)
#
# for i in range(2):
#     for j in range(2):
#         thresh = thresh_list[i, :]
#         acts = acts_list[j, :]
#
#         m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
#         m_.check()
#         m_.check_graph()
#         m_.find_orphaned_parameters()
#
#         m_.run()
#
#         ff_PT1[i, j, :] = m_.recorders['failure_frequency_PT1'].values()
#
# bubble_size = np.sqrt(ff_PT1)
#
# fig = make_subplots(rows=3, cols=5)
# for i in range(15):
#     fig.add_trace(
#         go.Scatter(
#             x=np.repeat(thresh_list[:, 0], 2),
#             y=np.tile(acts_list[:, 0], 2),
#             name="Scenario " + str(i + 1),
#             mode="markers",
#             marker=dict(
#                 size=bubble_size[:, :, i].flatten(),
#                 sizemode='area',
#                 sizeref=2.*np.max(bubble_size)/(40**2),
#                 # sizemin=4
#             )
#         ),
#         row=1 + i // 5,
#         col=1 + i % 5
#     )
#
# fig.show()


# PARALLEL COORDINATES PLOT

# results = pd.DataFrame()
# results['scenario'] = pd.Series(np.arange(1, 16))
# for i in range(5):
#     thresh = np.ones(num_DP) * -0.84 * i
#     acts = np.ones(num_DP) * 30
#
#     m_ = make_model(threshold_vals=thresh, action_vals=acts)  # function in example_sim_opt.py
#     m_.check()
#     m_.check_graph()
#     m_.find_orphaned_parameters()
#
#     m_.run()
#
#     ff_PT1 = m_.recorders['failure_frequency_PT1'].values()
#     ff_Ag = m_.recorders['failure_frequency_Ag'].values()
#
#     results[str(-0.84*i)] = [x for x in ff_PT1]
#
# fig = px.parallel_coordinates(
#     results,
#     color="scenario"
# )


# fig.show()


#%% Structures for plots

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import plotly.express as px
#
# # Heatmap
# fig1, ax1 = plt.subplots()
# x, y = np.meshgrid(np.arange(10), np.arange(10))
# z = x**2 - y**2
#
# c = ax1.imshow(z, cmap='Greens', vmin=z.min(), vmax=z.max(),
#                extent=[x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5],
#                interpolation='nearest', origin='lower')
# ax1.set_xticks(np.arange(10))
# ax1.set_yticks(np.arange(10))
# fig1.colorbar(c, ax=ax1)
# ax1.set_title('matplotlib.pyplot.imshow() function Example',
#           fontweight="bold")
# plt.show()
#
#
# # Parallel coordinates (opens in firefox)
# df = px.data.iris()
# fig = px.parallel_coordinates(df, color="species_id", labels={"species_id": "Species",
#                 "sepal_width": "Sepal Width", "sepal_length": "Sepal Length",
#                 "petal_width": "Petal Width", "petal_length": "Petal Length", },
#                              color_continuous_scale=px.colors.diverging.Tealrose,
#                              color_continuous_midpoint=2)
# fig.show()