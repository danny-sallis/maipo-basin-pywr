'''
Purpose: To initiate the optimization process with BORG, which will iteratively call the Pywr simulation model.
# documentation: https://waterprogramming.wordpress.com/2017/03/06/using-borg-in-parallel-and-serial-with-a-python-wrapper/
'''


import sys
sys.path.append("/home/danny/FletcherLab/maipo-basin-pywr")

# general package imports
import numpy as np
import pandas as pd
import platform  # helps identify directory locations on different types of OS
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


import example_sim_opt  # Import main optimization module that uses borg python wrapper

# Module within example
example_sim_opt.Optimization()

# ALMOST WORKING! NEED borg.dll, NOT libborg.so FOR WINDOWS
