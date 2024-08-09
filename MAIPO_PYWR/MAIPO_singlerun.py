import time
start = time.time()

import pyximport
pyximport.install()
from pywr.model import Model
from pywr.notebook import draw_graph
from pywr.recorders import TablesRecorder, Recorder
from pywr.parameters import IndexParameter, Parameter, load_parameter
import tables
import pandas
import numpy as np
import os
from MAIPO_parameters import *
from MAIPO_searcher import *

import warnings
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)

from matplotlib import pyplot as plt


m = Model.load("sc_MAIPO_.json")

TablesRecorder(m, "outputs/flows.h5", filter_kwds={"complib": "zlib", "complevel": 5 }, parameters= ["contract_value", "DP_index", "purchases_value", "Maipo_max_volume"])

weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
CustomizedAggregation(m, weights)


m.run()


data = {}
values = {}
for r in m.recorders:

    rdata = {
        'class': r.__class__.__name__,
    }

    try:
        rdata['value'] = r.aggregated_value()
    except NotImplementedError:
        pass

    try:
        rdata['node'] = r.node.name
    except AttributeError:
        pass

    try:
        values[r.name] = list(r.values())
    except NotImplementedError:
        pass

    if len(rdata) > 1:
        data[r.name] = rdata

writer = pandas.ExcelWriter('outputs/metrics.xlsx', engine = 'xlsxwriter')
metrics = pandas.DataFrame(data).T
metrics.to_csv("outputs/metrics.csv")
metrics.to_excel(writer, 'aggregated')
scenario_values = pandas.DataFrame(values).T
scenario_values.to_excel(writer, 'scenarios')
# writer.save() - note: save has been depreciated. Use close() instead.
writer.close()

print("--- %s seconds ---" % (time.time() - start))
