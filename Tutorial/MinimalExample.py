from pywr.core import Model, Input, Output, Timestepper
from pywr.recorders import NumpyArrayNodeRecorder
import pandas as pd
import datetime

def python_model():
    # create a model (including an empty network)
    model = Model()

    # create two nodes: a supply, and a demand
    supply = Input(model, name='supply')
    demand = Output(model, name='demand')

    # create a connection from the supply to the demand
    supply.connect(demand)

    # set maximum flows
    supply.max_flow = 10.0
    demand.max_flow = 6.0

    # set cost (+ve) or benefit (-ve)
    supply.cost = 3.0
    demand.cost = -100.0

    model.timestepper = Timestepper(
        pd.to_datetime('2015-01-01'),  # first day
        pd.to_datetime('2015-12-31'),  # last day
        datetime.timedelta(1)  # interval
    )

    recorder = NumpyArrayNodeRecorder(model, supply)

    # lets get this party started!
    model.run()

    scenario = 0
    timestep = 0
    print(recorder.data[scenario][timestep])  # prints 6.0

# python_model()

# load the model
model = Model.load("MinimalExample.json")
# run, forest, run!
model.run()