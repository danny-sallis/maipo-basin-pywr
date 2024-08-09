import json
import pandas
from MAIPO_parameters import *
import platypus
from pywr.optimisation.platypus import PlatypusWrapper
from matplotlib import pyplot as plt

# pesos como producto punto. Para hacer sumas ponderadas, los numeros deben sumar 1
# weights as a dot product. To make weighted sums, the numbers must add up to 1
weights = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]


class CustomPlatypusWrapper(PlatypusWrapper):
    def customise_model(self, model):
        CustomizedAggregation(model, weights)
    

def get_model_data():

    with open('sc_MAIPO_.json') as fh:
        data = json.load(fh)
    return data


def platypus_main():

    wrapper = CustomPlatypusWrapper(get_model_data())

    evaluator_class = platypus.ProcessPoolEvaluator
    #evaluator_class = platypus.MapEvaluator
    #evaluator_class = platypus.MultiprocessingEvaluator

    with evaluator_class() as evaluator:
        algorithm = platypus.NSGAIII(wrapper.problem, population_size=50, evaluator=evaluator, divisions_outer=6)
        algorithm.run(10000)
    
    objective_names = []
    objective_directions = []
    for o in wrapper.model_objectives:
        direction = 'MIN' if o.is_objective == 'minimise' else 'MAX'        
        n = 'MO_{}_{}'.format(direction, o.name.replace('_', ' '))
        objective_names.append(n)
        objective_directions.append(1 if o.is_objective == 'minimise' else -1)
    
    objective_directions = np.array(objective_directions)

    objectives = pandas.DataFrame([s.objectives[:]*objective_directions for s in algorithm.result], columns=objective_names)

    variables = pandas.DataFrame([s.variables[:] for s in algorithm.result],
                    columns=['DEC_XXX_{}'.format(p.name.replace('_', ' ')) for p in wrapper.model_variables])
    
    df = pandas.concat([objectives, variables], axis=1)
    df.index.name = 'ID'
    df.to_csv('MAIPO Results.csv')    


    from pandas.plotting import scatter_matrix
    scatter_matrix(objectives)
    plt.savefig('MAIPO Objectives.pdf', format='pdf')
    plt.show()

if __name__ == '__main__':
    platypus_main()
