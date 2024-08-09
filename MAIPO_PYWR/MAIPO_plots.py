import pandas
from matplotlib import pyplot as plt
import numpy as np
df = pandas.read_csv("MAIPO results.csv")


objectives = df[[c for c in df.columns if c.startswith('MO_')]]
from pandas.plotting import scatter_matrix
scatter_matrix = scatter_matrix(objectives, diagonal = "hist", figsize = (14,14))
for ax in scatter_matrix.ravel():
    ax.set_xlabel(ax.get_xlabel(), fontsize = 6, rotation = 0)
    ax.set_ylabel(ax.get_ylabel(), fontsize = 6, rotation = 0)
plt.savefig('MAIPO Objectives.pdf', format='pdf')
plt.show()

