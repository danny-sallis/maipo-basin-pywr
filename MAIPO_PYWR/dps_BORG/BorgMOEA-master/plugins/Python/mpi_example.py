from borg import *

Configuration.startMPI()

borg = Borg(2, 2, 0, lambda x,y : [x**2 + y**2, (x-2)**2 + y**2])
borg.setBounds([-50, 50], [-50, 50])
borg.setEpsilons(0.01, 0.01)

result = borg.solveMPI(maxEvaluations=100000)

# the result will only be returned from one node
if result:
	result.display()

Configuration.stopMPI()
