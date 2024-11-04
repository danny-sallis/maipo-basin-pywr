source("borg.R")
source("DTLZ2.R")

result <- borg(nvars, nobjs, 0, DTLZ2, 10000, epsilons=c(0.01, 0.01))
print(result)
