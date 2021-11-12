import include
import sys
import numpy as np
from epi import parameters as pm

fileName = sys.argv[1]

param = pm.Params()
param.load(fileName)

print('  ***  nParams: ' +str(param.nParams))
print('  ***  nPhases: ' +str(param.nPhases))
print('  ***  times:   ' +str(param.times))
print('  ***  params:  ' +str(param.params))
print('  ***  mask:    ' +str(param.mask))
print('  ***  lower:   ' +str(param.lower_bounds))
print('  ***  upper:   ' +str(param.upper_bounds))
