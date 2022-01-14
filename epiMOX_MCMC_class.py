import sys
import numpy as np
import pandas as pd
import datetime
import os.path
from epi import loaddata as ld
from epi import estimation as es
from epi.convert import converter
from epi.MCMC_class import solveMCMC
from epiMOX_class import epiMOX

if __name__ == '__main__':
    # Read data file
    if len(sys.argv) < 2:
        sys.exit("Error - at least 1 argument required: path to the folder containing the 'input.inp' file")
    testPath = sys.argv[1]
    fileName = testPath + '/input.inp'
    if os.path.exists(fileName):
        DataDic = ld.readdata(fileName)
    else:
        sys.exit('Error - Input data file ' + fileName + ' not found. Exit.')
    if len(sys.argv) > 2:
        nsimu = sys.argv[2]
    else:
        nsimu = 1e4

    if len(sys.argv) > 3:
        parallel = bool(sys.argv[3])
    else:
        parallel = False

    if len(sys.argv) > 4:
        nproc = nchains = int(sys.argv[4])
    else:
        nproc = nchains = 3

    model_solver = epiMOX(testPath)
    
    Y0 = model_solver.Y0.copy()

    n_comp_calibration = 6 if Y0.shape[0]==11 else 1 

    mcstat = solveMCMC(testPath, model_solver, Y0, nsimu=nsimu, sigma=np.array([0.1 * 3e2]*n_comp_calibration), parallel=parallel, nproc=nproc, nchains=nchains)
