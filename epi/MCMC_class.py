import os
import numpy as np
import pandas as pd
from pymcmcstat.MCMC import MCMC
from pymcmcstat.MCMC import DataStructure
from pymcmcstat.ParallelMCMC import ParallelMCMC

def solveMCMC(testPath, model_solver, Y0, nsimu = 1e4, sigma = 0.1*3e4, parallel=False, nproc=3, nchains=3):
    os.mkdir(testPath+'/MCMC')
    mcstat = MCMC()
    params = model_solver.params 
    mask = params.getMask()
    if params.nParams==4: #SEIRD
        names = ['beta','alpha','gamma','f']
    elif params.nParams==2: #SIR
        names = ['beta', 'gamma']
    elif params.nParams==13: #SUIHTER
        names = ['betaU','betaI','delta','omegaI','omegaH','rhoU','rhoI','rhoH','rhoT','gamma_T','gamma_I','theta_H','theta_T']

    t_list = model_solver.t_list[:int(max(model_solver.data.time.values))+1]
    mcstat.data.add_data_set(t_list,Y0)
    minimum = params.getLowerBounds()
    maximum = params.getUpperBounds()

    if params.nSites != 1:
        for i in range(params.nPhases):
            for j in range(params.nParams):
                for k in range(params.nSites):
                    if mask[i,j,k]:
                        mcstat.parameters.add_model_parameter(
                            name=str(k)+names[j]+str(i),
                            theta0=params.get()[i,j,k],
                            minimum=0.7*params.get()[i,j,k],
                            maximum=1.3*params.get()[i,j,k])
        for k in range(params.nSites):
            mcstat.parameters.add_model_parameter(
                name=str(k)+'I0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)
        for k in range(params.nSites):
            mcstat.parameters.add_model_parameter(
                name=str(k)+'R0',
                theta0=1,
                minimum=0.7,
                maximum=1.3)

    else:
        for i in range(params.nPhases):
            for j in range(params.nParams):
                if mask[i,j]:
                    mcstat.parameters.add_model_parameter(
                        name=names[j]+str(i),
                        theta0=params.get()[i,j],
                        minimum=0.7*params.get()[i,j],
                        maximum=1.3*params.get()[i,j])

        if len(names) == 13:
            mcstat.parameters.add_model_parameter(
                name='omegaI_err',
                theta0=0,
                prior_mu=0,
                prior_sigma=0.10)

            mcstat.parameters.add_model_parameter(
                name='omegaH_err',
                theta0=0,
                prior_mu=0,
                prior_sigma=0.20)

        mcstat.parameters.add_model_parameter(
            name='U0',
            theta0=1,
            minimum=0.7,
            maximum=1.3)

        mcstat.parameters.add_model_parameter(
            name='R0',
            theta0=1,
            minimum=0.7,
            maximum=1.3)

    mcstat.simulation_options.define_simulation_options(
        nsimu=nsimu,
        updatesigma=1,
        save_to_json=True,
        save_to_txt=True,
        results_filename='results_dict.json',
        savedir=testPath+'/MCMC/')
    mcstat.model_settings.define_model_settings(
        sos_function=model_solver.error_MCMC,
        sigma2=sigma**2)
    if parallel:
        parmcstat = ParallelMCMC()
        parmcstat.setup_parallel_simulation(mcstat,num_cores=nproc,num_chain=nchains)
        parmcstat.run_parallel_simulation()
        return parmcstat

    mcstat.run_simulation()
    return mcstat
