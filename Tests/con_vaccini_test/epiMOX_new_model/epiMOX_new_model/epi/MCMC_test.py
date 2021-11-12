import os
import numpy as np
import matplotlib.pyplot as plt
import functools
import shutil
from pymcmcstat.MCMC import MCMC
from pymcmcstat.MCMC import DataStructure
from pymcmcstat.ParallelMCMC import ParallelMCMC
from epi import estimation as es
from epi import models_test as md

def model(*args,epi_model='SUIHTER'):
    modello = getattr(md, epi_model + 'model')
    res = modello(*args)
    return res.reshape((-1,args[2].nSites)).squeeze()

def model_fun(q,data):
    t_data = data.xdata[0][:, 0]
    args = data.user_defined_object[0]
    epi_model = args[0]
    Ns = len(args[3])
    scenarios = args[-1]
    args[2].params[args[2].getMask()] = q[:-2]
    args[2].forecast(args[2].dataEnd,t_data[-1],0,scenarios)
    Y0 = args[1].copy()
    Y0[1] *= q[-2]
    Y0[6] *= q[-1]
    Y0[0] = args[3] - Y0[1:].sum()
    res = es.solve_rk4(getattr(md, epi_model + 'model'), [t_data[0], t_data[-1]], Y0, t_data[1] - t_data[0], args=args[2:-1])
    new_pos = np.concatenate([[data.ydata[0][-1,0]],(res[Ns:2*Ns,:-1] * np.array([args[2].delta(t) for t in t_data[1:]]).transpose()).flatten()])
    new_ICU = np.concatenate([[np.nan],(res[3*Ns:4*Ns,:-1] * np.array([args[2].omegaH(t) for t in t_data[1:]]).transpose()).flatten()])
    rec_det = new_pos#es.postProcessH(args[2],t_data,res[2*Ns:3*Ns,:], res[3*Ns:4*Ns,:], res[4*Ns:5*Ns,:],args[5]).flatten('F') + np.repeat(data.ydata[0][-2,0::int(args[2].dataEnd+1)],len(t_data)) 
    res = np.concatenate((res.reshape(-1,Ns*len(t_data)),new_pos.reshape(1,-1),rec_det.reshape(1,-1),new_ICU.reshape(1,-1)),axis=0)
    return res.transpose()

def model_fun_mod(q,data):
    t_data = data.xdata[0][:, 0]
    args = data.user_defined_object[0]
    epi_model = args[0]
    Ns = len(args[3])
    scenarios = args[-1]
    args[2].params[args[2].getMask()] = q[:-2]
    args[2].forecast(args[2].dataEnd,t_data[-1],0,scenarios)
    Y0 = args[1].copy()
    Y0[1] *= q[-2]
    Y0[6] *= q[-1]
    Y0[0] = args[3] - Y0[1:].sum()
    res = es.solve_rk4(getattr(md, epi_model + 'model'), [t_data[0], t_data[-1]], Y0, t_data[1] - t_data[0], args=args[2:-1])
    last = len(data.ydata[0][0])-1
    Y0 = np.zeros(res.shape[0])
    Y0[1] = res[1,last]
    #Y0[1] += 200/args[2].delta(last)
    Y0[6:] = res[6:,last]
    Y0[2:6] = data.ydata[0][:-2,-1]
    Y0[0] = args[3] - Y0[1:].sum()
    t_data = t_data[last:]
    res = es.solve_rk4(getattr(md, epi_model + 'model'), [t_data[0], t_data[-1]], Y0, t_data[1] - t_data[0], args=args[2:-1])
    new_pos = np.concatenate([[data.ydata[0][-1,-1]],(res[Ns:2*Ns,:-1] * np.array([args[2].delta(t) for t in t_data[1:]]).transpose()).flatten()])
    rec_det = np.concatenate([[0],es.postProcessH(args[2],t_data,res[2*Ns:3*Ns,:], res[3*Ns:4*Ns,:], res[4*Ns:5*Ns,:],args[5]).flatten('F')[:-1]]) + np.repeat(data.ydata[0][-2,last::int(args[2].dataEnd+1)],len(t_data)) 
    res = np.concatenate((res.reshape(-1,Ns*len(t_data)),new_pos.reshape(1,-1),rec_det.reshape(1,-1)),axis=0)
    return res.transpose()

def model_fun_var(q,data):
    variant_perc = 0.#40
    variant_factor = 1.5
    kappa1 = 33.5 / 51.1
    kappa2 = 80.9 / 86.8 
    t_data = data.xdata[0][:, 0]
    args = data.user_defined_object[0]
    epi_model = args[0]
    Ns = len(args[3])
    scenarios = args[-1]
    args[2].params[args[2].getMask()] = q[:-2]
    args[2].forecast(args[2].dataEnd,t_data[-1],0,scenarios)
    Y0 = args[1].copy()
    Y0[1] *= q[-2]
    Y0[6] *= q[-1]
    Y0[0] = args[3] - Y0[1:].sum()
    res = es.solve_rk4(getattr(md, epi_model + 'model'), [t_data[0], t_data[-1]], Y0, t_data[1] - t_data[0], args=args[2:-1])
    last = len(data.ydata[0][0])-1
    Y0 = np.zeros(res.shape[0]+2)
    #res[1,last] += 200/args[2].delta(last)
    Y0[1] = res[1,last] * (1 - variant_perc)
    Y0[2] = res[1,last] * variant_perc
    Y0[3] = data.ydata[0][0,-1] * (1 - variant_perc/2)
    Y0[4] = data.ydata[0][0,-1] * variant_perc/2
    Y0[8:] = res[6:,last]
    Y0[5:8] = data.ydata[0][1:-2,-1]
    Y0[0] = args[3] - Y0[1:].sum()
    t_data = t_data[last:]
    res = es.solve_rk4(getattr(md, epi_model + 'model_variant'), [t_data[0], t_data[-1]], Y0, t_data[1] - t_data[0], args=args[2:-1]+[variant_perc, variant_factor, kappa1, kappa2])
    res = np.vstack([res[0:Ns],np.sum(res[Ns:3*Ns],axis=0),np.sum(res[3*Ns:5*Ns],axis=0),res[5*Ns:]])

    new_pos = np.concatenate([[data.ydata[0][-1,-1]],(res[Ns:2*Ns,:-1] * np.array([args[2].delta(t) for t in t_data[1:]]).transpose()).flatten()])
    rec_det = np.concatenate([[0],es.postProcessH(args[2],t_data,res[2*Ns:3*Ns,:], res[3*Ns:4*Ns,:], res[4*Ns:5*Ns,:],args[5]).flatten('F')[:-1]]) + np.repeat(data.ydata[0][-2,last::int(args[2].dataEnd+1)],len(t_data)) 
    res = np.concatenate((res.reshape(-1,Ns*len(t_data)),new_pos.reshape(1,-1),rec_det.reshape(1,-1)),axis=0)
    return res.transpose()


def model_fun_var_new(q,data):
    variant_perc = 0.327 
    variant_factor = 1.5
    kappa1 = 33.5 / 51.1
    kappa2 = 80.9 / 86.8 
    t_data = data.xdata[0][:, 0]
    args = data.user_defined_object[0]
    epi_model = args[0]
    Ns = len(args[3])
    scenarios = args[-1]
    args[2].params[args[2].getMask()] = q[:-4]
    args[2].forecast(args[2].dataEnd,t_data[-1],0,scenarios)
    args[2].params_time[int(t_data[0]):int(t_data[-1])+1,3] = args[2].omegaI(t_data)+q[-4]
    args[2].params_time[int(t_data[0]):int(t_data[-1])+1,4] = args[2].omegaH(t_data)+q[-3]
    Y0 = args[1].copy()
    Y0[1] *= q[-2]
    Y0[6] *= q[-1]
    Y0[0] = args[3] - Y0[1:].sum()
    res = es.solve_rk4(getattr(md, epi_model + 'model'), [t_data[0], t_data[-1]], Y0, t_data[1] - t_data[0], args=args[2:-1])
    last = len(data.ydata[0][0])-1
    Y0 = np.zeros(res.shape[0]+1)
    #res[1,last] -= 100/args[2].delta(last)
    Y0[1] = res[1,last] * (1 - variant_perc)
    Y0[2] = res[1,last] * variant_perc
    Y0[7:] = res[6:,last]
    Y0[3:7] = data.ydata[0][0:-2,-1]
    Y0[0] = args[3] - Y0[1:].sum()
    t_data = t_data[last:]
    res = es.solve_rk4(getattr(md, epi_model + 'model_variant'), [t_data[0], t_data[-1]], Y0, t_data[1] - t_data[0], args=args[2:-1]+[variant_perc, variant_factor, kappa1, kappa2])
    res = np.vstack([res[0:Ns],np.sum(res[Ns:3*Ns],axis=0),res[3*Ns:]])

    new_pos = np.concatenate([[data.ydata[0][-1,-1]],(res[Ns:2*Ns,:-1] * np.array([args[2].delta(t) for t in t_data[1:]]).transpose()).flatten()])
    rec_det = np.concatenate([[0],es.postProcessH(args[2],t_data,res[2*Ns:3*Ns,:], res[3*Ns:4*Ns,:], res[4*Ns:5*Ns,:],args[5]).flatten('F')[:-1]]) + np.repeat(data.ydata[0][-2,last::int(args[2].dataEnd+1)],len(t_data)) 
    res = np.concatenate((res.reshape(-1,Ns*len(t_data)),new_pos.reshape(1,-1),rec_det.reshape(1,-1)),axis=0)
    return res.transpose()

def model_ss(params,data):
    t_data = data.xdata[0][:, 0]
    ydata = data.ydata[0]
    args = data.user_defined_object[0]
    epi_model = args[0]

    Y0 = args[1].copy()
    Y0[1] *= params[-2]
    Y0[6] *= params[-1]
    Y0[0] = args[3] - Y0[1:].sum()
    args[2].params[args[2].getMask()] = params[:-4]
    args[2].params_time[int(t_data[0]):int(t_data[-1])+1,3] = args[2].omegaI(t_data)+params[-4]
    args[2].params_time[int(t_data[0]):int(t_data[-1])+1,4] = args[2].omegaH(t_data)+params[-3]

    res = es.solve_rk4(getattr(md, epi_model + 'model'), [t_data[0], t_data[-1]], Y0, t_data[1]-t_data[0], args=args[2:]).reshape(-1,len(t_data)*len(args[3]))
    new_pos = np.concatenate([[data.ydata[0][-1,0]],(res[1,:-1] * np.array([args[2].delta(t) for t in t_data[1:]]).transpose()).flatten()])

    RtotD = es.postProcessH(args[2], t_data, res[2:3], res[3:4], res[4:5], args[5]).flatten('F') + ydata[-1,0]

    ss=np.zeros(6)

    #weights =np.array([1, 1, 1, 1, 0.01]).reshape(5,1)
    if res.shape[0]==5:
        ydata = ydata[:,0]
        ss = ((res[-1, :] - ydata) ** 2).sum()
    elif res.shape[0]==7:
        #ydata = ydata[:-1]
        ss = (((res[2:, :] - ydata) ** 2)/np.maximum(ydata,np.ones(ydata.shape))).sum(axis=1)
    elif res.shape[0]==9:
        #ydata = ydata[:-1]
        #ss = (((res[2:-2, :] - ydata]) ** 2)/np.maximum(weights*ydata,np.ones(ydata.shape))).sum(axis=1)
        ss[:3] = (((res[2:5, :] - ydata[:-3]) ** 2)/np.maximum(ydata[:-3],np.ones(ydata[:-3].shape))).sum(axis=1)
        ss[3] = (((np.diff(res[5]) - ydata[-3,1:])**2)/np.maximum(ydata[-3,1:],np.ones(ydata[-2,1:].shape))).sum()
        ss[4] = (((RtotD - ydata[-2])**2)/np.maximum(0.1*ydata[-2],np.ones(ydata[-2].shape))).sum()
        ss[5] = (((new_pos - ydata[-1])**2) / np.maximum(ydata[-1],np.ones(ydata[-1].shape))).sum()
    return ss



def solveMCMC(testPath,t_vals,obs,Y0,l_args,nsimu = 1e4,sigma = 0.1*3e4,epi_model='SEIRD',parallel=False,nproc=3,nchains=3):
    #if os.path.exists(testPath+'/MCMC'):
        #shutil.rmtree(testPath+'/MCMC')
    os.mkdir(testPath+'/MCMC')
    mcstat = MCMC()
    params=l_args[0]
    mask = params.getMask()
    if params.nParams==4: #SEIRD
        names = ['beta','alpha','gamma','f']
    elif params.nParams==13: #SUIHTER
        names = ['betaU','betaI','delta','omegaI','omegaH','rhoU','rhoI','rhoH','rhoT','gamma_T','gamma_I','theta_H','theta_T']

    args = [epi_model,Y0,*l_args]
    mcstat.data.add_data_set(t_vals,obs,user_defined_object=args)
    minimum = params.getLowerBounds()
    maximum = params.getUpperBounds()
    U0 = Y0[1]
    R0 = Y0[6]
    if params.nSites != 1:
        for i in range(params.nPhases):
            for j in range(params.nParams):
                for k in range(params.nSites):
                    if mask[i,j,k]:
                        mcstat.parameters.add_model_parameter(
                            name=str(k)+names[j]+str(i),
                            theta0=params.get()[i,j,k],
                            minimum=0.9*params.get()[i,j,k],
                            maximum=1.1*params.get()[i,j,k])
    else:
        for i in range(params.nPhases):
            for j in range(params.nParams):
                if mask[i,j]:
                    mcstat.parameters.add_model_parameter(
                        name=names[j]+str(i),
                        theta0=params.get()[i,j],
                        minimum=0.7*params.get()[i,j],
                        maximum=1.3*params.get()[i,j])

        mcstat.parameters.add_model_parameter(
            name='omegaI_err',
            theta0=0,
            prior_mu=0,
            prior_sigma=0.1*max(params.params_time[:,3]))

        mcstat.parameters.add_model_parameter(
            name='omegaH_err',
            theta0=0,
            prior_mu=0,
            prior_sigma=0.1*max(params.params_time[:,4]))

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
        sos_function=model_ss,
        sigma2=sigma**2)
    if parallel:
        parmcstat = ParallelMCMC()
        parmcstat.setup_parallel_simulation(mcstat,num_cores=nproc,num_chain=nchains)
        parmcstat.run_parallel_simulation()
        return parmcstat

    mcstat.run_simulation()
    return mcstat
