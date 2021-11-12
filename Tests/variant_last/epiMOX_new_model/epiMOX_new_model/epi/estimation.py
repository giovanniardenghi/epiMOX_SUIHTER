import numpy as np
import pandas as pd
from scipy.optimize import Bounds
from optimparallel import minimize_parallel
import matplotlib.pyplot as plt
from epi import models_test as md
from epi import parameters_const as pm

def solve_rk4(fun, t_span, y0, h, args):
    nsteps = int((t_span[1]-t_span[0])/h)
    y = np.zeros((len(y0),nsteps+1))
    y[:,0]=y0
    args[0].compute_param_over_time(int(t_span[1]))
    #rd0 = args[-1][0]
    #args[-1].clear()
    #args[-1].append(rd0)
    for i in range(nsteps):
        t = t_span[0]+i*h
        y0 = y[:,i]
        k1=fun(t      , y0         , *args)
        k2=fun(t+0.5*h, y0+0.5*h*k1, *args)
        k3=fun(t+0.5*h, y0+0.5*h*k2, *args)
        k4=fun(t+    h, y0+    h*k3, *args)
        y[:,i+1] = y0 + h*(k1+2*(k2+k3)+k4)/6.0
        args[6][i+1+int(t_span[0])] = args[6][i+int(t_span[0])]+(y[-7:-4,i+1]*args[0].params_time[i+int(t_span[0]),6:9]).sum()
    #print(args[-1])
    return y

def solve_rk4_mod(fun, t_span, y0, h, args):
    nsteps = int((t_span[1]-t_span[0])/h)
    y = np.zeros((len(y0),nsteps+1))
    y[:,0]=y0
    args[0].compute_param_over_time(int(t_span[1]))
    args1 = args
    for i in range(nsteps):
        t = t_span[0]+i*h
        args1 = (*args[0:4],args[4][int(t)],args[5][int(t)])
        args2 = (*args[0:4],args[4][int(t+1)],args[5][int(t+1)])
        y0 = y[:,i]
        k1=fun(t      , y0         , *args1)
        k2=fun(t+0.5*h, y0+0.5*h*k1, *args1)
        k3=fun(t+0.5*h, y0+0.5*h*k2, *args1)
        k4=fun(t+    h, y0+    h*k3, *args2)
        y[:,i+1] = y0 + h*(k1+2*(k2+k3)+k4)/6.0
    return y

def solve_euler(fun, t_span, y0, h, args):
    nsteps = int((t_span[1] - t_span[0]) / h)
    y = np.zeros((len(y0), nsteps + 1))
    y[:, 0] = y0
    for i in range(nsteps):
        t = t_span[0] + i * h
        y0 = y[:, i]
        y[:, i + 1] = y0 + h * fun(t,y0,*args)
    return y

def errorSUIHTER(params0, tspan, y0, method, Ns, l_args, data):
    Nstep = tspan[1]-tspan[0]
    time_list = np.arange(0,Nstep+1)
    #print('Est. params: ' +str(params0))
    l_args[0].get()[ l_args[0].getMask() ] = params0
    nSites = l_args[0].nSites
    solver = globals()[method]
    res = solver(md.SUIHTERmodel, tspan, y0, time_list[1]-time_list[0], l_args)
    #delta = l_args[0].params_time[:,2]
    delta = l_args[0].delta
    #Utot = res[Ns:2*Ns,:].sum(axis=0)
    #Itot = res[2*Ns:3*Ns,:].sum(axis=0)
    #Htot = res[3*Ns:4*Ns,:].sum(axis=0)
    #Ttot = res[4*Ns:5*Ns,:].sum(axis=0)
    #Etot = res[5*Ns:6*Ns,:].sum(axis=0)
    Utot = res[Ns:2*Ns,:].flatten('F')
    Itot = res[2*Ns:3*Ns,:].flatten('F')
    Htot = res[3*Ns:4*Ns,:].flatten('F')
    Ttot = res[4*Ns:5*Ns,:].flatten('F')
    Etot = res[5*Ns:6*Ns,:].flatten('F')
    RtotD = postProcessH(l_args[0], time_list, res[2*Ns:3*Ns,:], res[3*Ns:4*Ns,:], res[4*Ns:5*Ns,:], l_args[3]).flatten('F') + np.repeat(data[data['time']==0]['Recovered'].values,len(time_list)).squeeze()
    Rtot = res[6*Ns:7*Ns].flatten('F')
    RUD = estimateRecovered(Etot)
    #UDtot = estimateUndetected(data['Isolated'].values,data['Hospitalized'].values,data['Threatened'].values,data['Recovered'].values,data['Extinct'].values)
    #l2-error
    #data = data.groupby('time').sum()
    if 'Age' in data.columns:
        Rdata = data.sort_values(by=['Age','data'])['Recovered'].values
    else:
        Rdata = data['Recovered'].values
    errorI = Itot - data['Isolated'].values
    errorH = Htot - data['Hospitalized'].values
    errorT = Ttot - data['Threatened'].values
    #errorE = Etot - data['Extinct'].values
    errorE = np.diff(Etot) - data['Extinct'].diff().iloc[1:].rolling(window=7,min_periods=1,center=True).mean().values
    errorR = RtotD - data['Recovered'].values
    errorR2 = Rtot - RUD# Rdata
    #errorU = Utot - UDtot
    errorDeltaU = delta(time_list) * Utot - data['New_positives'].rolling(window=7,min_periods=1,center=True).mean().values
    errorParams =np.diff(l_args[0].params[:,[0,9,10,11]],axis=0)**2
    #dItot = Itot[1:]-Itot[0:-1]
    #dHtot = Htot[1:]-Htot[0:-1]
    #dTtot = Ttot[1:]-Ttot[0:-1]
    #dEtot = Etot[1:]-Etot[0:-1]
    #dEtot = np.diff(res[5*Ns:6*Ns,:]).flatten()
    #dRtot = Rtot[1:]-Rtot[0:-1]
    #dIdata =  data['Isolated'].values[1:]-data['Isolated'].values[0:-1]
    #dHdata =  data['Hospitalized'].values[1:]-data['Hospitalized'].values[0:-1]
    #dTdata =  data['Threatened'].values[1:]-data['Threatened'].values[0:-1]
    #dEdata =  data['Extinct'].values[1:]-data['Extinct'].values[0:-1]
    #dRdata =  data['Recovered'].values[1:]-data['Recovered'].values[0:-1]
    #h1-error
    #derrorI = np.linalg.norm(dItot - dIdata)
    #derrorH = np.linalg.norm(dHtot - dHdata)
    #derrorT = np.linalg.norm(dTtot - dTdata)
    #derrorE = np.linalg.norm(dEtot - dEdata)
    #derrorR = np.linalg.norm(dRtot - dRdata)
    #dEdata = data.sort_values(by=['Age','data']).groupby('Age')['Extinct'].diff().dropna().values
    #errorE = np.concatenate([[0],dEtot - dEdata])
    one = np.ones(len(Itot))
    weight = np.ones(len(errorI))
    weight[int(l_args[0].times[-2]):] = 1
    #weightsU = weight/UDtot
    #weightsU = 0
    weightsNP = weight/np.maximum(data['New_positives'].values,one)
    weightsI = weight/np.maximum(data['Isolated'].values,one)
    weightsH = weight/np.maximum(data['Hospitalized'].values,one)
    weightsT = weight/np.maximum(data['Threatened'].values,one)
    #weightsE = weight/np.maximum(data['Extinct'].values,one)
    weightsE = weight[1:]/np.maximum(np.diff(data['Extinct'].values),one[1:])
    #weightsE = weight/np.maximum(np.concatenate([[0],dEdata]),one)
    weightsR = 0.1*weight/Rdata.max()#np.maximum(Rdata,one)
    weightsR2 = 0.0*weight/np.maximum(RUD,one)
    #weightsR = 0
    weightP = 0#/np.maximum(np.diff(l_args[0].params[:,[0,9,10,11]],axis=0),np.ones((l_args[0].params.shape[0]-1,4)))
    #print((errorParams*weightP).sum())

    errorL2 = np.sqrt(((errorI ** 2)*weightsI + (errorH ** 2)*weightsH +
                      (errorT ** 2)*weightsT + 
                      (errorDeltaU ** 2)*weightsNP +
                      (errorR ** 2)*weightsR +
                      (errorR2 ** 2)*weightsR2).sum() +
                      ((errorE ** 2)*weightsE).sum() +
                      (errorParams*weightP).sum()
                      )
    #errorH1 = np.sqrt(derrorI ** 2 + derrorH ** 2 + derrorT ** 2 + derrorE ** 2 )
    error = errorL2

    #print(((errorI ** 2)*weightsI).sum(), 
    #      ((errorH ** 2)*weightsH).sum(),
    #      ((errorT ** 2)*weightsT).sum(),
    #      ((errorE ** 2)*weightsE).sum(),
    #      ((errorDeltaU ** 2)*weightsNP).sum(),
    #      ((errorR ** 2)*weightsR).sum())

    return error

def errorSUIHTER_age(params0, tspan, y0, method, Ns, l_args, data):
    Nstep = tspan[1]-tspan[0]
    time_list = np.arange(0,Nstep+1)
    l_args[0].get()[ l_args[0].getMask() ] = params0
    nSites = l_args[0].nSites
    solver = globals()[method]
    res = solver(md.SUIHTERmodel_age, tspan, y0, time_list[1]-time_list[0], l_args)
    Itot = res[2*Ns:3*Ns,:].flatten('F')
    Htot = res[3*Ns:4*Ns,:].flatten('F')
    Ttot = res[4*Ns:5*Ns,:].flatten('F')
    Etot = res[5*Ns:6*Ns,:].flatten('F')
    Rtot = postProcessH(l_args[0], time_list, res[2*Ns:3*Ns,:], res[3*Ns:4*Ns,:], res[4*Ns:5*Ns,:], l_args[3]).flatten('F') + np.repeat(data[data['time']==0]['Recovered'].values,len(time_list)).squeeze()
    Rdata = data.sort_values(by=['Age','data'])['Recovered'].values
    
    errorI = Itot - data['Isolated'].values
    errorH = Htot - data['Hospitalized'].values
    errorT = Ttot - data['Threatened'].values
    errorE = Etot - data['Extinct'].values
    errorR = Rtot - Rdata
    #dEdata = data.sort_values(by=['Age','data']).groupby('Age')['Extinct'].diff().dropna().values
    #errorE = dEtot - dEdata
    weight = 1
    one = np.ones(len(Itot))
    weightsI = weight/np.maximum(data['Isolated'].values,one)
    weightsH = weight/np.maximum(data['Hospitalized'].values,one)
    weightsT = weight/np.maximum(data['Threatened'].values,one)
    weightsE = weight/np.maximum(data['Extinct'].values,one)
    #weightsE = weight/5#/np.maximum(dEdata,np.ones(len(dEdata)))
    weightsR = weight/np.maximum(Rdata,one)
    weightsR = 0
    
    errorL2 = np.sqrt(((errorI ** 2)*weightsI + (errorH ** 2)*weightsH +
                      (errorT ** 2)*weightsT + 
                      (errorE ** 2)*weightsE +
                      (errorR ** 2)*weightsR).sum())
                      #+ ((errorE ** 2)*weightsE).sum()) 

    return errorL2

def errorSEIRD(params0, tspan, y0, method, Ns, l_args, data):
    Nstep = tspan[1] - tspan[0]
    time_list = np.arange(0, Nstep + 1)
    l_args[0].get()[l_args[0].getMask()] = params0
    res = solve_rk4(md.SEIRDmodel, tspan, y0, time_list[1] - time_list[0], l_args)

    Itot = res[2 * Ns:3 * Ns, :].sum(axis=0)
    Rtot = res[3 * Ns:4 * Ns, :].sum(axis=0)
    Dtot = res[4 * Ns:5 * Ns, :].sum(axis=0)
    dItot = Itot[1:] - Itot[0:-1]
    dRtot = Rtot[1:] - Rtot[0:-1]
    dDtot = Dtot[1:] - Dtot[0:-1]
    dIdata = data['Infected'].values[1:] - data['Infected'].values[0:-1]
    dRdata = data['Recovered'].values[1:] - data['Recovered'].values[0:-1]
    dDdata = data['Dead'].values[1:] - data['Dead'].values[0:-1]

    errorI = np.linalg.norm((Itot - data['Infected'].values))#/data['Infected'].values)
    errorR = np.linalg.norm((Rtot - data['Recovered'].values))#/data['Recovered'].values)
    errorD = np.linalg.norm((Dtot - data['Dead'].values))#/data['Dead'].values)
    derrorI = np.linalg.norm(dItot - dIdata)
    derrorR = np.linalg.norm(dRtot - dRdata)
    derrorD = np.linalg.norm(dDtot - dDdata)

    #error = np.sqrt(errorI ** 2 + errorR ** 2 + errorD ** 2 + derrorI ** 2 + derrorD ** 2 + derrorR ** 2)
    error = np.sqrt(errorD ** 2 + derrorD ** 2)

    return error

def estimate(error, params, tspan, y0, s_method, Ns, l_args, data):
    params0 = pm.maskParams( params.get() , params.getMask() )
    lower_b = pm.maskParams( params.getLowerBounds() , params.getMask() )
    upper_b = pm.maskParams( params.getUpperBounds() , params.getMask() )
    l_bounds = Bounds( lower_b, upper_b )

    #local mimimization
    result = minimize_parallel(error, params0, bounds=l_bounds,\
            options={'ftol': 1e-15, 'maxfun':1000, 'maxiter':1000,'iprint':1},\
            args=(tspan, y0, s_method, Ns, l_args, data))
    print('###########################################')
    print(result)
    print('###########################################')

    return(result)

def postProcessH(params, time_list, I, H, T, map_to_prov):
    R = np.zeros((len(time_list),I.shape[0]))
    old = 0
    for i, t in enumerate(time_list):
        beta_U, beta_I, delta, omega_I, omega_H, rho_U, \
            rho_I, rho_H, rho_T, gamma_T, gamma_I, theta_H, theta_T = params.atTime(t).dot(map_to_prov.transpose())
        #rho_I = np.clip(params.rhoI(t),0,1)
        #rho_H = np.clip(params.rhoH(t),0,1)
        R[i] = old + (rho_I*I[:,i]) + (rho_H*H[:,i]) + (rho_T*T[:,i])
        old = R[i]
    return(R)

def estimateUndetected(I,H,T,R,E):
    window = 14 #days
    bCFR =  0.012
    dE = np.concatenate([E[window//2:window],E[window:] - E[:-window],E[-1]-E[-window:-window//2]])
    dR = np.concatenate([R[window//2:window],R[window:] - R[:-window],R[-1]-R[-window:-window//2]])
    CFR_adj = dE/(dE+dR)
    UD = (CFR_adj/bCFR-1)*(I+H+T)
    return(np.maximum(UD,np.zeros(len(UD))))

def estimateRecovered(E):
    window = 14 #days
    bCFR =  0.012
    #dE = np.diff(E) 
    R = (1/bCFR-1) * E
    return(np.maximum(R,np.zeros(len(R))))

def computeR0(params,model,DO):
    nPhases = params.get().shape[0]
    nSites = params.nSites
    np.set_printoptions(precision=8, linewidth=120)
    #print('params:\n', params.get())
    if DO.shape==(1,1 ):
        DO = np.ones([1,1])
    R0 = np.zeros((nPhases,nSites)).squeeze()
    for i in range(nPhases):
        if model == 'SUIHTER':
            beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
                rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.atPhase(i)        
            r1 = delta + rho_U 
            r2 = omega_I + rho_I + gamma_I
            r3 = omega_H + rho_H
            #R0[i] = (beta_U + beta_I*beta_U*delta/r2)/r1
            R0[i] = (beta_U * DO.sum(axis=1) + beta_I*beta_U * DO.sum(axis=1)*delta/r2)/r1
            #R0[i] = (beta_U + beta_I*delta/r2 + delta*omega_I*beta_H/(r2*r3))/r1
        elif model == 'SEIRD':
            beta, alpha, gamma, f = params.atPhase(i)
            R0[i] = beta/gamma
    return(R0)

def computeRt_const(l_args,tspan,dt,S,V1,V2,R0):
    params = l_args[0]
    Pop = l_args[1].sum()
    times = np.arange(tspan[0],tspan[1]+dt,dt)
    nPhases = params.nPhases
    nSites = params.nSites
    Rt = np.zeros((len(times), nSites)).squeeze()
    for i,t in enumerate(times):
        if True:#params.dataEnd > 0 and t > params.dataEnd:
            beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
                rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.atTime(t)
            #delta = params.delta(t)
            omega_I = params.omegaI(t)
            rho_U *= 1 - 8*delta
            r1 = delta + rho_U
            r2 = omega_I + rho_I + gamma_I
            R0_tmp = (beta_U + beta_I*beta_U*delta/r2)/r1
            #if t>=209:
            #    Rt[i] = R0_tmp * (S[i]*0.4054/0.2985 + 0.3*V1[i]*0.5265/0.3909 + 0.12*V2[i]*0.5265/0.3909) / Pop
            #else:
            Rt[i] = R0_tmp * (S[i] + 0.3*V1[i] + 0.12*V2[i]) / Pop
        else:
            p = params.getPhase(0,t)
            Rt[i] = R0[p] * (S[i] + 0.3*V1[i] + 0.12*V2[i]) / Pop
    return (Rt)

def computeRt_lin(l_args,tspan,dt,S,R0): #To be used with linear parameters
    params = l_args[0]
    Pop = l_args[1].sum()
    times = np.arange(tspan[0],tspan[1]+dt,dt)
    nSites = params.nSites
    Rt = np.zeros((len(times), nSites)).squeeze()
    for i,t in enumerate(times):
        beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.atTime(t)
        r1 = delta + rho_U
        #r2 = omega_I + rho_I
        r2 = omega_I + rho_I + gamma_I
        r3 = omega_H + rho_H
        Rt[i] = (beta_U + beta_I*beta_U*delta/r2)/r1
        Rt[i] *= S[i] / Pop
    return (Rt)

