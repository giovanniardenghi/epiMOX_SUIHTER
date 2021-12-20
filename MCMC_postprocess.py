import sys
import os.path
import numpy as np
import pandas as pd
import json
import pickle as pl
import datetime
from epi import loaddata as ld
from epi import models_test as md
from epi import estimation as es
from epi.convert import converter
from epiMOX_test import epiMOX
from epi.MCMC_test import model_fun_var_new as model_fun
import matplotlib as mpl
import matplotlib.pyplot as plt
from pymcmcstat.MCMC import DataStructure
from pymcmcstat import mcmcplot as mcp
from pymcmcstat.propagation import calculate_intervals, plot_intervals, generate_quantiles


def MCMC_postprocess(ResPath,nsample=500,burnin=None,scenario=None):
InputFileName = ResPath + '/input.inp'

if os.path.exists(InputFileName):
    DataDic = ld.readdata(InputFileName)
    epi_model = DataDic['model']
else:
    sys.exit('Error - Input data file ' + InputFileName + ' not found. Exit.')
MCMCpath = ResPath + '/MCMC'	
if os.path.exists(ResPath):
    ResultsDict = json.load(open(MCMCpath + '/results_dict.json','r'))
    ResultsDict['theta'] = np.array(ResultsDict['theta'])
    chain = np.loadtxt(MCMCpath + '/chainfile.txt')
    s2chain = np.loadtxt(MCMCpath + '/s2chainfile.txt')
else:
    sys.exit('Error - MCMC folder in ' + ResPath + ' not found. Exit.')
DPC_start = DataDic['DPC_start']
DPC_ndays = DataDic['DPC_end']
country = DataDic['country']
if country is None:
    country = 'Italia'
if burnin is None:
    burnin = chain.shape[0]//5

chain = chain[burnin:,:]
Ns, Nc, sol, model, params, Pop, DO, \
        map_to_prov, dv1, dv2, dt = epiMOX(ResPath,scenari=scenario)

epi_start = datetime.date(year=2020, month=2, day=24)

day_init = pd.to_datetime(DPC_start)
day_end = pd.to_datetime(DPC_ndays) + pd.Timedelta(1,'day')

if country == 'Italia':
    eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
else:
    eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')

Tf_data = pd.to_datetime(DataDic['Tf'])
Tf = int((Tf_data-day_init).days)

eData['data'] = pd.to_datetime(eData.data)
#eData = eData[(eData["data"] >= day_init) & (eData["data"] <= Tf_data)]
eData = eData[(eData["data"] >= day_init) & (eData["data"] < day_end)]
eData = eData.reset_index(drop=True)
eDataG = converter(epi_model, eData, country, Nc)
eDataG = eDataG.reset_index(drop=True)
#Y0 = [sol[x * Ns:(x + 1) * Ns].sum() for x in range(Nc)]
Y0 = sol


R_d = np.zeros(Tf+1)
R_d[0] = eDataG['Recovered'].iloc[0]
l_args = (params, Pop, DO, map_to_prov, dv1, dv2, R_d)
if epi_model == 'SEIRD':
    dead = eDataG.groupby('time').sum()['Dead'].values
    obs = dead
elif epi_model == 'SUIHTER':
    isolated = eDataG['Isolated'].values
    hospitalized = eDataG['Hospitalized'].values
    threatened = eDataG['Threatened'].values
    extinct = eDataG['Extinct'].values
    recovered = eDataG['Recovered'].values
    New_positives = eDataG['New_positives'].values
    obs = np.concatenate([isolated,hospitalized,threatened, extinct, recovered,New_positives], axis=0).reshape((6,-1))
nstep = int(eDataG['time'].max() / dt)
t_vals = np.arange(0, nstep + 1) * dt

mask = params.getMask()

args = [model, Y0, *l_args, scenario]
data=DataStructure()
data.add_data_set(t_vals, obs, user_defined_object=args)


xmod = np.arange(0, Tf, 1)

pdata = DataStructure()
pdata.add_data_set(x=xmod, y=data.ydata[0], user_defined_object=data.user_defined_object[0])

param_mean = np.mean(chain, 0)
params.params[params.getMask()] = param_mean[:-4]
params.estimated=True
params.save(MCMCpath + '/params_mean.csv')

params.forecast(eDataG['time'].max(),Tf,1)
if False:#params.nSites == 1:
    R0 = np.zeros((chain.shape[0],params.nPhases))
    for j,parms in enumerate(chain):
        params.params[params.getMask()] = parms[:-4]
        R0[j] = es.computeR0(params, epi_model, DO)
    R0_np = np.zeros((params.nPhases,3))
    R0_np[:,0]=range(params.nPhases)
    R0_np[:,1]=R0.mean(axis=0)
    R0_np[:,2]=R0.std(ddof=0,axis=0)
    R0_df = pd.DataFrame(R0_np,columns=["Phases","Mean","Std"])
    R0_df.to_csv(ResPath+'/R0mean.csv',index=None)
    model_mean = model_fun(param_mean, pdata)
#np.savetxt(ResultsFilePath+f'/model_mean_scen{j}',model_mean)
    l_args = (params, Pop, DO, map_to_prov)
    Rt = es.computeRt_const(l_args,[pdata.xdata[0][0],pdata.xdata[0][-1]],1,model_mean[:,0],R0.mean(axis=0))

    np.savetxt(MCMCpath+'/Rt.csv',Rt)

# Clculate intervals
interval = calculate_intervals(chain, ResultsDict, pdata, model_fun,
                                nsample=nsample, waitbar=True)
if Ns > 1:
    for i,inter in enumerate(interval):
        interval[i]['credible'] = inter['credible'].reshape(inter['credible'].shape[0],Ns,-1).sum(axis=1)
    eDataG = eDataG.groupby('data').sum()
    isolated = eDataG['Isolated'].values
    hospitalized = eDataG['Hospitalized'].values
    threatened = eDataG['Threatened'].values
    extinct = eDataG['Extinct'].values
    recovered = eDataG['Recovered'].values
    New_positives = eDataG['New_positives'].values
    obs = np.concatenate([isolated,hospitalized,threatened, extinct, recovered,New_positives], axis=0).reshape((6,-1))
    data.ydata[0] = obs



day_0 = pd.to_datetime(DPC_ndays)
case = interval[-2]['credible']
hospitalizations = interval[2]['credible'] * params.params_time[-1,3]
    death = np.diff(np.hstack([extinct[-2]*np.ones((interval[5]['credible'].shape[0],1)),interval[5]['credible']]),axis=1)
    dates = pd.date_range(day_0,day_0+pd.Timedelta(case.shape[1]-1,'days'))
    
    case = pd.DataFrame(case.transpose(),index=dates)
    hosp = pd.DataFrame(hospitalizations.transpose(),index=dates)
    death= pd.DataFrame(death.transpose(),index=dates)

    df = pd.DataFrame(columns = ["forecast_date","scenario_id","target","target_end_date","location","type","quantile","value"])
    location = 'IT'
    for i in range(4):
        for c in ['case','hosp','death']:
            target = f'{i+1} wk ahead inc {c}' 
            target_end = day_0+pd.Timedelta(6+7*i,'days')
            temp = pd.DataFrame(columns = ["forecast_date","scenario_id","target","target_end_date","location","type","quantile","value"])
            case_temp = locals()[c].loc[day_0+pd.Timedelta(7*i,'days'):target_end].sum(axis=0)
            case_quantiles = generate_quantiles(case_temp,np.concatenate([[0.5, 0.01, 0.025],np.arange(5,100,5)/100,[0.975,0.99]])) 
            quantiles = np.concatenate([['NA', 0.01, 0.025],np.arange(5,100,5)/100,[0.975,0.99]])

            temp['value'] = list(map(int,np.round(case_quantiles)))
            temp['target'] = target
            temp['target_end_date'] = target_end.strftime('%Y-%m-%d')
            temp['quantile'] = quantiles
            temp['type'] = 'quantile'
            temp['type'].iloc[0] = 'point'

            df = df.append(temp,ignore_index=True)
    df['forecast_date'] = day_end.strftime('%Y-%m-%d')
    df['location'] = location
    df['scenario_id'] = 'forecast'
    df.to_csv('forecast_hub/'+day_end.strftime('%Y-%m-%d')+'-epiMOX-SUIHTER.csv',index=None)

    return data,pdata,interval,peaks
	

if __name__ == '__main__':
    # Read data file
    if len(sys.argv) < 2:
       sys.exit('Error - at least the path containing the resulting test cases is needed')
    ResultsFilePath = sys.argv[1]
    if not os.path.exists(ResultsFilePath):
    	sys.exit('Error - Input reults folder ' + ResultsFilePath + ' not found. Exit.')

    if len(sys.argv) > 2:
        nsample = int(sys.argv[2])
    else:
        nsample = 500

    if len(sys.argv) > 3:
        burnin = int(sys.argv[3])
    else:
        burnin = None
    
    data,pdata,interval,peaks = MCMC_postprocess(ResultsFilePath, nsample, burnin,scenario = None)
