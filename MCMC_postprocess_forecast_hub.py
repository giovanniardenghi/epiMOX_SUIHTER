import sys
import os.path
import numpy as np
import pandas as pd
import json
from epi import loaddata as ld
from epiMOX_class import epiMOX
from pymcmcstat.MCMC import DataStructure
from pymcmcstat.propagation import calculate_intervals, generate_quantiles
import pymcmcstat.chain.ChainProcessing as chproc


def MCMC_postprocess(ResPath, nsample=500, burnin=None, forecast=True, scenario=None):
    InputFileName = ResPath + '/input.inp'

    if os.path.exists(InputFileName):
        DataDic = ld.readdata(InputFileName)
        epi_model = DataDic['model']
    else:
        sys.exit('Error - Input data file ' + InputFileName + ' not found. Exit.')
    MCMCpath = ResPath + '/MCMC'	
    if os.path.exists(ResPath):
        ResultsDict = chproc.load_serial_simulation_results(MCMCpath, extension='txt')
        ResultsDict['theta'] = np.array(ResultsDict['theta'])
        chain = ResultsDict['chain']
        s2chain = ResultsDict['s2chain']
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
    model_solver = epiMOX(ResPath,scenari=scenario)
    t_list = model_solver.t_list.copy()
    Y0 = model_solver.Y0.copy()

    epi_start = datetime.date(year=2020, month=2, day=24)

    day_init = pd.to_datetime(DPC_start)
    day_end = pd.to_datetime(DPC_ndays)

    mask = model_solver.params.getMask()

    variant_prevalence = float(DataDic['variant_prevalence']) if 'variant_prevalence' in DataDic.keys() else 0
    if 'variant' in DataDic.keys() and DataDic['variant']:
        with open('util/variant_db.json') as variant_db:
            variants = json.load(variant_db)
        variant = variants[DataDic['variant']]
    else:
        variant = None

    args = (forecast, variant, variant_prevalence)
    data=DataStructure()
    data.add_data_set(t_list, Y0, user_defined_object=args)

    # Calculate intervals
    interval = calculate_intervals(chain, ResultsDict, data, model_solver.model_MCMC, nsample=nsample, waitbar=True)

    print(interval[0]['credible'].shape)
    day_0 = pd.to_datetime(DPC_ndays)
    case = interval[-2]['credible']
    hospitalizations = interval[2]['credible'] * model_solver.params.params_time[-1,3]
    death = np.diff(np.hstack([model_solver.data.Extinct.iloc[-2]*np.ones((interval[5]['credible'].shape[0],1)),interval[5]['credible']]),axis=1)
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

    return data, interval
	

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

    if len(sys.argv) > 4:
        forecast = int(sys.argv[4])
    else:
        forecast = True
    
    data,interval = MCMC_postprocess(ResultsFilePath, nsample, burnin, forecast, scenario = None)
