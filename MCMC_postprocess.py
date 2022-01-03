import sys
import os.path
import numpy as np
import pandas as pd
import json
from epi import loaddata as ld
from epi.convert import converter
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
    MCMCpath = ResPath + '/MCMC/'	
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
    epi_start = pd.to_datetime('2021-02-24') 

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
    t_list = model_solver.t_list.copy()

    compartments =  [
            'Individui in isolamento domiciliare',
            'Ricoverati (non in terapia intensiva)',
            'Ricoverati in terapia intensiva',
            'Deceduti',
            'Recovered',
            'Guariti',
            'Nuovi positivi',
            'Nuovi ingressi in terapia intensiva',
            'Deceduti giornalieri',
            'Positivi',
            'Totale ospedalizzati',
            'Tasso di letalità',
            'Nuovi positivi su 7 giorni',
            'Percentuale di ospedalizzati',
      #      'Percentuale occupazione terapie intensive',
            'R*'
    ]
    n_compartments = len(compartments)
    compartments += [c+' q=0.025' for c in compartments] + [c+' q=0.975' for c in compartments] 
    results_df = pd.DataFrame(index = pd.date_range(day_init+pd.Timedelta(t_list[0], 'days'), day_init+pd.Timedelta(t_list[-1], 'days')), columns = compartments)
    intervals_dict = dict(zip(['S', 'U', 'I', 'H', 'T', 'E', 'R', 'V1', 'V2', 'V2p', 'R_d', 'N_p', 'N_t'],[inter['credible'] for inter in interval]))
    
    wanted_compartments = ['I', 'H', 'T', 'E', 'R', 'R_d', 'N_p', 'N_t']
    n_wanted_compartments = len(wanted_compartments)
    results = np.zeros((model_solver.t_list.size, n_wanted_compartments, 3)) 
    for i,comp in enumerate(wanted_compartments):
        quantiles = generate_quantiles(intervals_dict[comp], np.array([0.025, 0.5, 0.975]))
        results[:,i,:] = quantiles.T
    results_df.iloc[:,:n_wanted_compartments] = results[...,1]
    results_df.iloc[:,n_compartments:n_compartments+n_wanted_compartments] = results[...,0]
    results_df.iloc[:,2*n_compartments:2*n_compartments+n_wanted_compartments] = results[...,2]

    deceduti_giornalieri = generate_quantiles(np.diff(intervals_dict['E'], axis=1, prepend=model_solver.data.Extinct.iloc[-2]), np.array([0.025, 0.5, 0.975]))
    results_df['Deceduti giornalieri'] = deceduti_giornalieri[1] 
    results_df['Deceduti giornalieri q=0.025'] = deceduti_giornalieri[0] 
    results_df['Deceduti giornalieri q=0.975'] = deceduti_giornalieri[2] 

    positivi = generate_quantiles(intervals_dict['I']+intervals_dict['H']+intervals_dict['T'], np.array([0.025, 0.5, 0.975]))
    results_df['Positivi'] = positivi[1] 
    results_df['Positivi q=0.025'] = positivi[0] 
    results_df['Positivi q=0.975'] = positivi[2] 

    totale_ospedalizzati = generate_quantiles(intervals_dict['H']+intervals_dict['T'], np.array([0.025, 0.5, 0.975]))
    results_df['Totale ospedalizzati'] = totale_ospedalizzati[1] 
    results_df['Totale ospedalizzati q=0.025'] = totale_ospedalizzati[0] 
    results_df['Totale ospedalizzati q=0.975'] = totale_ospedalizzati[2] 

    tasso_letale = generate_quantiles(intervals_dict['E']/(intervals_dict['E']+intervals_dict['R_d']), np.array([0.025, 0.5, 0.975]))
    results_df['Tasso di letalità'] = tasso_letale[1] 
    results_df['Tasso di letalità q=0.025'] = tasso_letale[0] 
    results_df['Tasso di letalità q=0.975'] = tasso_letale[2] 

    nuovi_positivi_settimana = generate_quantiles(pd.DataFrame(intervals_dict['N_p']).rolling(window=7).sum(), np.array([0.025, 0.5, 0.975]))
    results_df['Nuovi positivi su 7 giorni'] = nuovi_positivi_settimana[1] 
    results_df['Nuovi positivi su 7 giorni q=0.025'] = nuovi_positivi_settimana[0] 
    results_df['Nuovi positivi su 7 giorni q=0.975'] = nuovi_positivi_settimana[2] 

    percentuale_ospedalizzati = generate_quantiles((intervals_dict['H']+intervals_dict['T'])/(intervals_dict['I']+intervals_dict['H']+intervals_dict['T'])*100, np.array([0.025, 0.5, 0.975]))
    results_df['Percentuale di ospedalizzati'] = percentuale_ospedalizzati[1] 
    results_df['Percentuale di ospedalizzati q=0.025'] = percentuale_ospedalizzati[0] 
    results_df['Percentuale di ospedalizzati q=0.975'] = percentuale_ospedalizzati[2] 

    logI = pd.DataFrame(np.log(intervals_dict['I']+intervals_dict['H']+intervals_dict['T']))
    r_star = generate_quantiles((logI.diff(periods=7)/7*9+1).to_numpy(), np.array([0.025, 0.5, 0.975]))
    results_df['R*'] = r_star[1] 
    results_df['R* q=0.025'] = r_star[0] 
    results_df['R* q=0.975'] = r_star[2] 

    results_df.index = results_df.index.strftime('%Y-%m-%d')
    results_df.rename_axis('Date', inplace=True)
    results_df.reset_index(inplace=True)
    results_df.to_json(MCMCpath+'simdf_MCMC.json')

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
