import sys
import numpy as np
import pandas as pd
import networkx as nx
import os.path
import importlib
import json
import scipy.interpolate as si
from epi import loaddata as ld
from epi.convert import converter
from util.utilfunctions import *

def epiMOX(testPath,params=None,ndays=None,tf=None,estim_req=None,ext_deg_in=None,scenari=None):
    # Parse data file
    fileName = testPath + '/input.inp' 
    if os.path.exists(fileName):
        DataDic = ld.readdata(fileName)
    else:
        sys.exit('Error - Input data file ' +fileName+ ' not found. Exit.')
    
    if 'save_code' in DataDic.keys():
        save_code = bool(int(DataDic['save_code']))
    else:
        save_code = True

    if save_code:
        # copy all the code in a crompessed format to the test case folder.
        with zipfile.ZipFile(testPath + '/epiMOX.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipdir('.', zipf)

    model, Nc, country, param_type, param_file, Tf, dt, save_code, by_age, edges_file, \
        borders_file, map_file, mobility, mixing, estim_param, DPC_start,\
        DPC_end, data_ext_deg, ext_deg, out_type, restart_file, day_restart, only_forecast, scenario\
        = ld.parsedata(DataDic)
    
    if ndays:
        DPC_end = ndays
    if tf:
        Tf = tf
    if estim_req:
        estim_param = bool(estim_req)
    if ext_deg_in:
        ext_deg = ext_deg_in 
    
    pm = importlib.import_module('epi.parameters_'+param_type)
    param_file = testPath + '/' + param_file
    if by_age:
        sites_file = './util/Eta_Italia_sites.csv'
    else:
        sites_file = './util/Regioni_Italia_sites.csv'
    edges_file = './util/'+edges_file
    borders_file = './util/'+borders_file
    map_file = './util/'+map_file
    out_path = testPath 

    # Check paths
    if not os.path.exists(param_file):
        sys.exit('Error - File ' +param_file+ ' not found. Exit.')
    if not os.path.exists(sites_file):
        sys.exit('Error - File ' +sites_file+ ' not found. Exit.')
    if mobility == "transport" and not os.path.exists(edges_file):
        sys.exit('Error - File ' +edges_file+ ' not found. Exit.')
    if mobility == "mixing" and not os.path.exists(borders_file):
        sys.exit('Error - File ' +borders_file+ ' not found. Exit.')

    # Read sites file
    sites = pd.read_csv(sites_file)
    sites = sites.sort_values("Code")
    sites = sites.reset_index(drop=True)

    if country == 'Regions':
        sites = sites[sites['Name']!='Italia']
    else:
        sites = sites[sites['Name']==country]   
    geocodes = sites['Code'].to_numpy()
    Pop = sites['Pop'].to_numpy()
    Ns = sites.shape[0]

    # Read edges/borders file
    if mobility == "transport":
        edges = pd.read_csv(edges_file)
    elif mobility == "mixing":
        edges = pd.read_csv(borders_file)
    Ne = edges.shape[0]

    epi_start = pd.to_datetime('2020-02-24')

    day_init = pd.to_datetime(DPC_start)
    day_end = pd.to_datetime(DPC_end)

    Tf_data = pd.to_datetime(Tf)
    Tf = int((Tf_data-day_init).days)

    # Read param file
    if params is None:
        params = pm.Params(day_init, (day_end-day_init).days)
        params.load(param_file)

    map_to_prov=np.array(1)

    if params.nSites != 1 and params.nSites != Ns:
        if not os.path.exists(map_file):
            sys.exit('Error - File ' +map_file+ ' not found. Exit.')
        else:
            map_to_prov=pd.read_csv(map_file, header=None, sep=' ', dtype=int).to_numpy()
            if map_to_prov.shape[1] != params.nSites:
                sys.exit("Error - File " + map_file + " number of columns doesn't match number of parameters sites")

    # Read data for estimation
    if country == 'Regions':
        eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    else:
        eData = pd.read_csv("https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/regioni/"+country.replace(' ','%20')+".csv")
    
    eData['data'] = [x[:10] for x in eData.data]
    eData['data'] = pd.to_datetime(eData.data)
    
    eData = correct_data(eData, country)
    
    if model == 'SUIHTER':
        IFR_t = estimate_IFR(country, Pop)
        CFR_t = estimate_CFR(eData)
        Delta_t = params.compute_delta(IFR_t, CFR_t, day_end)

        UD = eData['nuovi_positivi'].rolling(center=True,window=7,min_periods=1).mean()/Delta_t
        UD.index=pd.date_range(epi_start,epi_start+pd.Timedelta(UD.index[-1],'days'))

    eData = eData[(eData["data"]>=day_init.isoformat()) & (eData["data"]<=day_end.isoformat())]
    eData = eData.reset_index(drop=True)
    eData = converter(model, eData, country, Nc)
    eData = eData.reset_index(drop=True)
    
    if model == 'SUIHTER':
        if country=='Italia':
            ric = pd.read_csv('https://raw.githubusercontent.com/floatingpurr/covid-19_sorveglianza_integrata_italia/main/data/latest/ricoveri.csv')
            #ric = pd.read_csv('https://raw.githubusercontent.com/floatingpurr/covid-19_sorveglianza_integrata_italia/main/data/2021-10-17/ricoveri.csv')
            ric = ric.iloc[:-1]
            ric['DATARICOVERO1'] = pd.to_datetime(ric['DATARICOVERO1'],format='%d/%m/%Y')
            ric.set_index('DATARICOVERO1',inplace=True)
            offset = 3 if day_end in ric.index else 4
            omegaI = pd.Series(pd.to_numeric((ric.loc[day_init:day_end-pd.Timedelta(offset,'day'),'RICOVERI']).rolling(center=True,window=7,min_periods=1).mean().values)/eData['Isolated'].values[:-offset]).rolling(center=True,window=7,min_periods=1).mean()
            params.omegaI = si.interp1d(range((day_end-day_init).days-offset+1),omegaI,fill_value='extrapolate',kind='nearest')
        else:
            params.omegaI_vec = np.loadtxt('omegaI.txt')
        omegaH = pd.Series(eData['New_threatened'].rolling(center=True,window=7,min_periods=1).mean().values/eData['Hospitalized'].values).bfill()#.rolling(center=True,window=7,min_periods=1).mean()

        params.omegaH = si.interp1d(range((day_end-day_init).days+1),omegaH,fill_value='extrapolate',kind='nearest')
        params.define_params_time(Tf)
        for t in range(Tf+1):
            params.params_time[t,2] = params.delta(t)
            if country=='Italia':
                params.params_time[t,3] = params.omegaI(t)
            else:
                params.params_time[t,3] = params.omegaI_vec[t]
            params.params_time[t,4] = params.omegaH(t)
        
        if country=='Italia':
            params.omegaI_vec = np.copy(params.params_time[:,3])
            np.savetxt('omegaI.txt',params.omegaI_vec)

        params.omegaH_vec = np.copy(params.params_time[:,4])
        np.savetxt('omegaH.txt',params.omegaH_vec)
    else:
        params.define_params_time(Tf)

    if by_age:
        perc = pd.read_csv(f'~/dpc-covid-data/SUIHTER/stato_clinico_{model}.csv')
        #perc = pd.read_csv(f'https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/stato_clinico_{model}.csv')
        perc = perc[(perc['Data']>=DPC_start) & (perc['Data']<=DPC_end)] 
        eData = pd.DataFrame(np.repeat(eData.values,Ns,axis=0),columns=eData.columns)
        eData[perc.columns[2:]] = eData[perc.columns[2:]].mul(perc[perc.columns[2:]].values)
        eData['Age'] = perc['Età'].values
        eData.sort_values(by=['Age','time'])

    initI = eData[eData['time']==0].copy()
    initI = initI.reset_index(drop=True)

    if model == 'SUIHTER':
        dates = pd.date_range(initI['data'].iloc[0]-pd.Timedelta(7,'days'),initI['data'].iloc[0]+pd.Timedelta(7,'days'))

        if Ns > 1:
            initI['Undetected'] = UD.loc[dates].groupby(level=1).mean()
        else:
            initI['Undetected'] = UD.loc[dates].mean()

        Recovered = (1/IFR_t.loc[day_init]-1)*initI['Extinct'].sum()
        initI['Recovered'] = Recovered

        day_init_vaccines = day_init - pd.Timedelta(14, 'days')
        vaccines = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/vaccini_regioni/'+country+'.csv')
        #vaccines = pd.read_csv('~/dpc-covid-data/data/vaccini_regioni/'+country+'.csv')
        vaccines['data'] = pd.to_datetime(vaccines.data)
        vaccines.set_index('data',inplace=True)
        vaccines.fillna(0,inplace=True)
        #vaccines[['prima_dose','seconda_dose']]=0
        vaccines_init = vaccines[:day_init_vaccines-pd.Timedelta(1,'day')].sum()
        vaccines = vaccines.loc[day_init_vaccines:]
        vaccines.index = pd.to_datetime(vaccines.index)
        vaccines = vaccines.reindex(pd.date_range(vaccines.index[0],pd.to_datetime(Tf_data)),columns=['prima_dose', 'seconda_dose', 'terza_dose']).ffill()
        maxV = 54009901
        #maxV = pd.read_csv('https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/platea.csv').totale_popolazione.sum()

        gp_from_test = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/gp_from_test.csv')
        gp_from_test = gp_from_test[gp_from_test['data']>=DPC_start] 
        tamponi = gp_from_test.issued_for_tests.values

    ### Init compartments
    # Init compartment for the "proven cases"
    if model == 'SUIHTER':
        Y0 = np.zeros((Nc+3+1,Ns)).squeeze() # 3 vaccinated compartments
        if restart_file:
            df_restart = pd.read_hdf(restart_file)
            init_restart = df_restart[df_restart['date']<=day_restart].iloc[:,3:14].values
            Y0[:2] = init_restart[-1,:2]
            Y0[2] = 0
            Y0[3:] = init_restart[-1,2:-1]
            R_d = init_restart[:,-1]
            Sfrac = np.zeros(init_restart.shape[0])
            Sfrac[0] = init_restart[0,0] / ( init_restart[0,0] + init_restart[0,-5] - init_restart[0,-1] )
            Sfrac[1:] = init_restart[:-1,0] / ( init_restart[:-1,0] + init_restart[:-1,-5] - init_restart[:-1,-1] )

        else:
            Y0[0] = Pop
            Y0[0] = Y0[0] \
                        - (initI['Undetected'].values\
                        + initI['Isolated'].values\
                        + initI['Hospitalized'].values\
                        + initI['Threatened'].values\
                        + initI['Extinct'].values
                        + initI['Recovered'].values
                        + vaccines_init['prima_dose'])
            Y0[1] = initI['Undetected'].values
            Y0[2] = 0 
            Y0[3] = initI['Isolated'].values
            Y0[4] = initI['Hospitalized'].values
            Y0[5] = initI['Threatened'].values
            Y0[6] = initI['Extinct'].values
            Y0[7] = initI['Recovered'].values
            Y0[8] = vaccines_init['prima_dose']-vaccines_init['seconda_dose'] 
            Y0[9] = vaccines_init['seconda_dose']
            Y0[10] = 0 
    ### Solve

    # Create transport matrix
    print('Creating OD matrix...')
    if by_age:
        OD = np.loadtxt('util/'+DataDic['AgeContacts'])
        DO = OD.T
    else:
        nodelist = list(sites['Code'])
        if mobility == "transport":
            nxgraph = nx.from_pandas_edgelist(edges,source='Origin_Code',
                target='Destination_Code',edge_attr='Flow',create_using=nx.DiGraph())
            OD = np.array(nx.to_numpy_matrix(nxgraph,nodelist=nodelist,weight="Flow"))
        else:
            nxgraph = nx.from_pandas_edgelist(edges,source='Origin_Code',
                target='Destination_Code',edge_attr='Border')
            OD = np.array(nx.to_numpy_matrix(nxgraph,nodelist=nodelist,weight="Border"))

        DO = OD.T
        DO = DO# - np.diag(np.diag(DO))
    print('...done!')

    PopIn = DO.sum(axis=1)

    if restart_file:
        T0 = (day_restart - day_init).days
        idx = np.concatenate((params.times,[(day_end-day_init).days]))<T0
        idy = ~params.getConstant()
        params.mask[np.ix_(idx, idy)] = 0 
    else:
        T0 = 0
   
    time_list = np.arange(T0,Tf+1)

    # 1. Definition of the list of parameters for each model
    # 2. Estimate the parameters for the chosen model [optional]
    # 3. Call the rk4 solver for the chosen model

    md = importlib.import_module('epi.'+model) 

    if by_age:
        model_type='_age'
    else:
        model_type=''
    model_class = getattr(md, model)
    
    print('Simulating...')

    if estim_param:
        print('  Estimating...')
        start_calibration_day = day_restart if day_restart else day_init
        if model == 'SUIHTER':
            model_solver = model_class(Y0, params, time_list[:(day_end-start_calibration_day).days+1], day_init, day_end, eData.iloc[T0:], Pop,
                       by_age, geocodes, vaccines, maxV, out_path, tamponi=tamponi, scenario=None, out_type=out_type)
        if restart_file:
            model_solver.Sfrac[:T0+1] = Sfrac
            model_solver.R_d[:T0+1] = R_d
        else:
            model_solver = model_class(Nc, params, time_list[:(day_end-start_calibration_day).days+1], day_init, day_end, eData.iloc[T0:], Pop,
                       by_age, geocodes, DO, out_path, out_type=out_type)

        model_solver.estimate()
        print('  ...done!')
    print('  Solving...')
    params.params[:,params.getConstant()] = params.params[0,params.getConstant()]
    if params.nSites > 1:
        params.params[:,params.getConstantSites(),:] = params.params[:,params.getConstantSites(),0][...,np.newaxis]
    
    params.forecast(eData['time'].max(),Tf, ext_deg,scenarios=scenari)
    params.extrapolate_scenario()

    if model == 'SUIHTER':
        model_solver = model_class(Y0, params, time_list, day_init, day_end, eData.iloc[T0:], Pop,
                       by_age, geocodes, vaccines, maxV, out_path, tamponi=tamponi, scenario=scenario, out_type=out_type)
        if restart_file:
            model_solver.R_d[:T0+1] = R_d
            model_solver.Sfrac[:T0+1] = Sfrac
    else:
        model_solver = model_class(Nc, params, time_list, day_init, day_end, eData.iloc[T0:], Pop,
                       by_age, geocodes, DO, out_path, out_type=out_type)
    
    model_solver.solve()

    print('  ...done!')
    print('...done!')
    
    # Forecast from data
    if only_forecast and __name__=='__main__':
        vaccines['prima_dose'].iloc[0]+=vaccines_init['prima_dose']
        vaccines['seconda_dose'].iloc[0]+=vaccines_init['seconda_dose']
        vaccines['terza_dose'].iloc[0]+=vaccines_init['terza_dose']

        initI = eData.iloc[-1].copy()
        T0 = int(initI['time'])
        time_list = np.arange(T0, Tf+1)
        
        # Init compartments"
        
        variant_prevalence = float(DataDic['variant_prevalence']) if 'variant_prevalence' in DataDic.keys() else 0
        if 'variant' in DataDic.keys() and DataDic['variant']:
            with open('util/variant_db.json') as variant_db:
                variants = json.load(variant_db)
            variant = variants[DataDic['variant']]
            model_solver.initialize_variant(variant, variant_prevalence)
        else: # No new variant spreading
            variant_prevalence = 0
        
        Y0 = model_solver.Y[...,T0].copy()
        if model == 'SUIHTER':
            Y0[2] = Y0[1] * variant_prevalence
            Y0[1] *= 1 - variant_prevalence
            Y0[3] = initI['Isolated']
            Y0[4] = initI['Hospitalized']
            Y0[5] = initI['Threatened']
            Y0[6] = initI['Extinct']


        model_solver.Y0 = Y0
        model_solver.t_list = time_list
        model_solver.forecast = True
        with open('util/scenarios.json','r') as scen_file:
            scenarios = json.load(scen_file) 
        model_solver.initialize_scenarios(scenarios)
        model_solver.solve()
    #model_solver.computeRt()


    # Save parameters
    if estim_param == True:
        params.estimated=True
        params.save(str(out_path)+ '/param_est_d' +str(DPC_end) + '-' + country + model_type + '.csv')

    return model_solver

### Pre-process data
if __name__=='__main__':
    # Read data file
    testPath = sys.argv[1]
    model_solver = epiMOX(testPath)
    model_solver.save()
