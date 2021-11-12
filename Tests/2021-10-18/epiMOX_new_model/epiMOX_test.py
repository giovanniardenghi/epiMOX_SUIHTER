import sys
import numpy as np
import pandas as pd
import networkx as nx
import os.path
from epi import models_test as md
from epi import loaddata as ld
from epi import estimation as es
from epi.convert import converter
import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
import zipfile
import importlib
import scipy.interpolate as si

def zipdir(path, ziph):
    # ziph is zipfile handle
    exclude = ['Tests','__pycache__','venv','.git','.gitignore']
    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

def correct_isolated_model(df, comp, date, DT, C=None, total=None, healed=None):
    if not C:
        C = df.loc[date,comp]-df.loc[date-pd.Timedelta(1,'day'),comp]
    Itot = df.loc[date-pd.Timedelta(DT,'days'):date-pd.Timedelta(1,'day'),comp].sum()
    Cn =df.loc[date-pd.Timedelta(DT,'days'):date-pd.Timedelta(1,'day'),comp]/ Itot * C
    df.loc[date-pd.Timedelta(DT,'days'):date-pd.Timedelta(1,'day'),comp] += Cn.cumsum()
    if total:
        df.loc[date - pd.Timedelta(DT, 'days'):date - pd.Timedelta(1, 'day'), total] += Cn.cumsum()
    if healed:
        df.loc[date - pd.Timedelta(DT, 'days'):date - pd.Timedelta(1, 'day'), healed] -= Cn.cumsum()
    return df

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
        zipf = zipfile.ZipFile(testPath + '/epiMOX.zip', 'w', zipfile.ZIP_DEFLATED)
        zipdir('.', zipf)
        zipf.close()

    model, Nc, country, param_type, param_file, Tf, dt, save_code, by_age, edges_file, \
        borders_file, map_file, mobility, mixing, estim_param, DPC_start,\
        DPC_ndays, data_ext_deg, ext_deg, out_type, only_forecast\
        = ld.parsedata(DataDic)
    
    if ndays is not None:
        DPC_ndays = ndays
    if tf is not None:
        Tf = tf
    if estim_req is not None:
        estim_param = bool(estim_req)
    if ext_deg_in is not None:
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
    #sites = sites[sites['Code']=='80-89']
    sites = sites.reset_index(drop=True)

    if country == 'Regions':
        sites = sites.iloc[1:,:]
    else:
        sites = sites[sites['Name']==country]   
    geocodes = sites['Code'].to_numpy()
    Pop = sites['Pop'].to_numpy()
    #Area = sites['Area'].to_numpy()    # Add this column to all datasets
    Ns = sites.shape[0]

    # Read edges/borders file
    if mobility == "transport":
        edges = pd.read_csv(edges_file)
    elif mobility == "mixing":
        edges = pd.read_csv(borders_file)
    Ne = edges.shape[0]

    epi_start = datetime.date(year=2020,month=2,day=24)

    day_init = datetime.date.fromisoformat(DPC_start)
    day_end = datetime.date.fromisoformat(DPC_ndays)+datetime.timedelta(days=1)

    Tf_data = datetime.date.fromisoformat(Tf)
    Tf = int((Tf_data-day_init).days)

    # Read param file
    if params is None:
        params = pm.Params(day_init)
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
    if country == 'Italia':
        eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
    else:
        eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    
    eData['data'] = [x[:10] for x in eData.data]
    eData['data'] = pd.to_datetime(eData.data)

    corrections = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/corrections.csv',keep_default_na=False,na_values=['NaN'])
    if country != 'Italia':
        corrections = corrections[corrections['region']==country]
    corrections.replace({'NaN': None})
    comp_ita =  [
            'Positivi',
            'Nuovi positivi',
            'Individui in isolamento domiciliare',
            'Ricoverati (non in terapia intensiva)',
            'Ricoverati in terapia intensiva',
            'Nuovi ingressi in terapia intensiva',
            'Guariti',
            'Deceduti',
            'Deceduti giornalieri',
            'Tamponi eseguiti',
            'Nuovi tamponi eseguiti'
    ]
    old_comp = [
            'Totale positivi',
            'Nuovi positivi',
            'Isolamento domiciliare',
            'Ricoverati con sintomi',
            'Terapia intensiva',
            'Ingressi terapia intensiva',
            'Dimessi guariti',
            'Deceduti',
            'Nuovi deceduti',
            'Tamponi',
            'Nuovi tamponi',
    ]
    comp_to_old = dict(zip(comp_ita, old_comp))

    for _, row in corrections.iterrows():
        eData = correct_isolated_model(eData.set_index('data'), comp_to_old[row['comp']].replace(' ', '_').lower(),
                                      pd.to_datetime(row['date']), int(row['DT']), int(row['C']),
                                      comp_to_old[row['Positivi']].replace(' ', '_').lower(),
                                      comp_to_old[row['Guariti']].replace(' ', '_').lower()).reset_index().rename(
            columns={'index': 'data'})
    eData['data'] = eData.data.dt.strftime('%Y-%m-%dT%H:%M:%S')
    

    window = 7 
    deltaD = eData['deceduti'].diff(periods=2*window).shift(-window-13)
    deltaR = eData['dimessi_guariti'].diff(periods=2*window).shift(-window-13)
    CFR_t = deltaD/(deltaD+deltaR)
    IFR = 0.012
    #IFR_max = 0.008
    #IFR = np.arange(len(CFR_t))/(len(CFR_t)-1)*IFR_max+(1-np.arange(len(CFR_t))/(len(CFR_t)-1))*IFR_min
    Delta_t = np.clip(IFR/CFR_t,0,1)/8
    Delta_t =  (pd.Series(Delta_t[int((day_init-epi_start).days):int((day_end-epi_start).days)]).rolling(center=True,window=7,min_periods=1).mean())#.mean()
    UD = eData['nuovi_positivi'].rolling(center=True,window=7,min_periods=1).mean()/Delta_t
    UD.index=pd.date_range('2020-02-24',pd.to_datetime('2020-02-24')+pd.Timedelta(UD.index[-1],'days'))

    #CFR_t = CFR_t.rolling(center=True,window=7,min_periods=1).mean()
    #CFR_t = CFR_t[int((day_init-epi_start).days):int((day_end-epi_start).days)].values 
    #mask = np.isnan(CFR_t)
    #idx = np.where(~mask,np.arange(mask.shape[0]),0)
    #np.maximum.accumulate(idx, out=idx)
    #CFR_t[mask] = CFR_t[idx[mask]]
    #CFR = np.pad(CFR_t, (0,Tf+1-len(CFR_t)), 'edge')

    eData = eData[(eData["data"]>=day_init.isoformat()) & (eData["data"]<day_end.isoformat())]
    eData = eData.reset_index(drop=True)
    eData = converter(model, eData, country, Nc)
    eData = eData.reset_index(drop=True)

    
    params.delta = si.interp1d(range(int((day_end-day_init).days)-20),Delta_t[:-20],fill_value="extrapolate",kind='nearest')
    ric = pd.read_csv('https://raw.githubusercontent.com/floatingpurr/covid-19_sorveglianza_integrata_italia/main/data/latest/ricoveri.csv')
    #ric = pd.read_csv('https://raw.githubusercontent.com/floatingpurr/covid-19_sorveglianza_integrata_italia/main/data/2021-08-15/ricoveri.csv')
    ric = ric.iloc[:-1]
    ric['DATARICOVERO1'] = pd.to_datetime(ric['DATARICOVERO1'],format='%d/%m/%Y')
    ric.set_index('DATARICOVERO1',inplace=True)
    omegaI = pd.Series(pd.to_numeric((ric.loc[day_init+pd.Timedelta(1,'day'):day_end-pd.Timedelta(3,'day'),'RICOVERI']).rolling(center=True,window=7,min_periods=1).mean().values)/eData['Isolated'].values[:-3]).rolling(center=True,window=7,min_periods=1).mean()
    omegaH = pd.Series(eData['New_threatened'].rolling(center=True,window=7,min_periods=1).mean().values[1:]/eData['Hospitalized'].values[:-1])#.rolling(center=True,window=7,min_periods=1).mean()
    

    params.omegaI = si.interp1d(range(1,(day_end-day_init).days-2),omegaI,fill_value='extrapolate',kind='nearest')
    params.omegaH = si.interp1d(range(1,(day_end-day_init).days),omegaH,fill_value='extrapolate',kind='nearest')
    params.define_params_time(Tf)
    for t in range(Tf+1):
        params.params_time[t,2] = params.delta(t)
        params.params_time[t,3] = params.omegaI(t)
        params.params_time[t,4] = params.omegaH(t)

    if by_age:
        perc = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/stato_clinico.csv')
    #    perc = pd.read_csv('~/dpc-covid-data/main/SUIHTER/stato_clinico.csv')
        perc = perc[(perc['Data']>=DPC_start) & (perc['Data']<=DPC_ndays)] 
        eData = pd.DataFrame(np.repeat(eData.values,Ns,axis=0),columns=eData.columns)
        eData[perc.columns[2:]] = eData[perc.columns[2:]].mul(perc[perc.columns[2:]])
        eData['Age'] = perc['Età']

    initI = eData[eData['time']==0].copy()
    initI = initI.reset_index(drop=True)
    dates = pd.date_range(pd.to_datetime(initI['data'].iloc[0][0:10])-pd.Timedelta(7,'days'),pd.to_datetime(initI['data'].iloc[0][0:10])+pd.Timedelta(7,'days'))
    initI['Undetected'] = UD.loc[dates].mean()


    Recovered = (1/IFR-1)*initI['Extinct'].sum()
    initI['Recovered'] = Recovered#*initI['Recovered']/initI['Recovered'].sum()
    print('Undetected', initI['Undetected'])
    print('Recovered', initI['Recovered'])

    day_init_vaccines = day_init - datetime.timedelta(days=14)
    vaccines = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/vaccini_regioni/'+country+'.csv')
    #vaccines = pd.read_csv('~/dpc-covid-data/data/vaccini_regioni/'+country+'.csv')
    vaccines.set_index('data',inplace=True)
    vaccines.fillna(0,inplace=True)
    vaccines_init = vaccines[:(day_init_vaccines-datetime.timedelta(days=1)).isoformat()].sum()
    vaccines = vaccines.loc[day_init_vaccines.isoformat():]
    vaccines.index = pd.to_datetime(vaccines.index)
    vaccines = vaccines.reindex(pd.date_range(vaccines.index[0],pd.to_datetime(Tf_data)),columns=['prima_dose', 'seconda_dose']).ffill()
    dv1 = vaccines['prima_dose']
    dv2 = vaccines['seconda_dose']

    ### Init compartments
    sol = np.zeros(Ns*(Nc+2))
    sol[0:Ns] = Pop
    # Init compartment for the "proven cases"
    if model == 'SUIHTER':
        sol[0:Ns] = sol[0:Ns] \
                    - (initI['Undetected'].values\
                    + initI['Isolated'].values\
                    + initI['Hospitalized'].values\
                    + initI['Threatened'].values\
                    + initI['Extinct'].values
                    + initI['Recovered'].values
                    + vaccines_init['prima_dose'])
        sol[Ns:2*Ns] = initI['Undetected'].values
        sol[2*Ns:3*Ns] = initI['Isolated'].values
        sol[3*Ns:4*Ns] = initI['Hospitalized'].values
        sol[4*Ns:5*Ns] = initI['Threatened'].values
        sol[5*Ns:6*Ns] = initI['Extinct'].values
        sol[6*Ns:7*Ns] = initI['Recovered'].values
        sol[7*Ns:8*Ns] = vaccines_init['prima_dose']-vaccines_init['seconda_dose'] 
        sol[8*Ns:9*Ns] = vaccines_init['seconda_dose']
    elif model == 'SEIRD':
        sol[0:Ns] -= (initI['Infected'].values
                    + initI['Recovered'].values
                    + initI['Dead'].values)
        sol[2*Ns:3*Ns] = initI['Infected'].values
        sol[3*Ns:4*Ns] = initI['Recovered'].values
        sol[4*Ns:5*Ns] = initI['Dead'].values
    print(sol)
    ### Solve

    # Create transport matrix
    print('Creating OD matrix...')
    if by_age:
        OD = np.loadtxt('util/'+DataDic['AgeContacts'])
        #DO = DO[4,4]
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
        DO = DO - np.diag(np.diag(DO))
    print('...done!')
    #np.savetxt('DO_matrix.txt',DO,fmt='%6d')

    PopIn = DO.sum(axis=1)

    # Create output array
    #if only_forecast:
    #    T0 = int(initI['time'].values)
    #else:
    T0 = 0
   
    Nstep = int((Tf-T0)/dt)
    time_list = np.arange(0,Nstep+1)*dt + T0
    # Extrapolate data in future for calibration
    if data_ext_deg is not None:
        ndays = 10 
        params.addPhase(ndays=eData['time'].max())
        new_data = np.zeros((ndays,len(eData.columns)-1))
        new_data[:,0] = [eData['Geocode'][0]]*ndays
        new_data[:,1] = np.arange(eData['time'].max()+1,eData['time'].max()+ndays+1,1)
        for i,idx in enumerate(eData.columns[2:-1]):
            x = eData['time'][-ndays:]
            y = eData[idx][-ndays:]
            tmp = np.poly1d(np.polyfit(x,y,data_ext_deg))
            new_data[:,i+2] = tmp(new_data[:,1])
        new_data_df = pd.DataFrame(new_data)
        new_data_df['data'] = [(datetime.datetime.fromisoformat(eData['data'][0]) + datetime.timedelta(days=Dt)).isoformat() for Dt in new_data[:,1]]
        new_data_df.columns = eData.columns
        eData = eData.append(new_data_df, ignore_index=True)

    #scenari = np.array([[eData['time'].max()+5,  0.87]])
    #scenari = np.array([[eData['time'].max()+20, 0.87]])
    
    zona_gialla = np.loadtxt('gialla.txt')
    zona_arancione = np.loadtxt('arancione.txt')
    zona_rossa = np.loadtxt('rosso.txt')
    
    index_gialla = np.argmax(zona_gialla > 50)
    index_arancione = np.argmax(zona_arancione > 150)
    index_rosso = np.argmax(zona_rossa > 250)

    BB = 0.3959
    BG = 0.3794
    BA = 0.2636
    BR = 0.2301
    BGP = 0.3158

    #scenari = np.array([[eData['time'].max()+index_gialla+5, 0.87]])    # GIOVANNI PERCHE' QUI ERA 0.61 ?
    
    #scenari = np.array([[eData['time'].max()+index_gialla+5, 0.87],
    #                    [eData['time'].max()+index_arancione+5, 0.77]])

    #scenari = np.array([[eData['time'].max()+index_gialla+5, 0.87],
    #                    [eData['time'].max()+index_arancione+5, 0.77],
    #                    [eData['time'].max()+index_rosso+5, 0.61]])
    
    #scenari = np.array([[eData['time'].max()+index_gialla+5, BG/BB]])
    
    #scenari = np.array([[eData['time'].max()+index_gialla+5, BG/BB],
    #                    [eData['time'].max()+index_arancione+5, BA/BB]])

    #scenari = np.array([[eData['time'].max()+index_gialla+5, BG/BB],
    #                    [eData['time'].max()+index_arancione+5, BA/BB],
    #                    [eData['time'].max()+index_rosso+5, BR/BB]])
   
    # Scenario green-pass
    #scenari =np.array([[eData['time'].max(), 0.85],
    #                   [eData['time'].max()+10, 0.85*BGP/BB]])
    #scenari = np.array([[125, 1.2 ]])

    params.forecast(eData['time'].max(),Tf, ext_deg,scenarios=scenari)
    params.extrapolate_scenario()
    #params.vaccines_effect_gammaT()
    #params.vaccines_effect_gammaH()
    # 1. Definition of the list of parameters for each model
    # 2. Estimate the parameters for the chosen model [optional]
    # 3. Call the rk4 solver for the chosen model
    print('Simulating...')
    print(eData.columns)
    #R_d = [eData['Recovered'].iloc[0]]
    R_d = np.zeros(Nstep+1)
    R_d[0] = eData['Recovered'].iloc[0]
    l_args = (params,Pop,DO,map_to_prov,dv1,dv2,R_d)#,CFR)
    if by_age:
        model_type='_age'
    else:
        model_type=''

    if estim_param:
        print('  Estimating...')
        estParams = es.estimate(getattr(es, 'error' + model + model_type), params,
                                [0, eData["time"].max()], sol, "solve_rk4", Ns, l_args, eData)
        print('  ***  Estimated parameters: ' + str(estParams.x))
        l_args[0].get()[l_args[0].getMask()] = estParams.x
        print('  ...done!')
    print('  Solving...')
    params.params[:,l_args[0].getConstant()] = params.params[0,l_args[0].getConstant()]
    if params.nSites > 1:
        params.params[:,l_args[0].getConstantSites(),:] = params.params[:,l_args[0].getConstantSites(),0][...,np.newaxis]
    
    #res = es.solve_rk4(getattr(md, model + 'model' + model_type), [T0, Tf], sol, time_list[1]-time_list[0], l_args)

    params.forecast(eData['time'].max(),Tf, ext_deg,scenarios=scenari)
    params.extrapolate_scenario()
    
    res = es.solve_rk4(getattr(md, model + 'model' + model_type), [T0, Tf], sol, time_list[1]-time_list[0], l_args)

    R0 = es.computeR0(l_args[0], model,DO)
    print('  R0 at the different phases:', R0)
    if param_type == 'const':
        Rt = es.computeRt_const(l_args, [T0,Tf], dt, res[0:Ns,:].sum(axis=0), res[7*Ns:8*Ns,:].sum(axis=0), res[8*Ns:9*Ns,:].sum(axis=0),R0)
    else:
        Rt = es.computeRt_lin(l_args, [T0,Tf], dt, res[0:Ns,:].sum(axis=0),R0)
    np.savetxt(out_path+'/Rt.csv', Rt, delimiter=',')
    print('  ...done!')
    print('...done!')
    # Plot Rt time series
    Rt_ISS = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/iss-rt-data/main/data/iss_rt.csv')
    Rt_ISS['Data'] = pd.to_datetime(Rt_ISS.Data)
    Rt_ISS.set_index('Data',inplace=True)
    plt.plot([day_init + datetime.timedelta(days=int(x)) for x in range(T0,Tf+1)], Rt, linewidth = 4, label = 'Rt SUIHTER')
    plt.plot(Rt_ISS[day_init:'2021-05-24'].index,Rt_ISS[day_init:'2021-05-24'], linewidth = 4, label = 'Rt ISS' )
    plt.legend(fontsize=20)
    fig = plt.gcf()
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize='large')
    ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=10))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    fig.set_size_inches((19.2, 10.8), forward=False)
    plt.savefig(out_path + '/Rt_plot.png', dpi=300)
    plt.close()
    
    # Forecast from data
    if only_forecast:
        params.vaccines_effect_omega()
        dv1[0]+=vaccines_init['prima_dose']
        dv2[0]+=vaccines_init['seconda_dose']
        #args[-1].clear()
        #args[-1].append(eData['Recovered'].iloc[-1])
        #omegaI = []
        #for t in range(int(eData.time.max()),250):
        #    omegaI.append(params.omegaI(t) * params.omegaI_vaccines(t))
        #plt.plot(params.omegaI(range(int(eData.time.max()),250))*params.omegaI_vaccines(range(int(eData.time.max()),250)))
        #plt.plot(omegaI)
        #plt.show()
        #return
        variant_perc = 1#0.327 
        variant_factor = 1.5
        kappa1 = 1#33.5 / 51.1 
        kappa2 =1# 80.9 / 86.8 
        
        initI = eData[eData['time']==eData['time'].values[-1]].copy()
#        initI['New_positives'] = eData['New_positives'].iloc[-7:].mean()
        T0 = int(initI['time'].values)
        Nstep = int((Tf-T0)/dt)
        time_list = np.arange(0,Nstep+1)*dt + T0

        l_args[-1][T0] = eData['Recovered'].iloc[-1]
        
        initI['Undetected'] = res[Ns:2*Ns,T0] 
        
        initI['Undetected_base'] = (1 - variant_perc) * initI['Undetected']
        initI['Undetected_variant'] = variant_perc * initI['Undetected']

        #print(params.params)
        #params.params[0,2] = initI['New_positives'].values/res[Ns:2*Ns,T0]
        #print(params.params)

        #params.forecast(eData['time'].max(),Tf, ext_deg,scenarios=scenari)

        sol = np.zeros(Ns*(Nc+3))
        sol[0:Ns] = Pop
        # Init compartment for the "proven cases"

        if model == 'SUIHTER':
            #sol[Ns:2*Ns] = initI['New_positives'].values/params.params[0,2] 
            sol[Ns:2*Ns] = initI['Undetected_base'].values 
            sol[2*Ns:3*Ns] = initI['Undetected_variant'].values
            sol[3*Ns:4*Ns] = initI['Isolated'].values
            sol[4*Ns:5*Ns] = initI['Hospitalized'].values
            sol[5*Ns:6*Ns] = initI['Threatened'].values
            sol[6*Ns:7*Ns] = initI['Extinct'].values
            sol[7*Ns:8*Ns] = res[6*Ns:7*Ns,T0]
            sol[8*Ns:9*Ns] = res[7*Ns:8*Ns,T0]
            sol[9*Ns:] = res[8*Ns:,T0] 
            sol[0:Ns] = sol[0:Ns] \
                    - sol[Ns:2*Ns] \
                    - sol[2*Ns:3*Ns] \
                    - sol[3*Ns:4*Ns] \
                    - sol[4*Ns:5*Ns] \
                    - sol[5*Ns:6*Ns] \
                    - sol[6*Ns:7*Ns] \
                    - sol[7*Ns:8*Ns] \
                    - sol[8*Ns:9*Ns] \
                    - sol[9*Ns:] 


        res = es.solve_rk4(getattr(md, model + 'model' + model_type + '_variant'), [T0, Tf], sol, time_list[1]-time_list[0], l_args + (variant_perc, variant_factor, kappa1, kappa2))
        print(res[2]/(res[1]+res[2]))
        res = np.vstack([res[0:Ns],np.sum(res[Ns:3*Ns],axis=0),res[3*Ns:]])
        #sol = np.delete(sol,1,0)
        #res = es.solve_rk4(getattr(md, model + 'model' + model_type ), [T0, Tf], sol, time_list[1]-time_list[0], l_args)
    ### Post-process

    print('Reorganizing and saving results...')

    # Save parameters
    if estim_param == True:
        params.estimated=True
        params.save(str(out_path)+ '/param_est_d' +str(DPC_ndays) + '-' + country + model_type + '.csv')

    results = np.zeros((Ns*(Nstep+1),7+Nc),dtype='O')
    fluxes = np.zeros((Ns*(Nstep+1),(2+Nc)*2-2))
    tt = 0
    # Create array with (normalized) results
    for t in range(Nstep+1):
        t=int(t)
        t_index_l = int(Ns*t)
        t_index_u = int(Ns*t + (Ns-1) +1)
        results[t_index_l:t_index_u,0] = geocodes
        results[t_index_l:t_index_u,1] = tt
        for c in range(Nc+2):
            #results[t_index_l:t_index_u,2+c] = sol[c*Ns:(c+1)*Ns] #Euler
            results[t_index_l:t_index_u,2+c] = res[c*Ns:(c+1)*Ns,t]
        if t>0:
            results[t_index_l:t_index_u,-2] = res[Ns:2*Ns,t-1] * params.delta(time_list[t])
            results[t_index_l:t_index_u,-1] = res[3*Ns:4*Ns,t-1] * params.omegaH(time_list[t])
            #results[t_index_l:t_index_u,-2] = res[Ns:2*Ns,t-1] * params.atTime(t).dot(map_to_prov.transpose())[2]
            #results[t_index_l:t_index_u,-1] = res[3*Ns:4*Ns,t-1] * params.atTime(t).dot(map_to_prov.transpose())[4]
            fluxes[t_index_l:t_index_u,0] = (res[:Ns,t-1] + 0.35 * res[7*Ns:8*Ns,t-1] + 0.3 * res[8*Ns:9*Ns,t-1]) * params.atTime(t).dot(map_to_prov.transpose())[0] *res[Ns:2*Ns,t-1] / Pop  # U in
            fluxes[t_index_l:t_index_u,1] = res[Ns:2*Ns,t-1] #* params.delta(t) # I in
            fluxes[t_index_l:t_index_u,2] = res[2*Ns:3*Ns,t-1] * params.omegaI(t) + res[4*Ns:5*Ns,t-1] * params.atTime(t).dot(map_to_prov.transpose())[12] # H in
            fluxes[t_index_l:t_index_u,3] = res[3*Ns:4*Ns,t-1] * params.omegaH(t) # T in
            fluxes[t_index_l:t_index_u,4] = res[5*Ns:6*Ns,t] - res[5*Ns:6*Ns,t-1] # E in
            fluxes[t_index_l:t_index_u,5] = (res[Ns:2*Ns,t-1] * params.atTime(t).dot(map_to_prov.transpose())[5:8]).sum() # R in
            fluxes[t_index_l:t_index_u,6] = dv1[t]  # V1 in
            fluxes[t_index_l:t_index_u,7] = dv2[t]  # V2 in

            #fluxes[t_index_l:t_index_u,8] = res[:Ns,t-1] * params.atTime(t).dot(map_to_prov.transpose())[0] * res[Ns:2*Ns,t-1] / Pop + dv1[t] * res[:Ns,t-1]/(res[:Ns,t-1]+res[6*Ns:7*Ns,t-1]) # S out 
            #fluxes[t_index_l:t_index_u,9] = res[Ns:2*Ns,t-1] * (params.delta(t) + params.atTime(t).dot(map_to_prov.transpose())[5]*(1-8*params.delta(t))) # U out
            #fluxes[t_index_l:t_index_u,10] =  res[2*Ns:3*Ns,t-1] * (params.omegaI(t) + params.rhoI(t) + params.atTime(t).dot(map_to_prov.transpose())[11]*params.gammaH(t)*res[3*Ns:4*Ns,t-1]/res[2*Ns:3*Ns,t-1]) # I out
            #fluxes[t_index_l:t_index_u,11] = res[3*Ns:4*Ns,t-1] * (params.omegaH(t) + params.rhoH(t) + (1-params.atTime(t).dot(map_to_prov.transpose())[11])*params.gammaH(t)) # H out
            #fluxes[t_index_l:t_index_u,12] = res[4*Ns:5*Ns,t-1] * (params.gammaT(t) + params.atTime(t).dot(map_to_prov.transpose())[12]) # T out 
            #fluxes[t_index_l:t_index_u,13] = dv1[t] * res[6*Ns:7*Ns,t-1]/(res[:Ns,t-1]+res[6*Ns:7*Ns,t-1]) # R Out 
            #fluxes[t_index_l:t_index_u,14] = dv2[t] + 0.35 * res[7*Ns:8*Ns,t-1] * params.atTime(t).dot(map_to_prov.transpose())[0] *res[Ns:2*Ns,t-1] / Pop # V1 out 
            #fluxes[t_index_l:t_index_u,15] = 0.3 * res[8*Ns:9*Ns,t-1] * params.atTime(t).dot(map_to_prov.transpose())[0] *res[Ns:2*Ns,t-1] / Pop # V2 out

        else:
            results[t_index_l:t_index_u,-2] = initI['New_positives']
            results[t_index_l:t_index_u,-1] = initI['New_threatened']
        tt = tt + dt

    results[:,-3] = es.postProcessH(params, time_list, res[2*Ns:3*Ns,:], res[3*Ns:4*Ns,:], res[4*Ns:5*Ns,:], map_to_prov).flatten() + np.tile(eData[eData['time']==T0]['Recovered'].values,len(time_list)).squeeze()
    #print(res[3,1:]*params.params[-1,-3]/np.diff(res[5]))

    compartments = np.array(['Suscept','Undetected','Isolated','Hospitalized','Threatened','Extinct','Recovered','First_dose','Second_dose'],dtype='O')

    # Save the results in a dataframe
    if model == 'SUIHTER':
        results_df = pd.DataFrame(results,columns=['Geocode','time','Suscept','Undetected','Isolated',
            'Hospitalized','Threatened','Extinct','Recovered','First_dose','Second_dose','Recovered_detected','New_positives','New_threatened'])
        results_df[compartments[1:]+'_in'] = fluxes[:,:8]
        results_df[compartments[np.arange(len(compartments))!=5]+'_out'] = fluxes[:,8:]

    elif model == 'SEIRD':
        results_df = pd.DataFrame(results,columns=['Geocode','time','Suscept','Exposed','Infected',
            'Recovered','Dead'])

    if not by_age:
        results_df = results_df.astype({"Geocode": int,"time": 'float64'})
    else:
        results_df = results_df.astype({"Geocode": str,"time": 'float64'})
    results_df = results_df.sort_values(by=['Geocode','time'])
    results_df = results_df.astype(dict(zip(['Suscept','Undetected','Isolated','Hospitalized','Threatened','Extinct','Recovered','First_dose','Second_dose','Recovered_detected','New_positives','New_threatened'],['float64']*12)))
    if only_forecast:
        results_df['date'] = pd.date_range(DPC_ndays,periods=len(results_df))
    else:
        results_df['date'] = pd.date_range(DPC_start,periods=len(results_df))
    if out_type == 'csv':
        csv_outFileName = out_path+'/simdf.csv'
        results_df.to_csv(csv_outFileName,index=False)
    elif out_type == 'h5':
        h5_outFileName = out_path+'/simdf.h5'
        results_df.to_hdf(h5_outFileName, key='results_df', mode='w')
    print('...done!')
    gialla = (results_df['New_positives'].rolling(window=7,min_periods=1).sum()/Pop*1e5).values
    #np.savetxt('gialla.txt',gialla)
    #np.savetxt('arancione.txt',gialla)
    #np.savetxt('rosso.txt',gialla)
    return Ns,Nc,sol,model,params, Pop, DO, map_to_prov, dv1, dv2, dt

### Pre-process data
if __name__=='__main__':
    # Read data file
    testPath = sys.argv[1]
    epiMOX(testPath)
