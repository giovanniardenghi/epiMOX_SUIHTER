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
from epi.MCMC_test import model_fun as model_fun
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

    eData['data'] = pd.to_datetime(eData.data)
    eData = eData[(eData["data"] >= day_init) & (eData["data"] < day_end)]
    eData = eData.reset_index(drop=True)
    eDataG = converter(epi_model, eData, country, Nc)
    eDataG = eDataG.reset_index(drop=True)
    #Y0 = [sol[x * Ns:(x + 1) * Ns].sum() for x in range(Nc)]
    Y0 = sol
    
    Tf_data = pd.to_datetime(DataDic['Tf'])
    Tf = int((Tf_data-day_init).days)
    
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
        New_threatened= eDataG['New_threatened']
        obs = np.concatenate([isolated,hospitalized,threatened, extinct, recovered,New_positives,New_threatened], axis=0).reshape((7,-1))
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
        New_threatened = eDataG['New_threatened'].values
        obs = np.concatenate([isolated,hospitalized,threatened, extinct, recovered,New_positives,New_threatened], axis=0).reshape((6,-1))
        data.ydata[0] = obs
    ciS = generate_quantiles(interval[0]['credible'],np.array([0.05,0.95]))
    ciI = generate_quantiles(interval[2]['credible'],np.array([0.05,0.95]))
    ciH = generate_quantiles(interval[3]['credible'],np.array([0.05,0.95]))
    ciT = generate_quantiles(interval[4]['credible'],np.array([0.05,0.95]))
    ciRD = generate_quantiles(interval[-1]['credible'],np.array([0.05,0.95]))
    meanI = generate_quantiles(interval[2]['credible'],np.array(0.5))
    meanH = generate_quantiles(interval[3]['credible'],np.array(0.5))
    meanT = generate_quantiles(interval[4]['credible'],np.array(0.5))
    stats = np.zeros((4,2))
    S_chain = interval[0]['credible']
    U_chain = interval[1]['credible']
    I_chain = interval[2]['credible']
    H_chain = interval[3]['credible']
    T_chain = interval[4]['credible']
    E_chain = interval[5]['credible']
    dE_chain = E_chain[:,-1] - E_chain[:,0]
    R_chain = interval[6]['credible']
    dR_chain = R_chain[:,-1] - R_chain[:,0]
    RD_chain = interval[-1]['credible']
    dRD_chain = RD_chain[:,-1] - RD_chain[:,0]
    TotInfected = S_chain[:, 0]-S_chain[:, -1]
    InfectedDet = (TotInfected - dR_chain - U_chain[:,-1] + dRD_chain)/TotInfected
    CFR = dE_chain / (dRD_chain + dE_chain)
    IFR = dE_chain / (dR_chain + dE_chain)
     
    stats[0] = (TotInfected.mean(),TotInfected.std())
    stats[1] = (InfectedDet.mean(),InfectedDet.std())
    stats[2] = (CFR.mean(),CFR.std())
    stats[3] = (IFR.mean(),IFR.std())
    stats_df = pd.DataFrame(stats,columns=["Mean","Std_dev"])    
    stats_df['Statistics'] = ['Infected','Infected detected','CFR','IFR']
    stats_df.to_csv(ResPath+'/Infected_statistics.csv',index=None)

    peaks = dict([])
    peaks['Data_calibration'] = day_end - datetime.timedelta(days=1)
    peaks['Extrapolation_type'] = 'linear'
    peaks['Isolated_time'] = (day_init+datetime.timedelta(days=int(meanI.argmax()))).isoformat()
    peaks['Hospitalized_time'] = (day_init+datetime.timedelta(days=int(meanH.argmax()))).isoformat()
    peaks['Threatened_time'] = (day_init+datetime.timedelta(days=int(meanT.argmax()))).isoformat()
    peaks['Isolated_lower'] = ciI[0].max()
    peaks['Isolated'] = meanI.max()
    peaks['Isolated_upper'] = ciI[1].max()
    peaks['Hospitalized_lower'] = ciH[0].max()
    peaks['Hospitalized'] = meanH.max()
    peaks['Hospitalized_upper'] = ciH[1].max()
    peaks['Threatened_lower'] = ciT[0].max()
    peaks['Threatened'] = meanT.max()
    peaks['Threatened_upper'] = ciT[1].max()
    


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
    #df.to_csv('forecast_hub/'+day_end.strftime('%Y-%m-%d')+'-epiMOX-SUIHTER.csv',index=None)

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
    
    datas = []
    pdatas = []
    intervals = []
    peaks_df = pd.DataFrame()
    data,pdata,interval,peaks = MCMC_postprocess(ResultsFilePath, nsample, burnin)
    intervals.append(interval)
    datas.append(data)
    pdatas.append(pdata)
    peaks_df = peaks_df.append(peaks,ignore_index=True)
    peaks_df.to_csv(ResultsFilePath + '/peaks_forecast_MCMC.csv',index=None)
    
    eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')

    eData['data'] = [pd.to_datetime(x[:10]) for x in eData.data]
    eData.set_index('data', inplace=True)
    
    if not os.path.exists(ResultsFilePath + '/img/'):
        os.mkdir(ResultsFilePath + '/img/')

    ## Modify here to change the starting date ##

    day_init = pd.to_datetime('2021-01-10')# datetime.date(day=20,month=2,year=2021) 
    day_start = pd.to_datetime('2021-01-10')#datetime.date(day=20,month=2,year=2021) 
    ndays = (day_start - day_init).days

    eData = eData[day_init + pd.Timedelta(int(datas[0].xdata[0][-1]),'days'):]
    #############################################

    int_colors = [(1.0, 0.6, 0.6), (1.0, 0.6, 0.6), (0.6, 0.6, 1.0), (0.6, 1.0, 0.6), (1.0, 1.0, 0.6)]
    model_colors = [(1.0, 0, 0.0), (1.0, 0, 0.0), (0, 0.0, 1.0), (0.0, 1.0, 0), (1.0, 1.0, 0.0)]
    int_colors = [ (1.0, 0.6, 0.6), (0.6, 0.6, 1.0), (0.6, 1.0, 0.6), (1.0, 0.7, 0.4), (0.4, 1.0, 0.7), (0.7, 0.4, 1.0)]
    model_colors =[ (1.0, 0, 0.0), (0, 0.0, 1.0), (0.0, 1.0, 0), (0.8, 0.4, 0.0), (0.0, 0.8, 0.4), (0.4, 0.0, 0.8)]

    linetype=['--','-','-','-','-']
    linetype=['-','-','-','-','-','-']

    ## Modify here to change labels ##

    #scenari = [r'With $\delta$ variant']
    #scenari = ['Green pass imposed at 100%']
    scenari = ['Model with vaccines']
    #intervals[0][5]['credible'] = np.diff(intervals[0][5]['credible'],axis=1)

    ##################################
    titoli = ['Undetected','Isolated','Hospitalized','Threatened','Extinct','New positives']
    f = plt.figure(1)
    ax = [plt.subplot(231+x) for x in range(len('UIHTEN'))]
    for i, c in enumerate('UIHTEN'):
        if i==8:
            plt.sca(ax[6])
            f, ax[6] = plot_intervals(intervals[0][i+2], pd.date_range(day_init + pd.Timedelta(int(datas[0].xdata[0][-1]),'days'), day_init + pd.Timedelta(int(pdatas[0].xdata[0][-1]),'days')).values , fig = f,ciset = dict(colors = [int_colors[0]]),
                                             interval_display=dict(alpha=0.3,label='_nolabel'), model_display=dict(color = model_colors[0],label=scenari[0]+' (from detected)',linestyle='--'),
                                             addcredible=True, addprediction=False, addmodel=True, adddata=False, limits=[95])
        elif i > 0:
            if i==5:
                j=i+4
            else:
                j=i+1
            if i==4:
                plt.sca(ax[i])
                f, ax[i] = plot_intervals(intervals[0][j],  pd.date_range(day_init + pd.Timedelta(int(datas[0].xdata[0][-1]),'days'), day_init + pd.Timedelta(int(pdatas[0].xdata[0][-1]),'days')).values,xdata=pd.date_range(day_start, day_init + pd.Timedelta(int(datas[0].xdata[0][-1]),'days')).values, ydata=datas[0].ydata[0][i-1,ndays:],
                                   fig=f, ciset=dict(colors=[int_colors[0]]), interval_display=dict(alpha=0.3,label='_nolabel'),
                                   model_display=dict(color=model_colors[0], label=scenari[0]+' (all)' if i==6 else scenari[0],linestyle=linetype[0]), data_display=dict(color='k',markersize=10),
                                   addcredible=True, addprediction=False, addmodel=True, adddata=True, limits=[95])
            elif i==5:
                plt.sca(ax[i])
                f, ax[i] = plot_intervals(intervals[0][j],  pd.date_range(day_init + pd.Timedelta(int(datas[0].xdata[0][-1]),'days'), day_init + pd.Timedelta(int(pdatas[0].xdata[0][-1]),'days')).values, xdata=pd.date_range(day_start, day_init + pd.Timedelta(int(datas[0].xdata[0][-1]),'days')).values, ydata=datas[0].ydata[0][i,ndays:],
                                   fig=f, ciset=dict(colors=[int_colors[0]]), interval_display=dict(alpha=0.3,label='_nolabel'),
                                   model_display=dict(color=model_colors[0], label=scenari[0]+' (all)' if i==6 else scenari[0],linestyle=linetype[0]), data_display=dict(color='k',markersize=10),
                                   addcredible=True, addprediction=False, addmodel=True, adddata=True, limits=[95])
            else:
                plt.sca(ax[i])
                f, ax[i] = plot_intervals(intervals[0][j],  pd.date_range(day_init + pd.Timedelta(int(datas[0].xdata[0][-1]),'days'), day_init + pd.Timedelta(int(pdatas[0].xdata[0][-1]),'days')).values, xdata=pd.date_range(day_start, day_init + pd.Timedelta(int(datas[0].xdata[0][-1]),'days')).values, ydata=datas[0].ydata[0][i-1,ndays:],
                                   fig=f, ciset=dict(colors=[int_colors[0]]), interval_display=dict(alpha=0.3,label='_nolabel'),
                                   model_display=dict(color=model_colors[0], label=scenari[0]+' (all)' if i==6 else scenari[0],linestyle=linetype[0]), data_display=dict(color='k',markersize=10),
                                   addcredible=True, addprediction=False, addmodel=True, adddata=True, limits=[95])
        else:
            plt.sca(ax[i])
            f, ax[i] = plot_intervals(intervals[0][1], pd.date_range(day_init + pd.Timedelta(int(datas[0].xdata[0][-1]),'days'), day_init + pd.Timedelta(int(pdatas[0].xdata[0][-1]),'days')).values , fig = f,ciset = dict(colors = [int_colors[0]]),
                                             interval_display=dict(alpha=0.3,label='_nolabel'), model_display=dict(color = model_colors[0],label=scenari[0],linestyle=linetype[0]),
                                             addcredible=True, addprediction=False, addmodel=True, adddata=False, limits=[95])
        plt.grid(alpha=0.3)

    ax[1].plot(eData.isolamento_domiciliare, '--k')
    ax[2].plot(eData.ricoverati_con_sintomi, '--k')
    ax[3].plot(eData.terapia_intensiva, '--k')
    ax[4].plot(eData.deceduti, '--k')
    #ax[5].plot(eData.dimessi_guariti, '--k')
    ax[5].plot(eData.nuovi_positivi, '--k')
    for i,x in enumerate(ax):
        x.set_title(titoli[i],fontsize = 16)
        x.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
        x.tick_params(axis='both', which='major', labelsize='large')
        #x.set_xlim(left=datetime.date(day=20,month=2,year=2021))
        handles, labels = x.get_legend_handles_labels()
        x.legend(handles, labels, loc="upper left", fontsize = 'large')
    fig = plt.gcf()
    fig.set_size_inches((19.2,16.2), forward=False)

    fig.savefig(ResultsFilePath + '/img/forecast_dicembre.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/forecast_dicembre.ptfig', 'wb'))
