import sys
import os.path
import numpy as np
import pandas as pd
import json
import pickle as pl
import datetime
from epi import loaddata as ld
from epi import models as md
from epi import estimation as es
from epi.convert import converter
from epiMOX import epiMOX
from epi.MCMC import model_fun as model_fun
import matplotlib as mpl
import matplotlib.pyplot as plt
from pymcmcstat.MCMC import DataStructure
from pymcmcstat import mcmcplot as mcp
from pymcmcstat.propagation import calculate_intervals, plot_intervals, generate_quantiles
import locale


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
    l_args = (params, Pop, DO, map_to_prov)
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

    Tf_data = pd.to_datetime(DataDic['Tf'])
    Tf = int((Tf_data-day_init).days)

    xmod = np.arange(0, Tf, 1)

    pdata = DataStructure()
    pdata.add_data_set(x=xmod, y=data.ydata[0], user_defined_object=data.user_defined_object[0])
    
    param_mean = np.mean(chain, 0)
    params.params[params.getMask()] = param_mean
    params.estimated=True
    params.save(MCMCpath + '/params_mean.csv')
    
    params.forecast(eDataG['time'].max(),Tf,1)
    if params.nSites == 1:
        R0 = np.zeros((chain.shape[0],params.nPhases))
        for j,parms in enumerate(chain):
            params.params[params.getMask()] = parms
            R0[j] = es.computeR0(params, epi_model)
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
        obs = np.concatenate([isolated,hospitalized,threatened, extinct, recovered,New_positives], axis=0).reshape((6,-1))
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
    return data,pdata,interval[1:-1],peaks
	

if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL,'en_US.UTF-8')
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
    
    ## Modify here to change the cases folders names ##

    cases = ['2021-03-07','2021-03-07_Orange','2021-03-07_Red']

    scenarios = [np.array([[204,0.963]]),np.array([[204,0.918]]),np.array([[204,0.724]])] # If no scenarios set to None
	
    ###################################################

    datas = []
    pdatas = []
    intervals = []
    peaks_df = pd.DataFrame()
    for i,c in enumerate(cases):
        data,pdata,interval,peaks = MCMC_postprocess(ResultsFilePath + '/' + c, nsample, burnin, scenarios[i])
        intervals.append(interval)
        datas.append(data)
        pdatas.append(pdata)
        peaks_df = peaks_df.append(peaks,ignore_index=True)
    peaks_df.to_csv(ResultsFilePath + '/peaks_forecast_MCMC.csv',index=None)
    
    if not os.path.exists(ResultsFilePath + '/img/'):
        os.mkdir(ResultsFilePath + '/img/')

    ## Modify here to change the starting date ##

    day_init = datetime.date(day=20,month=8,year=2020) 

    #############################################

    int_colors = [(1.0, 0.6, 0.6), (1.0, 0.6, 0.6), (0.6, 0.6, 1.0), (0.6, 1.0, 0.6), (1.0, 1.0, 0.6)]
    model_colors = [(1.0, 0, 0.0), (1.0, 0, 0.0), (0, 0.0, 1.0), (0.0, 1.0, 0), (1.0, 1.0, 0.0)]
    int_colors = [ (1.0, 0.6, 0.6), (0.6, 0.6, 1.0), (0.6, 1.0, 0.6), (1.0, 0.7, 0.4), (0.4, 1.0, 0.7), (0.7, 0.4, 1.0)]
    model_colors =[ (1.0, 0, 0.0), (0, 0.0, 1.0), (0.0, 1.0, 0), (0.8, 0.4, 0.0), (0.0, 0.8, 0.4), (0.4, 0.0, 0.8)]
    int_colors = [ (1.0, 1.0, 0.6), (1.0, 0.8, 0.6), (1.0, 0.6, 0.6)]
    model_colors =[(1.0,1.0,0.0), (1.0, 0.6, 0.0), (1.0, 0.0, 0.0)]
    int_colors = [ (0.2, 0.2, 0.2), (1.0, 0.8, 0.6), (1.0, 0.6, 0.6)]
    model_colors =[(0,0,0), (1.0, 0.6, 0.0), (1.0, 0.0, 0.0)]

    linetype=['--','-','-','-','-']
    linetype=['-','-','-','-','-','-']

    ## Modify here to change labels ##

    scenari = ['Calibrated until Nov 23rd','Calibrated Until Nov 18th','Calibrated until Nov 9th'] 
    scenari = ['Restrictions currently in effect','All regions orange from Mar 8','All regions Red from Mar 8',] 
    
    ##################################

    titoli = ['Undetected','Isolated','Hospitalized','Threatened','Extinct','Recovered']
    titoli = ['Isolated','Hospitalized','Threatened','Extinct']

    f = plt.figure(1)
    #ax = [plt.subplot(321+x) for x in range(len('UIHTER'))]
    ax = [plt.subplot(311+x) for x in range(len('IHT'))]
    #for i, c in enumerate('UIHTER'):
    for i, c in enumerate('IHT'):
        for j in range(len(cases)):
            if j == 0:
            #if j == 0 and i in range(1,5):
                plt.sca(ax[i])
                f, ax[i] = plot_intervals(intervals[j][i+1], np.array([day_init + datetime.timedelta(days=int(x)) for x in pdatas[j].xdata[0]]), xdata=np.array([day_init + datetime.timedelta(days=int(x)) for x in datas[j].xdata[0]]), ydata=datas[j].ydata[0][i],
                                       fig=f, ciset=dict(colors=[int_colors[j]]), interval_display=dict(alpha=0.3,label='_nolabel'),
                                       model_display=dict(color=model_colors[j], label=scenari[j],linestyle=linetype[j]), data_display=dict(color='k',markersize=10),
                                       addcredible=True, addprediction=False, addmodel=True, adddata=True, limits=[90])
            else:
                plt.sca(ax[i])
                f, ax[i] = plot_intervals(intervals[j][i+1], np.array([day_init + datetime.timedelta(days=int(x)) for x in pdatas[j].xdata[0]]), fig = f,ciset = dict(colors = [int_colors[j]]),
                                                 interval_display=dict(alpha=0.3,label='_nolabel'), model_display=dict(color = model_colors[j],label=scenari[j],linestyle=linetype[j]),
                                                 addcredible=True, addprediction=False, addmodel=True, adddata=False, limits=[90])
        plt.grid(alpha=0.3);
    for i,x in enumerate(ax):
        x.set_title(titoli[i],fontsize = 16)
        x.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:.0f}'))
        x.xaxis.set_major_locator(mpl.dates.MonthLocator())
        x.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b %Y'))
        x.tick_params(axis='both', which='major', labelsize='large')
        x.set_xlim(left=datetime.date(day=1,month=11,year=2020))
        handles, labels = x.get_legend_handles_labels()
        x.legend(handles, labels, loc="upper left", fontsize = 'large')
    fig = plt.gcf()
    fig.set_size_inches((19.2,19.0), forward=False)
    fig.tight_layout()

    fig.savefig(ResultsFilePath + '/img/forecast_color_regions.png',dpi=300)
    pl.dump(f, open(ResultsFilePath + '/img/forecast_color_regions.ptfig', 'wb'))
