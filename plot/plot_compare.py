#import include
import numpy as np
import pandas as pd
import os.path
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from epi import estimation as es
import datetime
from datetime import timedelta
from epiMOX import correct_isolated_model

if len(sys.argv) < 3:
    sys.exit('Expected at least 2 arguments: results file , days. Exit.')

fileNames = ['beta2','beta2_sigma1','beta2_sigma1_R']
fileNames = [x+'_luglio' for x in fileNames]
fileNames = ['beta2','test_variant','test_variant_R']
fileNames = ['2021-07-12','yellow_15_days','yellow_tomorrow']
#fileNames = ['2021-07-12','yellow_orange_auto']
colors = ['g','b','m','r'] 
labels = [r'$\beta_U$ doubled from August, 1st',r'$\beta_U$ doubled from August, 1st and $\sigma_1=\sigma_2=1$',r'$\beta_U$ doubled from August, 1st and $\sigma_1=\sigma_2=1$ and recoverd infected']
labels = ['Scenario 1','Scenario 2','Scenario 3']#,'Scenario 4','Scenario 5']
#labels = [r'$\kappa = 0.4$',r'$\kappa = 0.3$',r'$\kappa = 0.1$',r'$\kappa = 0$',]

fileNames = ['Tests/'+x+'/simdf.h5' for x in fileNames]
Ndata = pd.to_datetime(sys.argv[1]) + pd.Timedelta(1,'day')
if len(sys.argv)>2:
    startData = pd.to_datetime(sys.argv[2])
else:
    startData = 0

region = 'Italia'
if len(sys.argv)>3:
    region = sys.argv[3]
    print(region)
if region != 'Italia':
    data = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    data['denominazione_regione'] = [x.title() for x in data['denominazione_regione']]
    data = data[data['denominazione_regione']==region]
else:
    data = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')

    data['data'] = [x[:10] for x in data.data]
    data['data'] = pd.to_datetime(data.data)

    corrections = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/corrections.csv',keep_default_na=False,na_values=['NaN'])
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
        data = correct_isolated_model(data.set_index('data'), comp_to_old[row['comp']].replace(' ', '_').lower(),
                                      pd.to_datetime(row['date']), int(row['DT']), int(row['C']),
                                      comp_to_old[row['Positivi']].replace(' ', '_').lower(),
                                      comp_to_old[row['Guariti']].replace(' ', '_').lower()).reset_index().rename(
            columns={'index': 'data'})
    data['data'] = data.data.dt.strftime('%Y-%m-%dT%H:%M:%S')

by_age = False
if len(sys.argv)>4:
    age = sys.argv[4]
    by_age = True


data['data'] = pd.to_datetime(data['data'])
UD = es.estimateUndetected(data.isolamento_domiciliare.values,data.ricoverati_con_sintomi.values,data.terapia_intensiva.values,data.dimessi_guariti.values,data.deceduti.values)
UD = UD[(startData-pd.to_datetime('2020-02-24')).days:]
data = data[data["data"]>=startData]
data['time'] = np.arange(0,len(data))

dataRed = data[data["data"]<=Ndata].copy().reset_index(drop=True)
if by_age and age!='all':
    perc = pd.read_csv('util/stato_clinico.csv')
    perc = perc[perc['Età']==age].reset_index(drop=True) 
    perc.rename(columns=dict(zip(perc.columns[2:],['deceduti','ricoverati_con_sintomi','isolamento_domiciliare','dimessi_guariti','terapia_intensiva'])),inplace=True)
    dataRed[perc.columns[2:]] = dataRed[perc.columns[2:]].mul(perc[perc.columns[2:]])

idx = len(dataRed['data'].values)-1
dt = timedelta(days=1)
# Load the data for the output
out_path ='Tests/' 
print('Save output in ' +out_path)

ax=['dummy']
#fig,ax = plt.subplots(figsize=(12, 8))
for i in range(1,11):
    plt.figure(i)
    ax.append(plt.gca())

for k,fileName in enumerate(fileNames):
    simdf = pd.read_hdf(fileName)
    sites = pd.read_csv('util/Regioni_Italia_sites.csv')
    if by_age:
        sites = pd.read_csv('util/Eta_Italia_sites.csv')
    sites = sites[sites['Name']==region]
    if by_age:
        if age!='all':
            sites = sites[sites['Code']==age]
            simdf = simdf[simdf['Geocode']==age]
    else:
        simdf = simdf[simdf['Geocode']==sites['Code'].values[0]]
    
    data = data[data['data']>='2021-06-01']
    dataRed = dataRed[dataRed['data']>='2021-06-01']
    simdf = simdf[simdf['date']>='2021-06-01']
    UD = UD[-len(data):]
    
    totU = simdf.groupby(['time']).sum()["Undetected"]
    totDeltaU = simdf.groupby(['time']).sum()["New_positives"]
    totS = simdf.groupby(['time']).sum()["Suscept"]
    totI = simdf.groupby(['time']).sum()["Isolated"]
    totH = simdf.groupby(['time']).sum()["Hospitalized"]
    totT = simdf.groupby(['time']).sum()["Threatened"]
    totE = simdf.groupby(['time']).sum()["Extinct"].diff()
    totR = simdf.groupby(['time']).sum()["Recovered"]
    NewT = simdf.groupby(['time']).sum()["New_threatened"] 
    totR2 = es.estimateRecovered(dataRed.deceduti.values)
    totRD = simdf.groupby(['time']).sum()["Recovered_detected"]
    totV1 = simdf.groupby(['time']).sum()["First_dose"]
    totV2 = simdf.groupby(['time']).sum()["Second_dose"]
    logI = pd.Series(np.log(totI))
    gamma = 1/9
    Rt = logI.diff(periods=7) / 7 / gamma + 1

    every = 1
    #data = dataRed
    #ax1 = plt.subplot(241)

    if k==0:
        ax[1].plot(data['data'],data.isolamento_domiciliare.values,'k-',   alpha=0.3)
        ax[1].plot(dataRed['data'][::every],dataRed.isolamento_domiciliare.values[::every],'k+',  label='Data',markersize=14)
    ax[1].plot(simdf.date,totI, colors[k],  label=labels[k],linewidth=4)
    #ax[1].xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
    ax[1].xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax[1].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    ax[1].tick_params(axis='both', which='major', labelsize=20)

    ax[1].legend(fontsize=20)
    ax[1].set_title('Isolated',fontsize=26)

    #ax2 = plt.subplot(242)
    if k==0:
        ax[2].plot(data['data'],data.ricoverati_con_sintomi.values,'k-',  alpha=0.3)
        ax[2].plot(dataRed['data'][::every],dataRed.ricoverati_con_sintomi.values[::every],'k+',  label='Data', markersize=14)
    ax[2].plot(simdf.date,totH, colors[k],  label=labels[k],linewidth=4)
    #ax[2].xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
    ax[2].xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax[2].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    ax[2].tick_params(axis='both', which='major', labelsize=20)
    ax[2].legend(fontsize=20)
    ax[2].set_title('Hospitalized',fontsize=26)

    #ax3 = plt.subplot(243)
    if k==0:
        ax[3].plot(data['data'],data.terapia_intensiva.values,'k-',  alpha=0.3)
        ax[3].plot(dataRed['data'][::every],dataRed.terapia_intensiva.values[::every],'k+',  label='Data', markersize=14)
    ax[3].plot(simdf.date,totT, colors[k],  label=labels[k],linewidth=4)
    #ax[3].xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
    ax[3].xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax[3].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    ax[3].tick_params(axis='both', which='major', labelsize=20)
    ax[3].legend(fontsize=20)
    ax[3].set_title('Threatened',fontsize=26)

    #ax4 = plt.subplot(244)

    if k==0:
        ax[4].plot(data['data'],data.deceduti.diff().rolling(window=7,min_periods=1,center=True).mean().values,'k-',     alpha=0.3)
        ax[4].plot(dataRed['data'][::every],dataRed.deceduti.diff().values[::every],'k+',  label='Data', markersize=14)
    ax[4].plot(simdf.date,totE, colors[k],  label=labels[k],linewidth=4)
    #ax[4].xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
    ax[4].xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax[4].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    ax[4].tick_params(axis='both', which='major', labelsize=20)
    ax[4].legend(fontsize=20)
    ax[4].set_title('Daily Extinct',fontsize=26)

    #ax5 = plt.subplot(245)

    if k==0:
        ax[5].plot(data['data'],data.nuovi_positivi.values,'k-',     alpha=0.3)
        ax[5].plot(dataRed['data'][::every],dataRed.nuovi_positivi.values[::every],'k+',  label='Data',markersize=14)
    ax[5].plot(simdf.date,totDeltaU, colors[k],  label=labels[k], linewidth=4)
    #ax[5].hlines(np.array([50, 150, 250]) / 1e5 * sites.Pop.values[0] / 7, data.data.values[0], simdf.date.values[-1], colors=['y','orange','red'], linestyles='dashed', linewidth=4)

    #ax[5].xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
    ax[5].xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax[5].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    ax[5].tick_params(axis='both', which='major', labelsize=20)
    ax[5].legend(fontsize=20)
    ax[5].set_title('New positives',fontsize=26)

    #a[x]5.plot(simdf.date,Rt,'k',  label='R*',linewidth=2)
    #a[x]5.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
    #a[x]5.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    #a[x]5.legend()

    #a[x]6 = plt.subplot(246)
    if k==0:
        ax[6].plot(data['data'],data.dimessi_guariti.values,'k-',     alpha=0.3)
        ax[6].plot(dataRed['data'][::every],dataRed.dimessi_guariti.values[::every],'k+',  label='Data Identified', markersize=14)
        ax[6].plot(dataRed['data'][::every],totR2,'k+',  label='Data', markersize=14)
    ax[6].plot(simdf.date,totRD, colors[k],  label=labels[k]+' Identified',linewidth=4)
    ax[6].plot(simdf.date,totR, colors[k]+'--',  label=labels[k], linewidth=4)
    #ax[6].xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
    ax[6].xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax[6].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    ax[6].tick_params(axis='both', which='major', labelsize=20)
    ax[6].legend(fontsize=20)
    ax[6].set_title('Recovered',fontsize=26)

    #a[x]7 = plt.subplot(247)
    if k==0:
        ax[7].plot(data['data'],UD,'k+', label='Data', markersize=14)
    ax[7].plot(simdf.date,totU, colors[k], label=labels[k], linewidth=4)
    #ax[6].xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
    ax[7].xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax[7].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    ax[7].tick_params(axis='both', which='major', labelsize=20)
    ax[7].legend(fontsize=20)
    ax[7].set_title('Undetected',fontsize=26)

    
    if k==0:
        #vaccines = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/vaccini_regioni/'+region+'.csv')
        vaccines = pd.read_csv('~/dpc-covid-data/data/vaccini_regioni/'+region+'.csv')
        vaccines['data'] = pd.to_datetime(vaccines.data)
        vaccines.set_index('data',inplace=True)
        vaccines = vaccines.cumsum()
        vaccines = vaccines.loc[startData:simdf.date.values[-1]]
        vaccines = vaccines.reindex(pd.date_range(vaccines.index[0],simdf['date'].iloc[-1])).ffill()

        ax[10].plot(vaccines['prima_dose'],'k-',  label='First dose',linewidth=4)
        ax[10].plot(vaccines['seconda_dose'],'r-',  label='Second dose',linewidth=4)
        ax[10].xaxis.set_major_locator(mpl.dates.MonthLocator())
        ax[10].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
        ax[10].tick_params(axis='both', which='major', labelsize=20)
        ax[10].legend(fontsize=20)
        ax[10].set_title('Vaccines',fontsize=26)
    
    if k==0:
        ax[8].plot(data['data'],data.ingressi_terapia_intensiva,'k+',   label='Data', markersize=14)
    ax[8].plot(simdf.date,NewT, colors[k],  label=labels[k], linewidth=4)
    #ax[8].xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
    ax[8].xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax[8].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    ax[8].tick_params(axis='both', which='major', labelsize=20)
    ax[8].legend(fontsize=20)
    ax[8].set_title('New threatened',fontsize=26)


    ax[9].plot(simdf.date,Rt, colors[k],  label=labels[k], linewidth=4)
    ax[9].hlines(1,simdf.date.values[0],simdf.date.values[-1],linestyles='dashed',color='gray',linewidth=3)
    ax[9].xaxis.set_major_locator(mpl.dates.MonthLocator())
    ax[9].xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
    ax[9].tick_params(axis='both', which='major', labelsize=20)
    ax[9].legend(fontsize=20)
    ax[9].set_title('R*',fontsize=26)

comps = ['Isolated', 'Hospitalized', 'Threatened', 'Extinct', 'New_positives', 'Recovered', 'Undetected', 'New_threatened', 'Rt', 'Vaccines']

for i in range(1,11):
    plt.figure(i)
    fig = plt.gcf()
    fig.set_size_inches((8,8), forward=False)
    fig.tight_layout()
    plt.savefig(out_path+'/SUIHTER-' + region + '_' + str(Ndata - pd.Timedelta(1,'day') ).split()[0] + '_' + comps[i-1] + '.png')
plt.show()

