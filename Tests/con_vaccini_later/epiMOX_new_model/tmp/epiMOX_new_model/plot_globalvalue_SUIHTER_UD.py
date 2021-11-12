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

fileName = sys.argv[1]
Ndata = pd.to_datetime(sys.argv[2]) + pd.Timedelta(1,'day')
if len(sys.argv)>3:
    startData = pd.to_datetime(sys.argv[3])
else:
    startData = 0

region = 'Italia'
if len(sys.argv)>4:
    region = sys.argv[4]
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
    if region != 'Italia':
        corrections = corrections[corrections['region'] == region]

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
if len(sys.argv)>5:
    age = sys.argv[5]
    by_age = True

# Load the data for the output
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

idx = len(dataRed['data'].values)
dt = timedelta(days=1)

out_path = '/'.join(fileName.split('/')[:-1])
print('Save output in ' +out_path)

fig,ax = plt.subplots(figsize=(12, 8))

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
ax1 = plt.subplot(241)

ax1.plot(data['data'],data.isolamento_domiciliare.values,'y-',   alpha=0.3)
ax1.plot(dataRed['data'][::every],dataRed.isolamento_domiciliare.values[::every],'y+',  label='Isolated',linewidth=2)
ax1.plot(simdf.date,totI,'y',  label='Isolated Simulated',linewidth=2)
ax1.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax1.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
ax1.legend()

ax2 = plt.subplot(242)

ax2.plot(data['data'],data.ricoverati_con_sintomi.values,'b-',  alpha=0.3)
ax2.plot(dataRed['data'][::every],dataRed.ricoverati_con_sintomi.values[::every],'b+',  label='Hospitalized',linewidth=2)
ax2.plot(simdf.date,totH,'b',  label='Hospitalized Simulated',linewidth=2)
ax2.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax2.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
ax2.legend()

ax3 = plt.subplot(243)

ax3.plot(data['data'],data.terapia_intensiva.values,'r-',  alpha=0.3)
ax3.plot(dataRed['data'][::every],dataRed.terapia_intensiva.values[::every],'r+',  label='Threatened',linewidth=2)
ax3.plot(simdf.date,totT,'r',  label='Threatened Simulated',linewidth=2)
ax3.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax3.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
ax3.legend()

ax4 = plt.subplot(244)

ax4.plot(data['data'],data.deceduti.diff().rolling(window=7,min_periods=1,center=True).mean().values,'k-',     alpha=0.3)
ax4.plot(dataRed['data'][::every],dataRed.deceduti.diff().values[::every],'k+',  label='Extinct',linewidth=2)
ax4.plot(simdf.date,totE,'k',  label='Extinct Simulated',linewidth=2)
ax4.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax4.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
ax4.legend()

ax5 = plt.subplot(245)

ax5.plot(data['data'],data.nuovi_positivi.values,'k-',     alpha=0.3)
ax5.plot(dataRed['data'][::every],dataRed.nuovi_positivi.values[::every],'k+',  label='New positives',linewidth=2)
ax5.plot(simdf.date,totDeltaU,'k',  label='New positives Simulated',linewidth=2)
ax5.hlines(50 / 1e5 * sites.Pop / 7, ax5.get_xlim()[0], ax5.get_xlim()[1], colors='y', linestyles='dashed', linewidth=2)
ax5.hlines(150 / 1e5 * sites.Pop / 7, ax5.get_xlim()[0], ax5.get_xlim()[1], colors='orange', linestyles='dashed', linewidth=2)
ax5.hlines(250 / 1e5 * sites.Pop / 7, ax5.get_xlim()[0], ax5.get_xlim()[1], colors='red', linestyles='dashed', linewidth=2)
ax5.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax5.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
ax5.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax5.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
ax5.legend()

#ax5.plot(simdf.date,Rt,'k',  label='R*',linewidth=2)
#ax5.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
#ax5.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
#ax5.legend()

#ax6 = plt.subplot(246)

#ax6.plot(data['data'],data.dimessi_guariti.values,'g-',     alpha=0.3)
#ax6.plot(dataRed['data'][::every],dataRed.dimessi_guariti.values[::every],'g+',  label='Identified Recovered',linewidth=2)
#ax6.plot(dataRed['data'][::every],totR2,'g+',  label='Recovered',linewidth=2)
#ax6.plot(simdf.date,totRD,'g',  label='Identified Recovered Simulated',linewidth=2)
#ax6.plot(simdf.date,totR,'g--',  label='Recovered Simulated',linewidth=2)
#ax6.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
#ax6.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
#ax6.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
#ax6.legend()

ax7 = plt.subplot(247)
ax7.plot(data['data'],UD,'r+',   label='Undetected',linewidth=2)
ax7.plot(simdf.date,totU,'k',  label='Undetected Simulated',linewidth=2)
ax7.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax7.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
ax7.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax7.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
ax7.legend()

#vaccines = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/vaccini_regioni/'+region+'.csv')
vaccines = pd.read_csv('~/dpc-covid-data/data/vaccini_regioni/'+region+'.csv')
vaccines['data'] = pd.to_datetime(vaccines.data)
vaccines.set_index('data',inplace=True)
vaccines = vaccines.loc[startData:simdf.date.values[-1]]
vaccines = vaccines.reindex(pd.date_range(vaccines.index[0],simdf['date'].iloc[-1])).ffill()

vaccines = vaccines.cumsum()

ax6 = plt.subplot(246)
ax6.plot(simdf.date,totS,'g',  label='Susceptibles',linewidth=2)
ax6.plot(simdf.date,totV1,'k',  label='First dose simulated',linewidth=2)
ax6.plot(np.maximum(vaccines['prima_dose']-vaccines['seconda_dose'],0),'k--',  label='First dose',linewidth=2)
ax6.plot(simdf.date,totV2,'r',  label='Second dose simulated',linewidth=2)
ax6.plot(vaccines['seconda_dose'],'r--',  label='Second dose',linewidth=2)
ax6.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax6.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
ax6.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax6.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
ax6.legend()

ax8 = plt.subplot(248)
ax8.plot(data['data'],data.ingressi_terapia_intensiva,'r+',   label='New threatened',linewidth=2)
ax8.plot(simdf.date,NewT,'k',  label='New Threatened Simulated',linewidth=2)
ax8.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax8.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=2))
ax8.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax8.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
ax8.legend()


plt.savefig(out_path+'/SUIHTER-' + region + '_' + str(Ndata - pd.Timedelta(1,'day') ).split()[0]  + '.png')
plt.show()


plt.plot(simdf.date,Rt,'r',   label='R*',linewidth=2)
ax = plt.gca()
ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
ax.legend()
plt.show()
