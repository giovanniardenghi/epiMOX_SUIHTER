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
date = dataRed['data'].reset_index(drop=True)
for i in range(idx,int(simdf['time'].max())+1):
    date=date.append(pd.Series(date[i-1]+dt,index = [i]))
simdf['data']=date

out_path = '/'.join(fileName.split('/')[:-1])
print('Save output in ' +out_path)

fig,ax = plt.subplots(figsize=(12, 8))

totU = simdf.groupby(['time']).sum()["Undetected"]
totDeltaU = simdf.groupby(['time']).sum()["New_positives"]
totS = simdf.groupby(['time']).sum()["Suscept"]
totI = simdf.groupby(['time']).sum()["Isolated"]
totH = simdf.groupby(['time']).sum()["Hospitalized"]
totT = simdf.groupby(['time']).sum()["Threatened"]
totE = simdf.groupby(['time']).sum()["Extinct"]
totR = simdf.groupby(['time']).sum()["Recovered"]
totRD = simdf.groupby(['time']).sum()["Recovered_detected"]
logI = pd.Series(np.log(totI))
gamma = 1/9
Rt = logI.diff(periods=7) / 7 / gamma + 1

every = 1
data = dataRed
ax1 = plt.subplot(231)

ax1.plot(data['data'],data.isolamento_domiciliare.values,'y-',   alpha=0.3)
ax1.plot(dataRed['data'][::every],dataRed.isolamento_domiciliare.values[::every],'y+',  label='Isolated',linewidth=2)
ax1.plot(date,totI,'y',  label='Isolated Simulated',linewidth=2)
ax1.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax1.legend()

ax2 = plt.subplot(232)

ax2.plot(data['data'],data.ricoverati_con_sintomi.values,'b-',  alpha=0.3)
ax2.plot(dataRed['data'][::every],dataRed.ricoverati_con_sintomi.values[::every],'b+',  label='Hospitalized',linewidth=2)
ax2.plot(date,totH,'b',  label='Hospitalized Simulated',linewidth=2)
ax2.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax2.legend()

ax3 = plt.subplot(233)

ax3.plot(data['data'],data.terapia_intensiva.values,'r-',  alpha=0.3)
ax3.plot(dataRed['data'][::every],dataRed.terapia_intensiva.values[::every],'r+',  label='Threatened',linewidth=2)
ax3.plot(date,totT,'r',  label='Threatened Simulated',linewidth=2)
ax3.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax3.legend()

ax4 = plt.subplot(234)

ax4.plot(data['data'],data.deceduti.values,'k-',     alpha=0.3)
ax4.plot(dataRed['data'][::every],dataRed.deceduti.values[::every],'k+',  label='Extinct',linewidth=2)
ax4.plot(date,totE,'k',  label='Extinct Simulated',linewidth=2)
ax4.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax4.legend()

ax5 = plt.subplot(235)

ax5.plot(data['data'],data.nuovi_positivi.values,'k-',     alpha=0.3)
ax5.plot(dataRed['data'][::every],dataRed.nuovi_positivi.values[::every],'k+',  label='New positives',linewidth=2)
ax5.plot(date,totDeltaU,'k',  label='New positives Simulated',linewidth=2)
ax5.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax5.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax5.legend()

#ax5.plot(date,Rt,'k',  label='R*',linewidth=2)
#ax5.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
#ax5.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
#ax5.legend()

ax6 = plt.subplot(236)

ax6.plot(data['data'],data.dimessi_guariti.values,'g-',     alpha=0.3)
ax6.plot(dataRed['data'][::every],dataRed.dimessi_guariti.values[::every],'g+',  label='Identified Recovered',linewidth=2)
ax6.plot(date,totRD,'g',  label='Identified Recovered Simulated',linewidth=2)
ax6.plot(date,totR,'g--',  label='Recovered Simulated',linewidth=2)
ax6.xaxis.set_major_locator(mpl.dates.DayLocator(interval=15))
ax6.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
ax6.legend()

plt.savefig(out_path+'/SUIHTER-' + region + '_' + str(Ndata - pd.Timedelta(1,'day') ).split()[0]  + '.png')
plt.show()
