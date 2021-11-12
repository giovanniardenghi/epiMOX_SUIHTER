#import include
import numpy as np
import pandas as pd
#import geopandas as gpd
import os.path
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from epi import estimation as es
from datetime import datetime
from datetime import date
from datetime import timedelta

font = {'family' : 'normal',
        'size'   : 12}

mpl.rc('font', **font)

if len(sys.argv)<4:
    sys.exit('Error: at least 4 arguments required: ResPath, Ndata, startData. Exit.')
ResPath = sys.argv[1]
Ndata = int((date.fromisoformat(sys.argv[2])-date.fromisoformat('2020-02-24')).days)
startData = int((date.fromisoformat(sys.argv[3])-date.fromisoformat('2020-02-24')).days)
#ncases = int(sys.argv[4])

if len(sys.argv)>5:
    region = sys.argv[5]
    #.capitalize()
    data = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    data['denominazione_regione'] = [x.title() for x in data['denominazione_regione']]
    data = data[data['denominazione_regione']==region]
else:
    region = 'Italia'
    data = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')

# Load the data for the output
#data = pd.read_csv(dataFile)
data['data'] = pd.to_datetime(data.data) 
data['time'] = np.arange(0,len(data))
data = data[data["time"]>=startData]
#data = data[data["time"]<=Ndata]
data.loc[:,"time"] = data["time"]-startData

out_path = ResPath
print('Save output in ' +out_path)

col = ['b','sienna','orange','green','magenta','#8f00ff','b']
col = ['green', 'magenta','#8f00ff','b']

#cases = [region+'_coprifuoco',region]
#cases = [region+'_attuale',region+'_apertura',region+'_coprifuoco',region+'_coprifuoco2',region+'_coprifuoco3',region]
#cases = [region,region+'_coprifuoco3',region+'_coprifuoco17',region+'_coprifuoco24',region+'_coprifuoco31']
cases = ['bianca','bianca_no_gp']
label = ['Scenario 1','Scenario 2', 'Scenario 3',  'Scenario 4', 'Scenario 5']
#label = ['','Scenario 4','Scenario 5']

every = 1
#for i in range(ncases):
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
fig2,ax5 = plt.subplots()
for i,c in enumerate(cases):
    #simdf = pd.read_hdf(ResPath + f'/{c}/simdf.h5')
    #dataRed = data[data["time"]<=Ndata-startData].copy()

    #ax1.plot(data['data'],data.isolamento_domiciliare.values+data.ricoverati_con_sintomi.values+data.terapia_intensiva.values, f'{col[0]}-',   alpha=0.3)
    #ax1.plot(dataRed['data'][::every],dataRed.isolamento_domiciliare.values+dataRed.ricoverati_con_sintomi.values+dataRed.terapia_intensiva.values, f'{col[0]}+',linewidth=2)
    #ax2.plot(data['data'], data.ricoverati_con_sintomi.values, f'{col[0]}-', alpha=0.3)
    #ax2.plot(dataRed['data'][::every], dataRed.ricoverati_con_sintomi.values[::every], f'{col[0]}+',linewidth=2)
    #ax3.plot(data['data'],data.terapia_intensiva.values,f'{col[0]}-',  alpha=0.3)
    #ax3.plot(dataRed['data'][::every],dataRed.terapia_intensiva.values[::every],f'{col[0]}+',linewidth=2)
    #ax4.plot(data['data'],data.deceduti.values, f'{col[0]}-',alpha=0.3)
    #ax4.plot(dataRed['data'][::every],dataRed.deceduti.values[::every], f'{col[0]}+',linewidth=2)
    simdf = pd.read_hdf(ResPath + f'/{c}/simdf.h5')
    #simdf = simdf.loc[114:,:]
    totU = simdf.groupby(['time']).sum()["Undetected"]
    totI = simdf.groupby(['time']).sum()["Isolated"]
    totNP = simdf.groupby(['time']).sum()["New_positives"]
    totH = simdf.groupby(['time']).sum()["Hospitalized"]
    totT = simdf.groupby(['time']).sum()["Threatened"]
    totE = simdf.groupby(['time']).sum()["Extinct"]
    totdE = simdf.groupby(['time']).sum()["Extinct"].diff()
    #totR = es.postProcessH(totI,totH,totT,np.array(1))
    totR = simdf.groupby(['time']).sum()["Recovered"]
    
    gamma = 1/9
    R_star = pd.Series(np.diff(np.log(totI+totH+totT))).rolling(window=7).mean().values/gamma + 1
    R_star_data = pd.Series(np.diff(np.log(data.totale_positivi))).rolling(window=7).mean().values/gamma + 1
    
    if i==0:
        ax1.plot(data.data,data.totale_positivi, 'k+-', label='Dati',linewidth=2)
        ax2.plot(data.data,data.ricoverati_con_sintomi, 'k+-', label='Dati',linewidth=2)
        ax3.plot(data.data,data.terapia_intensiva, 'k+-', label='Dati',linewidth=2)
        ax4.plot(data.data,data.deceduti.diff(), 'k+-', label='Dati',linewidth=2)
        #ax5.plot(data.data[1:],R_star_data,'k+-', label='Dati')

    ax5.plot(simdf.date[1:],R_star,col[i],label=label[i])
    ax5.legend()
    ax5.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax5.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    ax5.set_xlim(left='2021-09-06',right='2021-12-31')

    ax1.plot(simdf.date, totI+totH+totT, col[i], label=label[i],linewidth=2)
    #ax1.plot(simdf.date, totNP, col[i], label=label[i],linewidth=2)
    ax2.plot(simdf.date, totH, col[i], label=label[i],linewidth=2)
    ax3.plot(simdf.date, totT, col[i], label=label[i],linewidth=2)
    ax4.plot(simdf.date, totdE, col[i], label=label[i],linewidth=2)

    ax1.set_title('Positivi')
    ax1.legend()
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    #ax1.set_xlim(left='2021-05-10',right='2021-06-15')
    #ax1.set_ylim([180000,400000])
    
    ax2.set_title('Ospedalizzati')
    ax2.legend()
    #ax2.legend(loc = "upper right")
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax2.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    #ax2.set_xlim(left='2021-05-10',right='2021-06-15')
    #ax2.set_ylim([7000,16000])

    ax3.set_title('Terapie intensive')
    ax3.legend()
    ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax3.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    #ax3.set_xlim(left='2021-05-10',right='2021-06-15')
    #ax3.set_ylim([1000,2200])
    
    ax4.set_title('Deceduti Giornalieri')
    ax4.legend()
    ax4.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax4.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    #ax4.set_xlim(left='2021-05-10',right='2021-06-15')
    #ax4.set_ylim([122000,129000]) #Italia
    #ax4.set_ylim([31000,36000]) #Lombardia
    #ax4.set_ylim([6000,8500])  #Lazio
 
    #fig = plt.gcf()
    fig.set_size_inches((19.2, 10.8), forward=False)
    plt.hlines(1,'2021-09-06','2021-12-31',colors='grey',linestyles='dashed')
    fig2.set_size_inches((19.2, 10.8), forward=False)
    fig.savefig(ResPath + f'/scenari_scuola.png', dpi=300)
    fig2.savefig(ResPath + f'/scenari_scuola_Rt.png', dpi=300)
    #fig.savefig(ResPath + f'/{region}_giugno.png', dpi=300)
    
    #plt.savefig(ResPath + f'/case{i}/case{i}_IHTE_comp.png')
plt.show() 
plt.close()

