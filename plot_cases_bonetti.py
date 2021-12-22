#import include
import numpy as np
import pandas as pd
#import geopandas as gpd
import os.path
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl

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

xleft = '2021-12-01'
xright= '2022-03-20'

if len(sys.argv)>4:
    region = sys.argv[4]
    #.capitalize()
    data = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')
    data['denominazione_regione'] = [x.title() for x in data['denominazione_regione']]
    data = data[data['denominazione_regione']==region]
else:
    region = 'Italia'
    data = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')

# Load the data for the output
#data = pd.read_csv(dataFile)
data['data'] = pd.to_datetime([x[:10] for x in data.data]) 
data['time'] = np.arange(0,len(data))
data = data[data["time"]>=startData]
data = data[data["time"]<=Ndata]
data.loc[:,"time"] = data["time"]-startData
sites = pd.read_csv('util/Regioni_Italia_sites.csv')
sites = sites[sites['Name']==region]

out_path = ResPath
print('Save output in ' +out_path)

#col = ['b','sienna','orange','green','magenta','#8f00ff','b']
#col = ['sienna','orange','green','magenta','#8f00ff','b']
#col = ['green', 'orange', 'magenta','#8f00ff','b']

#cases = [region+'_coprifuoco',region]
#cases = [region+'_attuale',region+'_apertura',region+'_coprifuoco',region+'_coprifuoco2',region+'_coprifuoco3',region]
#cases = [region,region+'_coprifuoco3',region+'_coprifuoco17',region+'_coprifuoco24',region+'_coprifuoco31']
#cases = ['gp_100','gp_50','no_gp']

#cases = ['factor1.4','factor1.6','factor1.8','factor2.0',]
#label = ['+40%','+60%','+80%','+100%'] 

cases = ['Controllato','Giallo','Arancione','Rosso','Rosso27dic','Rosso7gen']
label = ['Controllato','Giallo dal 22/12','Arancione dal 22/12','Rosso dal 22/12','Rosso dal 27/12','Rosso dal 7/1']

#cases = ['P0.01','P0.05','P0.10','P0.20','P0.50','P0.70','P0.90']
#label = ['1%','5%','10%','20%','50%','70%','90%'] 

col = pl.cm.rainbow(np.linspace(0,1,len(cases)))

#cases = ['vaccini_paper_new','no_vaccini_new']
#cases = ['controlled_attuale','vaccines_300K']#,'vaccines_400K','vaccines_500K','vaccines_600K']#,'controlled']
#label = ['vaccini','no vaccini']
#label = ['','Scenario 4','Scenario 5']

every = 1
#for i in range(ncases):
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(nrows=2,ncols=2)
fig2,ax5 = plt.subplots()
for i,c in enumerate(cases):
    simdf = pd.read_hdf(ResPath + f'/{c}/simdf.h5')
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
   
    

    if i==len(cases)-1:

        maxH = 57705
        maxT = 9044
        #maxH = 6646
        #maxT = 1530

        ax1.plot(data.data,data.nuovi_positivi, 'k+-',linewidth=2)
        #ax1.plot(data.data,data.isolamento_domiciliare, 'k+-', label='Dati',linewidth=2)
        ax1.hlines(50 / 1e5 * sites.Pop / 7,  xleft,xright, colors='y', linestyles='dashed', linewidth=2)
        ax1.hlines(150 / 1e5 * sites.Pop / 7, xleft,xright, colors='orange', linestyles='dashed', linewidth=2)
        ax1.hlines(250 / 1e5 * sites.Pop / 7, xleft,xright,colors='r', linestyles='dashed', linewidth=2)
        ax1.text('2022-03-30', 50 / 1e5 * sites.Pop / 7, '50 ', fontsize=8, va='center', ha='center', backgroundcolor='w')
        ax1.text('2022-03-30', 150 / 1e5 * sites.Pop / 7, '150 ', fontsize=8, va='center', ha='center', backgroundcolor='w')
        ax1.text('2022-03-30', 250 / 1e5 * sites.Pop / 7, '250 ', fontsize=8, va='center', ha='center', backgroundcolor='w')
        ax2.plot(data.data,data.ricoverati_con_sintomi, 'k+-',linewidth=2)
        ax2.hlines(0.15*maxH, xleft,xright, colors='y', linestyles='dashed', linewidth=2)
        ax2.hlines(0.30*maxH, xleft,xright, colors='orange', linestyles='dashed', linewidth=2)
        ax2.hlines(0.40*maxH, xleft,xright, colors='r', linestyles='dashed', linewidth=2)
        ax2.text('2022-03-30', 0.15*maxH, '15%', fontsize=8, va='center', ha='center', backgroundcolor='w')
        ax2.text('2022-03-30', 0.30*maxH, '30% ', fontsize=8, va='center', ha='center', backgroundcolor='w')
        ax2.text('2022-03-30', 0.40*maxH, '40% ', fontsize=8, va='center', ha='center', backgroundcolor='w')
        ax3.plot(data.data,data.terapia_intensiva, 'k+-', linewidth=2)
        ax3.hlines(0.1*maxT, xleft,xright, colors='y', linestyles='dashed', linewidth=2)
        ax3.hlines(0.2*maxT, xleft,xright, colors='orange', linestyles='dashed', linewidth=2)
        ax3.hlines(0.3*maxT, xleft,xright, colors='r', linestyles='dashed', linewidth=2)
        ax3.text('2022-03-30', 0.10*maxT, '10%', fontsize=8, va='center', ha='center', backgroundcolor='w')
        ax3.text('2022-03-30', 0.20*maxT, '20% ', fontsize=8, va='center', ha='center', backgroundcolor='w')
        ax3.text('2022-03-30', 0.30*maxT, '30% ', fontsize=8, va='center', ha='center', backgroundcolor='w')
        ax4.plot(data.data,data.deceduti.diff(), 'k+-', linewidth=2)
        #ax5.plot(data.data[1:],R_star_data,'k+-', label='Dati')

    ax5.plot(simdf.date[1:],R_star,color=col[i], label=label[i])
    ax5.legend(loc='upper left')
    ax5.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax5.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    ax5.set_xlim(left='2021-03-01',right='2022-03-31')

    #ax1.plot(simdf.date, totI+totH+totT, col[i], label=label[i],linewidth=2)
    ax1.plot(simdf.date, totNP, color=col[i], label=label[i],linewidth=2)
    ax2.plot(simdf.date, totH,  color=col[i], label=label[i],linewidth=2)
    ax3.plot(simdf.date, totT,  color=col[i], label=label[i],linewidth=2)
    ax4.plot(simdf.date, totdE, color=col[i], label=label[i],linewidth=2)

    ax1.set_title('Nuovi Positivi')
    ax1.legend(loc='upper left')
    ax1.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax1.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax1.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=3))
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%y'))
    #ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    #ax1.set_xlim(left='2021-05-10',right='2021-06-15')
    #ax1.set_ylim([180000,400000])

    ax2.set_title('Ospedalizzati')
    ax2.legend(loc='upper left')
    #ax2.legend(loc = "upper right")
    ax2.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax2.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax2.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=3))
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%y'))
    #ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    #ax2.set_xlim(left='2021-05-10',right='2021-06-15')
    #ax2.set_ylim([7000,16000])

    ax3.set_title('Terapie intensive')
    ax3.legend(loc='upper left')
    ax3.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax3.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax3.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=3))
    ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%y'))
    #ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    #ax3.set_xlim(left='2021-05-10',right='2021-06-15')
    #ax3.set_ylim([1000,2200])
    
    ax4.set_title('Deceduti Giornalieri')
    ax4.legend(loc='upper left')
    ax4.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax4.xaxis.set_major_locator(mpl.dates.MonthLocator(bymonthday=[1,15]))
    ax4.xaxis.set_major_locator(mpl.dates.MonthLocator(interval=3))
    ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter('%b-%y'))
    #ax4.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d/%m'))
    #ax4.set_xlim(left='2021-05-10',right='2021-06-15')
    #ax4.set_ylim([122000,129000]) #Italia
    #ax4.set_ylim([31000,36000]) #Lombardia
    #ax4.set_ylim([6000,8500])  #Lazio
 
    #fig = plt.gcf()
    fig.set_size_inches((19.2, 10.8), forward=False)
    plt.hlines(1,'2021-09-06','2021-12-31',colors='grey',linestyles='dashed')
    fig2.set_size_inches((19.2, 10.8), forward=False)
    fig.savefig(ResPath + f'/scenari_omicron.png', dpi=300)
    fig2.savefig(ResPath + f'/scenari_omicron_Rt.png', dpi=300)
    #fig.savefig(ResPath + f'/{region}_giugno.png', dpi=300)
    
    #plt.savefig(ResPath + f'/case{i}/case{i}_IHTE_comp.png')
plt.show() 
plt.close()

