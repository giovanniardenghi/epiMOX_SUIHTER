import pandas as pd
import matplotlib.pyplot as plt
import sys

file_name = sys.argv[1]

data = pd.read_csv(file_name)
data['forecast_date'] = pd.to_datetime(data.forecast_date)
data['target_end_date'] = pd.to_datetime(data.target_end_date)

targets = ['case', 'death']
weeks = 4

dpc = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/data/andamento_nazionale.csv')
dpc['data'] = pd.to_datetime([x[:10] for x in dpc.data])
dpc.set_index('data', inplace=True)

dpc['case'] = dpc['nuovi_positivi']
dpc['death'] = dpc['deceduti'].diff()

dpc = dpc[data.forecast_date[0]-pd.Timedelta(1,'day'):data.target_end_date.max()]

target_dates = sorted(list(set(data.target_end_date)))

for t in targets:
    point = []
    low = []
    high = []
    dpc_data = []
    for w in range(1,weeks+1):
        tmp = data[data['target'] == f'{w} wk ahead inc {t}'] 
        date_range = pd.date_range(target_dates[w-1]-pd.Timedelta(6,'days'),target_dates[w-1])
        point.append(tmp[tmp['type']=='point'].value.values[0])
        low.append(tmp[tmp['quantile']==0.025].value.values[0])
        high.append(tmp[tmp['quantile']==0.975].value.values[0])
        if all(x in dpc.index for x in date_range):
            dpc_data.append(dpc.loc[date_range,t].sum())
    plt.plot(target_dates, point, '.-', markersize=10, label='SUIHTER')
    plt.scatter(target_dates[:len(dpc_data)], dpc_data, color='k', label='DPC data')
    plt.fill_between(target_dates, low, high, alpha=0.3)
    plt.legend()
    plt.title(t)
    plt.xticks(target_dates)
    plt.savefig('./forecast_hub_images/'+file_name.split('/')[-1][:-4]+f'_{t}'+'.png')
    plt.show()
