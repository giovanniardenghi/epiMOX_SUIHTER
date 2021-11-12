import sys
import numpy as np
import pandas as pd
import datetime
import os.path
from epi import loaddata as ld
from epi import estimation as es
from epi.convert import converter
from epi.MCMC import solveMCMC
from epiMOX_test import epiMOX,correct_isolated_model

if __name__ == '__main__':
    # Read data file
    if len(sys.argv) < 4:
        sys.exit("Error - at least 3 arguments required: path to the folder containing the 'input.inp' file, last day of data to use for calibration in iso format (e.g. '2021-01-01'), first day of the simulation in iso format")
    testPath = sys.argv[1]
    fileName = testPath + '/input.inp'
    if os.path.exists(fileName):
        DataDic = ld.readdata(fileName)
    else:
        sys.exit('Error - Input data file ' + fileName + ' not found. Exit.')
    if len(sys.argv) > 4:
        nsimu = sys.argv[4]
    else:
        nsimu = 1e4
    DPC_start = sys.argv[3]
    DPC_ndays = sys.argv[2]

    if len(sys.argv) > 5:
        parallel = bool(sys.argv[5])
    else:
        parallel = False

    if len(sys.argv) > 6:
        nproc = int(sys.argv[6])
    else:
        nproc = 3

    if len(sys.argv) > 7:
        nchains = int(sys.argv[7])
    else:
        nchains = 3

    Ns, Nc, sol, epi_model, params, Pop, DO, map_to_prov, dv1, dv2, dt = epiMOX(testPath)
    if 'country' in DataDic.keys():
        region = DataDic['country']
    else:
        region = 'Italia'

    epi_start = datetime.date(year=2020, month=2, day=24)

    day_init = pd.to_datetime(DPC_start)
    day_end = pd.to_datetime(DPC_ndays) + pd.Timedelta(1,'day')

    if region == 'Italia':
        eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv')
    else:
        eData = pd.read_csv('https://raw.githubusercontent.com/pcm-dpc/COVID-19/master/dati-regioni/dpc-covid19-ita-regioni.csv')


    eData['data'] = [x[:10] for x in eData.data]
    eData['data'] = pd.to_datetime(eData.data)

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
        eData = correct_isolated_model(eData.set_index('data'), comp_to_old[row['comp']].replace(' ', '_').lower(),
                                      pd.to_datetime(row['date']), int(row['DT']), int(row['C']),
                                      comp_to_old[row['Positivi']].replace(' ', '_').lower(),
                                      comp_to_old[row['Guariti']].replace(' ', '_').lower()).reset_index().rename(
            columns={'index': 'data'})
    eData['data'] = eData.data.dt.strftime('%Y-%m-%dT%H:%M:%S')

    eData['data'] = pd.to_datetime(eData.data)
    eData = eData[(eData["data"] >= day_init) & (eData["data"] < day_end)]
    eData = eData.reset_index(drop=True)
    eDataG = converter(epi_model, eData, region, Nc)
    eDataG = eDataG.reset_index(drop=True)
    #Y0 = [sol[x * Ns:(x + 1) * Ns].sum() for x in range(Nc)]
    Y0 = sol
    Nstep = int((day_end - day_init).days)
    R_d = np.zeros(Nstep+1)
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
        extinct_diff = np.concatenate([[np.nan],eDataG['Extinct'].diff().iloc[1:].rolling(window=7,min_periods=1,center=True).mean().values])
        extinct_diff =eDataG['Extinct'].diff().values
        recovered = eDataG['Recovered'].values
        undetected = es.estimateUndetected(isolated,hospitalized,threatened,recovered,extinct)
        new_positive = eDataG['New_positives'].values
        obs = np.concatenate([isolated,hospitalized,threatened, extinct_diff, recovered, new_positive], axis=0).reshape((6,-1))
    nstep = int(eDataG['time'].max() / dt)
    times = np.arange(0, nstep + 1) * dt
    mcstat = solveMCMC(testPath,times, obs, Y0, l_args, nsimu=nsimu, sigma=np.array([0.1 * 3e2]*6), epi_model=epi_model, parallel=parallel, nproc=nproc, nchains=nchains)
