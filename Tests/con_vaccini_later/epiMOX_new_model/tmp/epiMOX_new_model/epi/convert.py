import sys
import pandas as pd
import numpy as np

def converter(model, dati, regione, Nc):

    if regione=='Regions':
        res = dati[:][["data", "codice_regione", "ricoverati_con_sintomi", "terapia_intensiva",
                       "isolamento_domiciliare","nuovi_positivi", "dimessi_guariti", "deceduti","ingressi_terapia_intensiva"]]
    elif regione=='Italia':
        res = dati[:][["data", "ricoverati_con_sintomi", "terapia_intensiva",
                       "isolamento_domiciliare","nuovi_positivi", "dimessi_guariti", "deceduti","ingressi_terapia_intensiva"]]
        geocode = [0]*len(res)
    else:
        dati = dati[dati['denominazione_regione']==regione]
        if len(dati)==0:
            sys.exit('Error - please provide a valid region name')
        res = dati[:][["data", "ricoverati_con_sintomi", "terapia_intensiva",
                       "isolamento_domiciliare","nuovi_positivi", "dimessi_guariti", "deceduti","ingressi_terapia_intensiva"]]
        geocode = dati['codice_regione'].values 

    if regione == 'Regions':
        res = res.sort_values(['codice_regione','data'])
        geocode = res['codice_regione'].values
    else:
        res = res.sort_values("data")
    res = res.reset_index(drop=True)
    res_np = np.zeros((len(res), 4 + Nc))
    res_np[:, 0] = geocode  # Geocode
    res_np[:, 1] = np.tile(np.arange(0, len(res)/len(set(geocode)), 1),len(set(geocode)))

    if model == 'SUIHTER':
        # Diagnosed
        res_np[:, 4] = res['isolamento_domiciliare'].values
        # Recognized
        res_np[:, 5] = res['ricoverati_con_sintomi'].values
        # Threatened
        res_np[:, 6] = res['terapia_intensiva'].values
        # Extinct
        res_np[:, 7] = res['deceduti'].values
        # Healed, actually discharged from hospital and healed
        res_np[:, 8] = res['dimessi_guariti'].values
        # New positives
        res_np[:, 9] = res['nuovi_positivi'].values
        # New treathened 
        res_np[:, 10] = res['ingressi_terapia_intensiva'].values
    elif model == 'SEIRD':
        # Infected
        res_np[:, 4] = res['ricoverati_con_sintomi'].values + \
                       res['terapia_intensiva'].values + \
                       res['isolamento_domiciliare'].values
        # Recovered
        res_np[:, 5] = res['dimessi_guariti'].values
        # Dead
        res_np[:, 6] = res['deceduti'].values

    if model == 'SUIHTER':
        results_df = pd.DataFrame(res_np, columns=['Geocode', 'time', 'Suscept', 'Undetected', 'Isolated',
                                                   'Hospitalized', 'Threatened', 'Extinct', 'Recovered', 'New_positives','New_threatened'])
        results_df['data'] = res['data']
    elif model == 'SEIRD':
        results_df = pd.DataFrame(res_np, columns=['Geocode', 'time', 'Suscept', 'Exposed', 'Infected',
                                                   'Recovered', 'Dead'])
        results_df['data'] = res['data']
    results_df = results_df.astype({"Geocode": int})
    results_df = results_df.sort_values(by=['Geocode', 'time'])

    return results_df
