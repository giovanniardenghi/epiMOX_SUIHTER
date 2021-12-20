import os.path
import zipfile
import pandas as pd

def zipdir(path, ziph):
    # ziph is zipfile handle
    exclude = ['Tests','__pycache__','venv','.git','.gitignore','forecast_hub','forecast_hub_images']
    for root, dirs, files in os.walk(path, topdown=True):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))

def correct_isolated_model(df, comp, date, DT, C=None, total=None, healed=None):
    if not C:
        C = df.loc[date,comp]-df.loc[date-pd.Timedelta(1,'day'),comp]
    Itot = df.loc[date-pd.Timedelta(DT,'days'):date-pd.Timedelta(1,'day'),comp].sum()
    Cn =df.loc[date-pd.Timedelta(DT,'days'):date-pd.Timedelta(1,'day'),comp]/ Itot * C
    df.loc[date-pd.Timedelta(DT,'days'):date-pd.Timedelta(1,'day'),comp] += Cn.cumsum()
    if total:
        df.loc[date - pd.Timedelta(DT, 'days'):date - pd.Timedelta(1, 'day'), total] += Cn.cumsum()
    if healed:
        df.loc[date - pd.Timedelta(DT, 'days'):date - pd.Timedelta(1, 'day'), healed] -= Cn.cumsum()
    return df

def correct_data(eData, country):
    corrections = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/corrections.csv',keep_default_na=False,na_values=['NaN'])
    if country != 'Italia':
        corrections = corrections[corrections['region']==country]
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
    return eData

