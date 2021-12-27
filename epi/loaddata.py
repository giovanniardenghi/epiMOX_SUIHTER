import sys
import numpy as np

# Read data file and build a dictionary
def readdata(fileName):
    with open(fileName) as f:
        content = f.readlines()

    Data = {}

    for line in content:
        if line.startswith('#'):
            pass
        elif '=' in line:
            a, b = line.split('=')
            tmp = {a.strip():b.strip()}
            Data.update(tmp)
        else:
            pass
    return(Data)


### Parse data of the programs

# Parse the dictionary: main program
def parsedata(Data):
    model, Nc = _parsedata_model(Data)

    country = Data['country']
    
    if 'param_type' in Data.keys():
        param_type =Data['param_type']
    else:
        param_type = 'const'
    param_file = Data['param_file']

    Tf = Data['Tf']
    dt = float(Data['dt'])

    if 'save_code' in Data.keys():
        save_code = bool(int(Data['save_code']))
    else:
        save_code = True

    if 'by_age' in Data.keys():
        by_age = bool(int(Data['by_age']))
    else:
        by_age = False

    if 'scenario' in Data.keys():
        scenario = Data['scenario']
    else:
        scenario = None

    if 'dataExtrapolation' in Data.keys():
        data_ext_deg = int(Data['dataExtrapolation'])
    else:
        data_ext_deg = None
    
    ext_deg = 0
    if 'extrapolation' in Data.keys():
        ext_deg = Data['extrapolation']
        if ext_deg not in ['exp','rbf']:
            try:
                ext_deg = int(ext_deg)
            except:
                raise ValueError("Error - 'extrapolation' can be only an integer, 'exp', or 'rbf'")


    edges_file = Data['edges_file']
    borders_file = Data['borders_file']
    if 'map_file' in Data.keys():
        map_file = Data['map_file']
    else:
        map_file = ""
    mobility = Data['mobility']
    if mobility not in ['mixing', 'transport']:
        sys.exit('Error - mobility type should be either \"mixing\" or \"transport\"')
    mixing = float(Data['mixing'])

    estim_param, DPC_start, DPC_ndays = _parsedata_estim(Data)

    if 'only_forecast' in Data.keys():
        only_forecast = bool(int(Data['only_forecast']))
    else:
        only_forecast = False

    out_type = Data['out_type']
    if out_type not in ['csv','h5']:
        sys.exit('Error - only \"csv\" and \"h5\" output are allowed')

    return(model, Nc, country, param_type, param_file, Tf, dt, save_code, by_age, edges_file, \
           borders_file, map_file, mobility, mixing, estim_param, DPC_start,\
           DPC_ndays, data_ext_deg, ext_deg, out_type, only_forecast, scenario)

# Parse the dictionary: plot
def parsedata_plot(Data):
    model, Nc = _parsedata_model(Data)

    CaseName = Data['CaseName']

    param_file = Data['param_file']

    out_type = Data['out_type']
    if out_type not in ['csv','h5']:
        sys.exit('Error - only \"csv\" and \"h5\" output are allowed')
    plotting_step = int(Data['plotting_step'])
    gen_video = bool(int(Data['gen_video']))
    shape_file = Data['shape_file']

    return(model, Nc, CaseName, param_file, \
           out_type, plotting_step, gen_video, shape_file)

### Parse subsections

# Parse the model
def _parsedata_model(Data):
    model = Data['model']
    if model not in ['SUIHTER', 'SEIRD']:
        sys.exit('Error - model should be either "SUIHTER" or "SEIRD"')
    elif model == 'SUIHTER':
        Nc = 7
    elif model == 'SEIRD':
        Nc = 5

    return(model, Nc)

# Parse the estimation
def _parsedata_estim(Data):
    estim_param = bool(int(Data['estim_param']))
    if 'DPC_start' in Data.keys():
        DPC_start = Data['DPC_start']
    DPC_ndays = Data['DPC_end']
    w_l = 0

    return(estim_param, DPC_start, DPC_ndays)

