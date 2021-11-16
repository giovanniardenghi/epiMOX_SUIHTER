# Definition of the parameters of the models
# At the moment parameters depend on space and time
import numpy as np
from scipy.special import erfc
import scipy.interpolate as si
from lmfit import Model
import functools
import datetime
import pandas as pd

# Mask function
def maskParams(params,m_mask):
    m_mask = np.invert(m_mask)
    return(np.ma.compressed( np.ma.masked_array( params, mask=m_mask) ))

def expgaussian(x, amplitude=1, center=0, sigma=1.0, gamma=1.0):
    """ an alternative exponentially modified Gaussian."""
    dx = center-x
    return amplitude* np.exp(gamma*dx) * erfc( dx/(np.sqrt(2)*sigma))

def EMGextrapol(x,y):

    model =  Model(expgaussian)
    params = model.make_params(sigma=10, gamma=0.01, amplitude=y.max(), center=y.argmax())
    result  = model.fit(y, params, x=x,  nan_policy='propagate')

    return result

# Utility for reading a section
def readSection(content,section,values):
    counter = 0
    data = np.zeros(values)
    sec_content = []
    found = False

    if values == 0:
        return()

    # extract from section
    for line in content:
        if line.startswith(section) or found:
            found = True
            sec_content.append(line)
        else:
            pass

    for line in sec_content:
        if line.startswith(section):
            pass
        elif line.startswith(b'#'):
            pass
        elif not line:
            pass
        else:
            tokens = line.split()
            for v in tokens:
                data[counter] = float(v)
                counter = counter + 1
                if counter == values:
                    return data

    return

def readTimes(content,section,values):
    counter = 0
    data = []
    sec_content = []
    found = False

    if values == 0:
        return()

    # extract from section
    for line in content:
        if line.startswith(section) or found:
            found = True
            sec_content.append(line)
        else:
            pass

    for line in sec_content:
        if line.startswith(section):
            pass
        elif line.startswith(b'#'):
            pass
        elif not line:
            pass
        else:
            data.append(datetime.date.fromisoformat(line.decode("utf-8").replace('\n','').replace('\r','')))  
            counter = counter + 1
            if counter == values:
                return data

    return


# Params class
class Params():

    def __init__(self,dataStart):
        self.nParams = 0
        self.nSites = 0
        self.nPhases = 0
        self.estimated = False
        self.times = np.zeros(0)
        self.dataStart = dataStart
        self.dataEnd = 0
        self.degree = 0
        self.extrapolator = []
        self.scenario = np.zeros((0,2)) 
        self.constant = np.zeros(0)
        self.constantSites = np.zeros(0)
        self.params  = np.zeros((0, 0))
        self.params_time = np.zeros((0,0))
        self.mask    = np.zeros((0, 0))
        self.lower_bounds = np.zeros((0, 0))
        self.upper_bounds = np.zeros((0, 0))
    def omegaI_vaccines(self,t): return 1 
    def gammaT_vaccines(self,t): return 1 
    def gammaH_vaccines(self,t): return 1 

    def get(self):
        return(self.params)

    def getMask(self):
        return( np.array(self.mask, dtype=bool) )

    def getConstant(self):
        return( np.array(self.constant, dtype=bool) )

    def getConstantSites(self):
        return( np.array(self.constantSites, dtype=bool) )

    def getLowerBounds(self):
        return(self.lower_bounds)

    def getUpperBounds(self):
        return(self.upper_bounds)

    def save(self,paramFileName):
        if paramFileName.lower().endswith(('.csv', '.txt')):
            self.__saveCsv__(paramFileName)
        elif paramFileName.lower().endswith('.npy'):
            self.__saveNpy__(paramFileName)
        return()

    def load(self,paramFileName):
        if paramFileName.lower().endswith(('.csv', '.txt')):
            self.__loadCsv__(paramFileName)
        elif paramFileName.lower().endswith('.npy'):
            self.__loadNpy__(paramFileName)
        return()

    def define_params_time(self, Tf):
        self.params_time = np.zeros((Tf+1,self.nParams,self.nSites)).squeeze()

    def compute_param_over_time(self,Tf):
        times = np.arange(0,Tf+1,).astype(int)
        for i,t in enumerate(times):
            self.params_time[i,self.getMask()[0]] = self.atTime(t)[self.getMask()[0]]
            #self.params_time[i] = self.atTime(t)


    def addPhase(self,ndays):
        self.nPhases += 1
        self.times        = np.append(self.times,ndays)
        self.params       = np.append(self.params,[self.params[-1]],axis=0)
        self.mask         = np.append(self.mask,[self.mask[-1]],axis=0)
        self.lower_bounds = np.append(self.lower_bounds,[self.lower_bounds[-1]],axis=0)
        self.upper_bounds = np.append(self.upper_bounds,[self.upper_bounds[-1]],axis=0)

    def getPhase(self,p,t):
        if self.constant[p]:
            phase = 0
        else:
            phase = self.nPhases-1
            for i, interval in enumerate(self.times):
                if ( t <= interval ):
                    phase = i 
                    break
        return (phase)

    def atTime(self,t):
        params_time = np.zeros((self.nParams,self.nSites)).squeeze()
        transient = 3
        if self.nSites==1:
            if self.dataEnd>0 and t>self.dataEnd:
                m = 1
                if len(self.scenario) > 0:
                    d,s = self.scenario.transpose()
                    i = np.searchsorted(d,t,side='right')-1
                    if i>=0:
                        if len(d)==1:
                            for q in range(self.nParams):
                                if i==0 and q==0 and (t-d[0])<=4:
                                    transient = 4
                                    params_time[0] = self.params[-1, 0] * (1 - (t - d[0]) / transient) + \
                                                     self.params[-1, 0] * s[0] * (t - d[0]) / transient
                                #elif q==3:
                                #    params_time[q] = np.maximum(self.scenario_extrapolator[q](t)*self.omegaI_vaccines(t), 0)
                                elif q==9:
                                    params_time[q] = np.maximum(self.scenario_extrapolator[q](t)*self.gammaT_vaccines(t), 0)
                                elif q==10:
                                    params_time[q] = np.maximum(self.scenario_extrapolator[q](t)*self.gammaH_vaccines(t), 0)
                                else:
                                    params_time[q] = np.maximum(self.scenario_extrapolator[q](t), 0)
                            return params_time
                        else:
                            t = d[0] - 1
                            m = s[i]
                        #if len(d)==1:
                        #    for q in range(self.nParams):
                        #        params_time[q] = np.maximum(self.scenario_extrapolator[q](t), 0)
                        #    return params_time

                params_time = np.array(self.params[-1])
                if type(self.degree)==int:
                    for q in range(self.nParams):
                        if q==0:
                            params_time[q] = np.maximum(self.extrapolator[q](t) * m,0)
                        elif q==3:
                            params_time[q] = np.maximum(self.extrapolator[q](t)*self.omegaI_vaccines(t), 0)
                        elif q==9:
                            params_time[q] = np.maximum(self.extrapolator[q](t)*self.gammaT_vaccines(t), 0)
                        elif q==10:
                            params_time[q] = np.maximum(self.extrapolator[q](t)*self.gammaH_vaccines(t), 0)
                        else:
                            params_time[q] = np.maximum(self.extrapolator[q](t),0)
                else:
                    params_time[0] = self.extrapolator(x=t) * m
                    params_time[3] *= self.omegaI_vaccines(t) 
                    params_time[9] *= self.gammaT_vaccines(t) 
                    params_time[10] *= self.gammaH_vaccines(t) 
            else:
                for p in range(self.nParams):
                    phase = self.getPhase(p,t)
                    phasetime = self.times[phase - 1]
                    if (t > phasetime + transient) or (phase == 0) or (abs(t-self.dataEnd)<6):
                        params_time[p] = self.params[phase,p]
                    else:
                        params_time[p] = self.params[ phase-1 , p ]*(1-(t-phasetime)/transient)+self.params[ phase , p ]*(t-phasetime)/transient
                    if p==9:
                        params_time[p] *= self.gammaT_vaccines(t)
                    elif p==10:
                        params_time[p] *= self.gammaH_vaccines(t)
        else:
            if self.dataEnd>0 and t>self.dataEnd:
                for p in range(self.nSites):
                    m = 1
                    if len(self.scenario) > 0:
                        d,s = self.scenario[p].transpose()
                        i = np.searchsorted(d,t,side='right')-1
                        if i>=0:
                            if len(d) == 1:
                                for q in range(self.nParams):
                                    params_time[q,p] = np.maximum(self.scenario_extrapolator[p][q](t), 0)
                                return params_time
                            else:
                                t = d[0] - 1
                                m = s[i]
                    params_time[:,p] = np.array(self.params[-1,:,p])
                    if type(self.degree)==int:
                        for q in range(self.nParams):
                            if q==0:
                                params_time[q,p] = np.maximum(self.extrapolator[p][q](t) * m,0)
                            else:
                                params_time[q,p] = np.maximum(self.extrapolator[p][q](t),0)
                    else:
                        params_time[0,p] = self.extrapolator[p](x=t) * m
            else:
                for p in range(self.nParams):
                    if self.constantSites[p]:
                        phase = self.getPhase(p, t)
                        phasetime = self.times[phase - 1]
                        if (t > phasetime + transient) or phase == 0 or (abs(t-self.dataEnd)<6):
                            params_time[p,:] = self.params[phase, p, 0]
                        else:
                            params_time[p,:] = self.params[phase - 1, p, 0] * (1 - (t - phasetime) / transient) + self.params[phase, p, 0] * (t - phasetime) / transient
                    else:
                        for s in range(self.nSites):
                            phase = self.getPhase(p, t)
                            phasetime = self.times[phase - 1]
                            if (t > phasetime + transient) or phase == 0 or (abs(t-self.dataEnd)<6):
                                params_time[p,s] = self.params[phase, p, s]
                            else:
                                params_time[p,s] = self.params[phase - 1, p, s] * (1 - (t - phasetime) / transient) + self.params[phase, p, s] * (t - phasetime) / transient
        return params_time

    def atPhase(self,i):
        return(self.params[ i , ...])

    def atSite(self,i): # works only if more than 1 site
        if self.nSites > 1:
            return(self.params[ ... , i ])
        return ()

    def forecast(self, DPC_time, Tf, deg, scenarios=None):
        if DPC_time>=Tf:
            return ()
        self.degree = deg
        self.dataEnd = DPC_time
        tmp_times = np.concatenate(([0],self.times,[self.dataEnd]))
        if self.nSites == 1:
            if type(self.degree)==int:
                x = tmp_times[-(deg+1):]
                self.extrapolator = []
                for q in range(self.nParams):
                    y = self.get()[-(deg+1):,q]
                    self.extrapolator.append(np.poly1d(np.polyfit(x,y,self.degree)))
            elif self.degree == 'exp':
                x = tmp_times[1:]
                y = self.get()[:,0]
                EMG = EMGextrapol(x,y)
                self.extrapolator = functools.partial(EMG.eval,**EMG.best_values)
        else:
            self.extrapolator = []
            if type(self.degree)==int:
                for p in range(self.nSites):
                    tmp_extrapolator = []
                    x = tmp_times[-(deg+1):]
                    for q in range(self.nParams):
                        y = self.get()[-(deg+1):,q,p]
                        tmp_extrapolator.append(np.poly1d(np.polyfit(x,y,self.degree)))
                    self.extrapolator.append(tmp_extrapolator)
            elif self.degree == 'exp':
                x = tmp_times[1:]
                for p in range(self.nSites):
                    y = self.get()[:,0,p]
                    EMG = EMGextrapol(x,y)
                    self.extrapolator.append(functools.partial(EMG.eval,**EMG.best_values))

        if scenarios is not None:
            self.scenario = scenarios
            if self.nSites != 1:
                if len(scenarios.shape) == 2:
                    self.scenario = np.tile(self.scenario,(self.nSites,1,1))
        return ()

    def extrapolate_scenario(self):
        if self.nSites == 1:
            if self.scenario.shape[0] != 1:
                return()
            d,s = self.scenario.transpose()
            tmp_times = np.concatenate(([0],self.times,[self.dataEnd],d))
            if type(self.degree)==int:
                x = tmp_times[-(self.degree+1):]
                #x = tmp_times[-1:]
                self.scenario_extrapolator = []
                for q in range(self.nParams):
                    if q==0:
                        y = np.concatenate((self.get()[:,q],self.extrapolator[q](d-1)*s))
                    else:
                        y = np.concatenate((self.get()[:,q],self.extrapolator[q](d)))
                    self.scenario_extrapolator.append(np.poly1d(np.polyfit(x,y[-(self.degree+1):],self.degree)))
                    #self.scenario_extrapolator.append(np.poly1d(np.polyfit(x,y[-1:],0)))
        else:
            if self.scenario.shape[1] != 1:
                return()
            self.scenario_extrapolator = []
            for p in range(self.nSites):
                d,s = self.scenario[p].transpose()
                tmp_times = np.concatenate(([0],self.times,[self.dataEnd],d))
                if type(self.degree)==int:
                    x = tmp_times[-(self.degree+1):]
                    tmp_scenario_extrapolator = []
                    for q in range(self.nParams):
                        if q==0:
                            y = np.concatenate((self.get()[:,q,p],self.extrapolator[p][q](d-1)*s))
                        else:
                            y = np.concatenate((self.get()[:,q,p],self.extrapolator[p][q](d)))
                        tmp_scenario_extrapolator.append(np.poly1d(np.polyfit(x,y[-(self.degree+1):],self.degree)))
                    self.scenario_extrapolator.append(tmp_scenario_extrapolator)
        return ()
    
#    def vaccines_effect_omega(self):
#        age_data = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/stato_clinico.csv')
#        vaccines = pd.read_csv('https://raw.githubusercontent.com/italia/covid19-opendata-vaccini/master/dati/somministrazioni-vaccini-latest.csv')
#        age_groups = {'0-9': '0-19',
#              '10-19': '0-19',
#              '20-29': '20-39',
#              '30-39': '20-39',
#              '40-49': '40-59',
#              '50-59': '40-59',
#              '60-69': '60-79',
#              '70-79': '60-79',
#              '80-89': '80-89',
#              '>90': '90+'}
#        vaccines['data_somministrazione'] = pd.to_datetime(vaccines.data_somministrazione)
#        vaccines[
#        vaccines = vaccines.groupby(['data_somministrazione',age_groups],level=[0,1]).sum()
#        print(vaccines)
#        age_data['Data'] = pd.to_datetime(age_data.Data)
#        age_data = age_data[age_data['Data']>=pd.to_datetime(self.dataStart)]
#        age_data = age_data[age_data['Data']<=pd.to_datetime(self.dataStart)+pd.Timedelta(self.dataEnd,'days')]
#        ages_dfs = [x[['Data','Isolated','Hospitalized']].set_index('Data') for ages,x in age_data.groupby('Età')]
#        f_I = [si.interp1d(range(len(x)),x.Isolated.rolling(window=7,min_periods=1,center=True).mean(),fill_value="extrapolate") for x in ages_dfs]
#        f_H = [si.interp1d(range(len(x)),x.Hospitalized.rolling(window=7,min_periods=1,center=True).mean(),fill_value="extrapolate") for x in ages_dfs]
#        ages_dfs = [x.reset_index(drop=True) for x in ages_dfs]
#        medie = pd.DataFrame(columns=['Isolated','Hospitalized'])
#        for i,x in enumerate(ages_dfs):
#            medie = medie.append(x[int(self.times[-1])+1:].mean(),ignore_index=True)
#        def omegaI_reduction(t):
#            multiplier=0
#            for i,x in enumerate(ages_dfs):
#                multiplier += np.clip(f_H[i](t),0,1)**2/np.clip(f_I[i](t-5),0,1)
#            return multiplier/np.sum(medie.Hospitalized.values**2/medie.Isolated.values)
#        self.omegaI_vaccines = omegaI_reduction


    def vaccines_effect_omega(self):
        age_data = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/stato_clinico.csv')
        age_data['Data'] = pd.to_datetime(age_data.Data)
        age_data = age_data[age_data['Data']>=pd.to_datetime(self.dataStart)]
        age_data = age_data[age_data['Data']<=pd.to_datetime(self.dataStart)+pd.Timedelta(self.dataEnd,'days')]
        ages_dfs = [x[['Data','Isolated','Hospitalized']].set_index('Data') for ages,x in age_data.groupby('Età')]
        f_I = [si.interp1d(range(len(x)),x.Isolated.rolling(window=7,min_periods=1,center=True).mean(),fill_value="extrapolate") for x in ages_dfs]
        f_H = [si.interp1d(range(len(x)),x.Hospitalized.rolling(window=7,min_periods=1,center=True).mean(),fill_value="extrapolate") for x in ages_dfs]
        ages_dfs = [x.reset_index(drop=True) for x in ages_dfs]
        medie = pd.DataFrame(columns=['Isolated','Hospitalized'])
        for i,x in enumerate(ages_dfs):
            medie = medie.append(x[int(self.times[-1])+1:].mean(),ignore_index=True)
        def omegaI_reduction(t):
            multiplier=0
            for i,x in enumerate(ages_dfs):
                multiplier += np.clip(f_H[i](t),0,1)**2/np.clip(f_I[i](t-5),0,1)
            return multiplier/np.sum(medie.Hospitalized.values**2/medie.Isolated.values)
        self.omegaI_vaccines = omegaI_reduction

    def vaccines_effect_gammaT(self):
        age_data = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/stato_clinico.csv')
        age_data['Data'] = pd.to_datetime(age_data.Data)
        age_data = age_data[age_data['Data']>=pd.to_datetime(self.dataStart)]
        age_data = age_data[age_data['Data']<=pd.to_datetime(self.dataStart)+pd.Timedelta(self.dataEnd,'days')]
        ages_dfs = [x[['Data','Threatened','Extinct','Daily_extinct']].set_index('Data') for ages,x in age_data.groupby('Età')]
        f_T = [si.interp1d(range(len(x)),x.Threatened.rolling(window=7,min_periods=1,center=True).mean(),kind='nearest',fill_value="extrapolate") for x in ages_dfs]
        f_dE = [si.interp1d(range(len(x)),x.Daily_extinct.rolling(window=14,min_periods=1,center=True).mean(),kind='nearest',fill_value="extrapolate") for x in ages_dfs]
        f_E = [si.interp1d(range(len(x)),x.Extinct.rolling(window=7,min_periods=1,center=True).mean(),kind='nearest',fill_value="extrapolate") for x in ages_dfs]
        ages_dfs = [x.reset_index(drop=True) for x in ages_dfs]
        medie = pd.DataFrame(columns=['Threatened','Extinct','Daily_etinct'])
        for i,x in enumerate(ages_dfs):
            medie = medie.append(x[:int(self.times[6])].mean(),ignore_index=True)
        global gammaT_reduction
        def gammaT_reduction(t):
            multiplier=0
            for i,x in enumerate(ages_dfs):
                #multiplier += np.clip(f_E[i](t),0,1)*np.clip(f_dE[i](t),0,1)/np.clip(f_T[i](t-5),1e-5,1)
                multiplier += np.clip(f_dE[i](t),0,1)*medie.iloc[i].Daily_extinct/medie.iloc[i].Threatened
            return np.max(multiplier/np.sum(medie.Extinct.values*medie.Daily_extinct.values/medie.Threatened),0)
        self.gammaT_vaccines = gammaT_reduction

    def vaccines_effect_gammaH(self):
        age_data = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/dpc-covid-data/main/SUIHTER/stato_clinico.csv')
        age_data['Data'] = pd.to_datetime(age_data.Data)
        age_data = age_data[age_data['Data']>=pd.to_datetime(self.dataStart)]
        age_data = age_data[age_data['Data']<=pd.to_datetime(self.dataStart)+pd.Timedelta(self.dataEnd,'days')]
        ages_dfs = [x[['Data','Hospitalized','Extinct','Daily_extinct']].set_index('Data') for ages,x in age_data.groupby('Età')]
        f_H = [si.interp1d(range(len(x)),x.Hospitalized.rolling(window=7,min_periods=1,center=True).mean(),kind='nearest',fill_value="extrapolate") for x in ages_dfs]
        f_dE = [si.interp1d(range(len(x)),x.Daily_extinct.rolling(window=14,min_periods=1,center=True).mean(),kind='nearest',fill_value="extrapolate") for x in ages_dfs]
        f_E = [si.interp1d(range(len(x)),x.Extinct.rolling(window=7,min_periods=1,center=True).mean(),kind='nearest',fill_value="extrapolate") for x in ages_dfs]
        ages_dfs = [x.reset_index(drop=True) for x in ages_dfs]
        medie = pd.DataFrame(columns=['Hospitalized','Extinct','Daily_extinct'])
        for i,x in enumerate(ages_dfs):
            medie = medie.append(x[:int(self.times[6])].mean(),ignore_index=True)
        global gammaH_reduction
        def gammaH_reduction(t):
            multiplier=0
            for i,x in enumerate(ages_dfs):
                #multiplier += np.clip(f_E[i](t),0,1)*np.clip(f_dE[i](t),0,1)/np.clip(f_T[i](t-5),1e-5,1)
                multiplier += np.clip(f_dE[i](t),0,1)*medie.iloc[i].Daily_extinct/medie.iloc[i].Hospitalized
            return np.max(multiplier/np.sum(medie.Extinct.values*medie.Daily_extinct.values/medie.Hospitalized),0)
        self.gammaH_vaccines = gammaH_reduction

    def __saveCsv__(self,paramFileName):
        with open(paramFileName, "w") as f:
            print(f"[nParams]", file=f)
            print(self.nParams, file=f)
            print(f"[nSites]", file=f)
            print(self.nSites, file=f)
            print(f"[nPhases]", file=f)
            print(self.nPhases, file=f)
            print(f"[times]", file=f)
            if len(self.times) != 0:
                tmp = '\n'.join(map(lambda x: (self.dataStart + datetime.timedelta(days=int(x))).isoformat(), self.times))
                print(tmp, file=f)
            print(f"[constant]", file=f)
            if len(self.constant) != 0:
                tmp = ' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in self.constant)
                print(tmp, file=f)
            if len(self.constantSites) != 0:
                print(f"[constantSites]", file=f)
                tmp = ' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in self.constantSites)
                print(tmp, file=f)
            print(f"[Estimated]", file=f)
            print(int(self.estimated),file=f)
            print("", file=f)
            print(f"[params]", file=f)
            if len(self.params) != 0:
                if self.nSites==1:
                    tmp = '\n'.join(' '.join(np.format_float_positional(x,precision=8,pad_right=8).rstrip('0').rstrip('.') \
                        for x in y) for y in self.params)
                else:
                    tmp = '\n\n'.join('\n'.join(' '.join(np.format_float_positional(x,precision=8,pad_right=8).rstrip('0').rstrip('.') \
                        for x in y) for y in z) for z in np.moveaxis(self.params,-1,0))
                print(tmp, file=f)
            print("", file=f)
            print(f"[mask]", file=f)
            if len(self.mask) != 0:
                if self.nSites == 1:
                    tmp = '\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in self.mask)
                else:
                    tmp = '\n\n'.join('\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in z) for z in np.moveaxis(self.mask,-1,0))
                print(tmp, file=f)
            print("", file=f)
            print(f"[l_bounds]", file=f)
            if len(self.lower_bounds) != 0:
                if self.nSites == 1:
                    tmp = '\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in self.lower_bounds)
                else:
                    tmp = '\n\n'.join('\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in z) for z in np.moveaxis(self.lower_bounds,-1,0))
                print(tmp, file=f)
            print("", file=f)
            print(f"[u_bounds]", file=f)
            if len(self.upper_bounds) != 0:
                if self.nSites == 1:
                    tmp = '\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in self.upper_bounds)
                else:
                    tmp = '\n\n'.join('\n'.join(' '.join(('%f' % x).rstrip('0').rstrip('.') \
                         for x in y) for y in z) for z in np.moveaxis(self.upper_bounds,-1,0))
                print(tmp, file=f)

    def __saveNpy__(self,paramFileName):
        with open(paramFileName, 'wb') as f:
            np.savez(f, nParams = self.nParams, \
                        nSites = self.nSites, \
                        nPhases = self.nPhases, \
                        estimated = self.estimated,\
                        times = self.times, \
                        constant = self.constant, \
                        params = self.params, \
                        mask = self.mask, \
                        lower_bounds = self.lower_bounds, \
                        upper_bounds = self.upper_bounds )

    def __loadCsv__(self,paramFileName):
        with open(paramFileName, 'rb') as f:
            content = f.readlines()

        self.nParams = int(readSection(content,b'[nParams]',1))
        try:
            self.nSites = int(readSection(content,b'[nSites]',1))
        except:
            self.nSites = 1
        self.nPhases = int(readSection(content,b'[nPhases]',1))

        tmp = readTimes(content, b'[times]', self.nPhases - 1)
        self.times = np.reshape([int((x-self.dataStart).days) for x in tmp],self.nPhases - 1)
        try:
            self.constant = np.reshape( \
            readSection(content, b'[constant]', self.nParams), \
            self.nParams)
        except:
            self.constant = np.zeros(self.nParams)
        if self.nSites > 1:
            try:
                self.constantSites = np.reshape( \
                readSection(content, b'[constantSites]', self.nParams), \
                self.nParams)
            except:
                self.constantSites = np.zeros(self.nParams)
        try:
            self.estimated = bool(readSection(content,b'[Estimated]',1))
        except:
            self.estimated = False
        nParams = self.nParams * self.nPhases if not self.estimated else self.nParams * self.nPhases * self.nSites
        self.params = readSection(content, b'[params]', nParams)
        if not self.estimated:
            self.params = np.tile(self.params, (self.nSites,1))
        self.params = np.reshape(self.params, (self.nSites, self.nPhases, self.nParams))
        self.params=np.moveaxis(self.params,0,-1).squeeze()
        self.mask = readSection(content, b'[mask]', nParams)
        if not self.estimated:
            self.mask = np.tile(self.mask, (self.nSites,1))
        self.mask = np.reshape(self.mask, (self.nSites, self.nPhases, self.nParams))
        self.mask=np.moveaxis(self.mask,0,-1).squeeze()
        self.lower_bounds = readSection(content, b'[l_bounds]', nParams)
        if not self.estimated:
            self.lower_bounds = np.tile(self.lower_bounds, (self.nSites,1))
        self.lower_bounds = np.reshape(self.lower_bounds, (self.nSites,self.nPhases, self.nParams))
        self.lower_bounds = np.moveaxis(self.lower_bounds,0,-1).squeeze()
        self.upper_bounds = readSection(content, b'[u_bounds]', nParams)
        if not self.estimated:
            self.upper_bounds = np.tile(self.upper_bounds, (self.nSites,1))
        self.upper_bounds = np.reshape(self.upper_bounds, (self.nSites, self.nPhases, self.nParams))
        self.upper_bounds = np.moveaxis(self.upper_bounds,0,-1).squeeze()

    def __loadNpy__(self,paramFileName):
        with open(paramFileName, 'rb') as f:
            data = np.load(f)
            self.nParams = data['nParams']
            try:
                self.nSites = data['nSites']
            except:
                self.nSites = 1
            self.nPhases = data['nPhases']
            self.times = data['times']
            try:
                self.constant = data['constant']
            except:
                self.constant= np.zeros(self.nPhases)
            try:
                self.estimated = data['estimated']
            except:
                self.estimated = False
            self.params = data['params']
            self.mask = data['mask']
            self.lower_bounds = data['lower_bounds']
            self.upper_bounds = data['upper_bounds']
