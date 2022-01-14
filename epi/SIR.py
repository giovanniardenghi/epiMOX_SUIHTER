import numpy as np
import pandas as pd
from scipy.optimize import Bounds
from optimparallel import minimize_parallel
from epi import parameters_const as pm

# SUIHTER model class
class SIR:

    def __init__(self, Nc, params, t_list, DPC_start, DPC_end, data, Pop, 
                       by_age, codes, DO, out_path, out_type='h5'):
        # initialize compartments
        # Y0: Nc x Ns
        self.Y0 = np.zeros((Nc, Pop.size)).squeeze()
        # Y: Nc x T x Ns
        self.Y = np.zeros((Nc, t_list[-1]+1, Pop.size)).squeeze()
        #initialize parameters
        self.params = params
        self.Ns = Pop.size

        self.t_list = t_list
        self.DPC_start = DPC_start
        self.DPC_end = DPC_end

        self.data = data

        self.forecast = False
        
        self.Pop = Pop

        self.by_age = by_age 
        self.codes = codes

        self.DO = DO if by_age else 1

        self.out_type = out_type
        self.out_path = out_path

        self.initialize_compartments()

    def initialize_compartments(self):
        initI = self.data[self.data['time']==0].copy() 
        self.Y0[1] = initI['Infected']
        self.Y0[2] = initI['Removed']
        self.Y0[0] = self.Pop - self.Y0[1:].sum(axis=0)
        return

    def model(self, t, y0):
        t_int  = int(np.floor(t))
        beta, gamma = self.params.params_time[t_int]
        S, I, R = y0
        
        dSdt = - beta * S * np.dot(self.DO, I / self.Pop) 
        dIdt = beta * S * np.dot(self.DO, I / self.Pop) - gamma * I 
        dRdt = gamma * I 

        return np.vstack((dSdt, dIdt, dRdt)).squeeze()

    def model_MCMC(self, params, data):
        t_list = data.xdata[0].squeeze()
        self.t_list = t_list.copy()
        self.params.params[self.params.getMask()] = params[:-2*self.Ns]
        self.params.forecast(self.params.dataEnd,self.t_list[-1],0,None)
        
        Y0 = data.ydata[0].squeeze()

        self.Y0 = Y0.copy()
        self.Y0[1] *= params[-2*self.Ns:-self.Ns]
        self.Y0[2] *= params[-1*self.Ns:]
        self.Y0[0] = Pop - self.Y0[1:].sum(axis=0)

        self.solve()
        
        forecast = data.user_defined_object[0][0]

        if forecast:
            T0 = int(self.data.time.iloc[-1])
            self.t_list = np.arange(T0, self.t_list[-1]+1) 
            self.Y0 = self.Y[...,T0].copy()
            self.solve()
        
        results = self.Y[:,self.t_list].copy()
        return results.transpose()
                
    def solve(self):
        t_start = int(self.t_list[0])
        self.params.compute_param_over_time(int(self.t_list[-1]))
        self.Y[:,t_start] = self.Y0 
        for i,t in enumerate(self.t_list[:-1]):
            y0 = self.Y[:,i+t_start]
            k1=self.model(t      , y0     )
            k2=self.model(t+0.5, y0+0.5*k1)
            k3=self.model(t+0.5, y0+0.5*k2)
            k4=self.model(t+1  , y0+    k3)
            self.Y[:,t_start+i+1] = y0 + (k1+2*(k2+k3)+k4)/6.0
        return

    def estimate(self):
        params0 = pm.maskParams( self.params.get() , self.params.getMask() )
        lower_b = pm.maskParams( self.params.getLowerBounds() , self.params.getMask() )
        upper_b = pm.maskParams( self.params.getUpperBounds() , self.params.getMask() )
        bounds = Bounds( lower_b, upper_b )
        
        #local mimimization
        result = minimize_parallel(self.error_LS, params0, bounds=bounds,\
                options={'ftol': 1e-15, 'maxfun':1000, 'maxiter':1000,'iprint':1})
        print('###########################################')
        print(result)
        print('###########################################')
        # assign estimated parameters to model pparameters
        self.params.params[self.params.getMask()] = result.x
        return

    def error(self, params0):
        self.params.params[self.params.getMask()] = params0
        self.solve()
        S, I, R = self.Y[:,self.t_list]

        # compute errors
        # Flatten the solution arrays to match data format
        errorI = I.flatten() - self.data['Infected'].values 

        # compute errors weights
        one = np.ones(len(errorI))
        weight = np.ones(len(errorI)) # modify this if you want to weight more specific time steps
        weightsI = weight/np.maximum(self.data['Infected'].values,one)

        errorL2 = (errorI ** 2*weightsI).sum()

        return errorL2
    
    def error_LS(self, params):
        error = self.error(params)
        return np.sqrt(error)

    def error_MCMC(self, params, data):
        Y0 = data.ydata[0].squeeze()

        self.Y0 = Y0.copy()
        self.Y0[1] *= params[-2*self.Ns:-self.Ns]
        self.Y0[2] *= params[-1*self.Ns:]
        self.Y0[0] = self.Pop - self.Y0[1:].sum(axis=0)

        return self.error(params[:-2*self.Ns])

    def computeRt(self):
        nPhases = self.params.nPhases
        nSites = self.params.nSites
        self.Rt = np.zeros((len(self.t_list), nSites)).squeeze()
        for i,t in enumerate(self.t_list):
            beta,gamma = self.params.params_time[t]
            R0_tmp = beta / gamma
            self.Rt[i] = R0_tmp * self.Y[0] / self.Pop
        np.savetxt(self.out_path+'/Rt.csv', self.Rt, delimiter=',')
        return

    def save(self):
        print('Reorganizing and saving results...')
        # Sum undetected from base and variant
        self.Y = self.Y[:,self.t_list]

        Nc = self.Y.shape[0]
        codes = np.tile(self.codes, len(self.t_list))
        times = np.repeat(self.t_list, len(self.codes))
        dates = [self.DPC_start + pd.Timedelta(t, 'days') for t in times]

        results = np.zeros((Nc+3, len(times)), dtype='O')
        
        results[:3] = codes, dates, times
        results[3:3+Nc] = self.Y.reshape(Nc, len(times))
        
        Code = "Age" if self.by_age else "Geocode"
        results_df = pd.DataFrame(results.T,columns=[Code,'date','time','Suscept','Infected','Removed'])
        
        if not self.by_age:
            results_df = results_df.astype({Code: int,"time": 'float64'})
        else:
            results_df = results_df.astype({Code: str,"time": 'float64'})

        results_df = results_df.sort_values(by=[Code,'date'])
        results_df = results_df.astype(dict(zip(['Suscept','Infected','Removed'],['float64']*3)))

        outFileName = self.out_path+'/simdf.'+self.out_type
        if self.out_type == 'csv':
            results_df.to_csv(outFileName,index=False)
        elif self.out_type == 'h5':
            results_df.to_hdf(outFileName, key='results_df', mode='w')
        print('...done!')

        return

    def plot_Rt_vs_ISS(self):
        day_init = self.DPC_start
        day_end = self.DPC_start + pd.Timedelta(t_list[-1], 'days')
        Rt_ISS = pd.read_csv('https://raw.githubusercontent.com/giovanniardenghi/iss-rt-data/main/data/iss_rt.csv')
        Rt_ISS['Data'] = pd.to_datetime(Rt_ISS.Data)
        Rt_ISS.set_index('Data',inplace=True)
        plt.plot(pd.date_range(day_init+pd.Timedelta(self.t_list[0],'days'), day_end), self.Rt, linewidth = 4, label = 'Rt SUIHTER')
        plt.plot(Rt_ISS[day_init:day_end].index,Rt_ISS[day_init:day_end], linewidth = 4, label = 'Rt ISS' )
        plt.legend(fontsize=20)
        fig = plt.gcf()
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize='large')
        ax.xaxis.set_major_locator(mpl.dates.DayLocator(interval=10))
        ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d %b'))
        fig.set_size_inches((19.2, 10.8), forward=False)
        plt.savefig(out_path + '/Rt_plot.png', dpi=300)
        plt.close()
        return
