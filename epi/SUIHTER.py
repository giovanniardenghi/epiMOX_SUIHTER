import numpy as np
import pandas as pd
from scipy.optimize import Bounds
from optimparallel import minimize_parallel
from epi import parameters_const as pm

# SUIHTER model class
class SUIHTER:

    def __init__(self, Y0, params, t_list, DPC_start, DPC_end, data, Pop, 
                       by_age, codes, vaccines, maxV, out_path, sigma1=0.29, sigma2=0.12, sigma2p=0.45, tamponi=None, scenario=None,
                       out_type='h5'):
        # initialize compartments
        # Y0: Nc x Ns
        #self.S, self.U, self.I, self.H, self.T, self.E,\
        #        self.R, self.V1, self.V2, self.V2p = Y0
        self.Y0 = Y0
        # Y: Nc x T x Ns
        self.Y = np.zeros((len(Y0), len(t_list)))
        #initialize parameters
        self.params = params
        self.Ns = Pop.size
        #initialize vaccines effectiveness parameters
        self.sigma1, self.sigma2, self.sigma2p = sigma1, sigma2, sigma2p
        self.h1 = 0.19/sigma1 
        self.h2 = 0.05/sigma2 
        self.t1 = 0.12/0.19   
        self.t2 = 0.03/0.05   
        self.m1 = 0.21/sigma1 
        self.m2 = 0.042/sigma2

        self.t_list = t_list
        self.DPC_start = DPC_start
        self.DPC_end = DPC_end
        self.scenario = scenario

        self.data = data

        self.inYellow = False
        self.inOrange = False
        self.inRed = False
        self.timeNPI = 0
        self.adapNPI = 5

        self.maxV = maxV
        self.dV1vec = vaccines['prima_dose']
        self.dV2vec = vaccines['seconda_dose']
        self.dV3vec = vaccines['terza_dose']

        self.variant_prevalence = 0 
        self.variant_prevalence_hosp = 0
        self.sigma1v = self.sigma2v = self.sigma2pv = 0
        self.variant_factor = self.kappa1 = self.kappa2 = 0

        self.R_d = np.zeros((t_list[-1]+1, self.Ns))
        self.Sfrac = np.zeros((t_list[-1]+1, self.Ns))
        
        self.Pop = Pop

        self.by_age = by_age 
        self.codes = codes

        self.tamponi = tamponi

        self.out_type = out_type
        self.out_path = out_path

    def initialize_variant(self, variant, variant_prevalence):
        self.variant_prevalence = variant_prevalence
        self.variant_prevalence_hosp = 0.25
        self.variant_factor = variant['factor']
        self.kappa1 = variant['kappa1']
        self.kappa2 = variant['kappa2']
        self.kappa2p = variant['kappa2p']
        self.xi = variant['xi']
        self.sigma1v = 1 - self.kappa1 + self.kappa1 * self.sigma1
        self.sigma2v = 1 - self.kappa2 + self.kappa2 * self.sigma2
        self.sigma2pv = 1 - self.kappa2p + self.kappa2p * self.sigma2p
        return

    def model(self, t, y0):
        t_int  = int(np.floor(t))
        beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_H,gamma_I,theta_T = self.params.params_time[t_int]
        S, Ub, Uv, I, H, T, E, R, V1, V2, V2p = y0

        beta_Ub = beta_U / (1+(self.variant_factor-1)*self.variant_prevalence)
        beta_Uv = self.variant_factor * beta_Ub 

        rho_U *= 1 - 8 * delta

        dV1 = self.dV1vec[t_int]
        dV2 = self.dV2vec[t_int]
        dV3 = self.dV3vec[t_int]

        #t_vac = t_int -1 if t>=1 else t_int
        #print(self.Y[0,t_vac],self.Y[-4,t_vac],self.R_d[t_int])
        #dV1S = dV1 * self.Y[0,t_vac] / (self.Y[0,t_vac] + self.Y[-4,t_vac] - self.R_d[t_int])
        #dV2S = dV2 * self.Y[0,t_vac] / (self.Y[0,t_vac] + self.Y[-4,t_vac] - self.R_d[t_int])
        #dV2pS =self.dV2vec[t_vac-150] *  self.Y[0,t_vac-150]/ (self.Y[0,t_vac-150] + self.Y[-4,t_vac-150] - self.R_d[t_int-150]) if t>150 else 0
        #dV3S = dV3 * self.Y[0,t_vac] / (self.Y[0,t_vac] + self.Y[-4,t_vac] - self.R_d[t_int])
        dV1S = dV1 * self.Sfrac[t_int]
        dV2S = dV2 * self.Sfrac[t_int]
        dV2pS =self.dV2vec[t_int-150] *  self.Sfrac[t_int-150] if t>=150 else 0
        dV3S = dV3 * self.Sfrac[t_int] 
        #if self.t_list[0]>0:
        #    dV3S *= 2

        if self.t_list[0] > 0:
            totV1 = self.dV1vec[:t_int+1].sum()
            totV2 = self.dV2vec[:t_int+1].sum()
            totV1ini = self.dV1vec[:int(self.params.dataEnd)+1].sum()
            totV2ini = self.dV2vec[:int(self.params.dataEnd)+1].sum()

            popS = (self.maxV - totV1)/self.maxV
            popV1 = (totV1-totV2)/self.maxV
            popV2 = totV2/self.maxV

            popSini = (self.maxV - totV1ini)/self.maxV
            popV1ini = (totV1ini-totV2ini)/self.maxV
            popV2ini = totV2ini/self.maxV

            cases = popS + self.sigma1*popV1 + self.sigma2*popV2
            casesS = popS/cases
            casesV1 = self.sigma1*popV1/cases
            casesV2 = self.sigma2*popV2/cases

            casesini = popSini + self.sigma1*popV1ini + self.sigma2*popV2ini
            casesSini = popSini/casesini
            casesV1ini = self.sigma1*popV1ini/casesini
            casesV2ini = self.sigma2*popV2ini/casesini

        tamponi = self.tamponi[t_int-1] + self.tamponi[t_int]
        S_non_vaccinabili = (self.Pop - self.maxV) * self.Sfrac[t_int]
        S_vaccinabili = S - S_non_vaccinabili
        S_gp = tamponi * self.Sfrac[t_int]
        S_no_gp = S_vaccinabili - S_gp

        # attuale
        beta_gp =       0.0404
        beta_test =     0.0326
        beta_novax =    0.0197
                              
        # gialla              
        beta_gp_y =     0.036 #0.0404
        beta_test_y =   0.029 #0.0318
        beta_novax_y =  0.017 #0.0195
                              
        # arancione           
        beta_gp_a =     0.032#0.0399
        beta_test_a =   0.026#0.0296
        beta_novax_a =  0.015#0.0189
                              
        # rossa               
        beta_gp_r =     0.0159
        beta_test_r =   0.0149
        beta_novax_r =  0.0079

        vax = V1 + V2 + V2p
        betaV_now = betaV_new = beta_gp
        betaS_now = (beta_gp * (S - S_no_gp) + beta_novax * S_no_gp) / S
        betaS_new = (beta_test * S_gp + beta_gp * S_non_vaccinabili + beta_novax * S_no_gp) / S
        betaS_now_y = (beta_gp * (S - S_no_gp) + beta_novax_y * S_no_gp) / S
        betaS_now_a = (beta_gp * (S - S_no_gp) + beta_novax_a * S_no_gp) / S
        betaS_now_r = (beta_gp_r * (S - S_no_gp) + beta_novax_r * S_no_gp) / S
        betaS_new_y = (beta_test_y * S_gp + beta_gp * S_non_vaccinabili + beta_novax_y * S_no_gp) / S
        betaS_new_a = (beta_test_a * S_gp + beta_gp * S_non_vaccinabili + beta_novax_a * S_no_gp) / S
        betaS_new_r = (beta_test_r * S_gp + beta_gp_r * S_non_vaccinabili + beta_novax_r * S_no_gp) / S

        StoUb = S * beta_Ub * Ub / self.Pop
        StoUv = S * beta_Uv * Uv / self.Pop
        V1toUb = self.sigma1 * V1 * beta_Ub * Ub / self.Pop
        V1toUv = self.sigma1v * V1 * beta_Uv * Uv / self.Pop
        V2toUb = self.sigma2 * V2 * beta_Ub * Ub / self.Pop
        V2toUv = self.sigma2v * V2 * beta_Uv * Uv / self.Pop
        V2ptoUb = self.sigma2p * V2p * beta_Ub * Ub / self.Pop
        V2ptoUv = self.sigma2pv * V2p * beta_Uv * Uv / self.Pop

        maxH = 57705
        maxT = 9044

        U = Ub + Uv
        current_variant_prevalence = Uv/U

        tauratioS = 1
        tauratio = 1

        if self.t_list[0]>0:
            # Nessuno screnario 
            if self.scenario==None:
                pass
            # Screnario giallo
            elif self.scenario=='Yellow':
                if t - self.t_list[0] > 3:
                    if self.inYellow == False:
                        self.inRed =False 
                        self.inOrange = False
                        self.inYellow = True
                        self.timeNPI = t
                    if t-self.timeNPI > self.adapNPI:
                        tauratioS = betaS_new_y / betaS_new
                        tauratio  = beta_gp_y / betaV_new
                    else:
                        tauratioS = (t-self.timeNPI)/self.adapNPI*betaS_new_y / betaS_new + (1-(t-self.timeNPI)/self.adapNPI)*1
                        tauratio  = (t-self.timeNPI)/self.adapNPI*beta_gp_y / betaV_new + (1-(t-self.timeNPI)/self.adapNPI)*1
            # Screnario arancione
            elif self.scenario=='Orange':
                if t - self.t_list[0] > 3:
                    if self.inOrange == False:
                        self.inRed = False
                        self.inOrange = True
                        self.inYellow = False
                        self.timeNPI = t
                    if t-self.timeNPI > self.adapNPI:
                        tauratioS = betaS_new_a / betaS_new
                        tauratio  = beta_gp_a / betaV_new
                    else:
                        tauratioS = (t-self.timeNPI)/self.adapNPI*betaS_new_y / betaS_new + (1-(t-self.timeNPI)/self.adapNPI)*betaS_new_y / betaS_new
                        tauratio  = (t-self.timeNPI)/self.adapNPI*beta_gp_y / betaV_new + (1-(t-self.timeNPI)/self.adapNPI)*beta_gp_y / betaV_new
            # Screnario rosso
            elif self.scenario=='Red':
                if t - self.t_list[0] > 3:
                    if self.inRed == False:
                        self.inRed = True
                        self.inOrange = False
                        self.inYellow = False
                        self.timeNPI = t
                    if t-self.timeNPI > self.adapNPI:
                        tauratioS = betaS_new_r / betaS_new
                        tauratio  = beta_gp_r / betaV_new
                    else:
                        tauratioS = (t-self.timeNPI)/self.adapNPI*betaS_new_r / betaS_new + (1-(t-self.timeNPI)/self.adapNPI)*1
                        tauratio  = (t-self.timeNPI)/self.adapNPI*beta_gp_r / betaV_new + (1-(t-self.timeNPI)/self.adapNPI)*1
            # Scenario Controllato
            elif self.scenario=='Controlled':
                if (delta * U > 250/1e5/7*self.Pop) and (H > 0.4*maxH) and (T > 0.3*maxT):
                    if self.inRed == False:
                        self.inRed = True
                        self.inOrange = False
                        self.inYellow = False
                        self.timeNPI = t
                    if t-self.timeNPI > self.adapNPI:
                        tauratioS = betaS_new_r / betaS_new
                        tauratio  = beta_gp_r / betaV_new
                    else:
                        tauratioS = (t-self.timeNPI)/self.adapNPI*betaS_new_r / betaS_new + (1-(t-self.timeNPI)/self.adapNPI)*betaS_new_a / betaS_new
                        tauratio  = (t-self.timeNPI)/self.adapNPI*beta_gp_r / betaV_new + (1-(t-self.timeNPI)/self.adapNPI)*beta_gp_a / betaV_new
                    print(t, ' - Rossa', self.inYellow, self.inOrange, self.inRed, self.timeNPI, tauratioS)
                elif (delta * U > 150/1e5/7*self.Pop) and (H > 0.3*maxH) and (T > 0.2*maxT):
                    if self.inOrange == False:
                        self.inRed = False
                        self.inOrange = True
                        self.inYellow = False
                        self.timeNPI = t
                    if t-self.timeNPI > self.adapNPI:
                        tauratioS = betaS_new_a / betaS_new
                        tauratio  = beta_gp_a / betaV_new
                    else:
                        tauratioS = (t-self.timeNPI)/self.adapNPI*betaS_new_y / betaS_new + (1-(t-self.timeNPI)/self.adapNPI)*betaS_new_y / betaS_new
                        tauratio  = (t-self.timeNPI)/self.adapNPI*beta_gp_y / betaV_new + (1-(t-self.timeNPI)/self.adapNPI)*beta_gp_y / betaV_new
                    print(t, ' - Arancione', self.inYellow, self.inOrange, self.inRed, self.timeNPI, tauratioS)
                elif (delta * U > 150/1e5/7*self.Pop) or ((delta * U > 50/1e5/7*self.Pop) and (H > 0.15*maxH) and (T > 0.1*maxT)):
                    if self.inYellow == False:
                        self.inRed = False
                        self.inOrange = False
                        self.inYellow = True
                        self.timeNPI = t
                    if t-self.timeNPI > self.adapNPI:
                        tauratioS = betaS_new_y / betaS_new
                        tauratio  = beta_gp_y / betaV_new
                    else:
                        tauratioS = (t-self.timeNPI)/self.adapNPI*betaS_new_y / betaS_new + (1-(t-self.timeNPI)/self.adapNPI)*1
                        tauratio  = (t-self.timeNPI)/self.adapNPI*beta_gp_y / betaV_new + (1-(t-self.timeNPI)/self.adapNPI)*1
                    print(t, ' - Gialla', self.inYellow, self.inOrange, self.inRed, self.timeNPI, tauratioS)
            StoUb *=   tauratioS
            StoUv *=   tauratioS
            V1toUb *=  tauratio
            V1toUv *=  tauratio
            V2toUb *=  tauratio
            V2toUv *=  tauratio
            V2ptoUb *= tauratio
            V2ptoUv *= tauratio

        UbtoI = delta * Ub
        UvtoI = delta * Uv
        UbtoR = rho_U * Ub
        UvtoR = rho_U * Uv
        ItoH = omega_I * I * ((casesS + self.h1 * casesV1 + self.h2 * casesV2)/(casesSini + self.h1 * casesV1ini + self.h2 * casesV2ini) *
                              (1 - self.xi * current_variant_prevalence)/(1-self.xi*self.variant_prevalence_hosp) if self.t_list[0] > 0 else 1)
        ItoR = rho_I*I
        ItoE = gamma_I * I * ((casesS + self.m1 * casesV1 + self.m2 * casesV2)/(casesSini + self.m1 * casesV1ini + self.m2 * casesV2ini) if self.t_list[0] > 0 else 1)
        HtoR = rho_H*H
        HtoT = omega_H * H * ((casesS + self.t1 * casesV1 + self.t2 * casesV2)/(casesSini + self.t1 * casesV1ini + self.t2 * casesV2ini) if self.t_list[0] > 0 else 1)
        HtoE = gamma_H * H
        TtoH = theta_T * T
        TtoE = gamma_T * T

        RtoNull = min(dV1 - dV1S, R - UbtoR - UvtoR - ItoR - HtoR)
        StoV1 = min(dV1S, S - StoUb - StoUv)
        V1toV2 = min(dV2S,  V1 - V1toUb - V1toUv)
        V2toV2p = min(dV2pS,V2 - V2toUb - V2toUv)
        V2ptoV2 = min(dV3S, V2p - V2ptoUb - V2ptoUv)
        
        dSdt = - StoUb - StoUv - StoV1
        dUbdt = StoUb + V1toUb + V2toUb + V2ptoUb - UbtoI - UbtoR
        dUvdt = StoUv + V1toUv + V2toUv + V2ptoUv - UvtoI - UvtoR
        dIdt = UbtoI + UvtoI - ItoH - ItoR - ItoE
        dHdt = ItoH - HtoR - HtoT - HtoE + TtoH
        dTdt = HtoT - TtoE - TtoH
        dEdt = TtoE + HtoE + ItoE
        dRdt = UbtoR + UvtoR + ItoR + HtoR - RtoNull
        dV1dt = StoV1 - V1toUb - V1toUv - V1toV2
        dV2dt = - V2toUb - V2toUv + V1toV2 - V2toV2p + V2ptoV2
        dV2pdt = V2toV2p - V2ptoV2 - V2ptoUb - V2ptoUv

        return np.vstack((dSdt, dUbdt, dUvdt, dIdt, dHdt, dTdt, dEdt, dRdt, dV1dt, dV2dt, dV2pdt)).squeeze()

                
    def solve(self):
        t_start = int(self.t_list[0])
        self.params.compute_param_over_time(int(self.t_list[-1]))
        self.Y[:,t_start] = self.Y0 
        self.R_d[t_start] = self.data.Recovered.iloc[t_start]
        self.Sfrac[t_start] = self.Y0[0] / (self.Y0[0] + self.Y0[-4] - self.R_d[t_start])
        for i,t in enumerate(self.t_list[:-1]):
            y0 = self.Y[...,i+t_start]
            self.R_d[t_start + i + 1] = self.R_d[t_start + i] + (y0[-8:-6] * self.params.params_time[t_start + i + 1,6:8]).sum()
            self.Sfrac[t_start + i + 1] = y0[0] / (y0[0] + y0[-4] - self.R_d[t_start+i])
            k1=self.model(t      , y0     )
            k2=self.model(t+0.5, y0+0.5*k1)
            k3=self.model(t+0.5, y0+0.5*k2)
            k4=self.model(t+1  , y0+    k3)
            self.Y[...,t_start+i+1] = y0 + (k1+2*(k2+k3)+k4)/6.0
        return

    def estimate(self):
        params0 = pm.maskParams( self.params.get() , self.params.getMask() )
        lower_b = pm.maskParams( self.params.getLowerBounds() , self.params.getMask() )
        upper_b = pm.maskParams( self.params.getUpperBounds() , self.params.getMask() )
        bounds = Bounds( lower_b, upper_b )
        
        #local mimimization
        result = minimize_parallel(self.error, params0, bounds=bounds,\
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
        _, Ub, Uv, I, H, T, E, R, _, _, _ = self.Y
        U = Ub + Uv

        dE = np.diff(E, not (self.Ns - 1), prepend=E[0]-self.data['Daily_extinct'].iloc[0])
        # compute errors
        # Flatten the solution arrays to match data format
        # Daily Extinct and New positives data are already smoothed with a weekly rolling mean
        errorI = I.flatten('F') - self.data['Isolated'].values
        errorH = H.flatten('F') - self.data['Hospitalized'].values
        errorT = T.flatten('F') - self.data['Threatened'].values
        errorR = self.R_d.flatten('F') - self.data['Recovered'].values
        errorE = dE - self.data['Daily_extinct'].values
        errorNP = (self.params.params_time[self.t_list,2] * U).flatten('F') - self.data['New_positives'].rolling(window=7,min_periods=1,center=True).mean().values

        # compute errors weights
        one = np.ones(len(errorI))
        weight = np.ones(len(errorI)) # modify this if you want to weight more specific time steps
        weightsNP = weight/np.maximum(self.data['New_positives'].rolling(window=7,min_periods=1,center=True).mean().values,one)
        weightsI = weight/np.maximum(self.data['Isolated'].values,one)
        weightsH = weight/np.maximum(self.data['Hospitalized'].values,one)
        weightsT = weight/np.maximum(self.data['Threatened'].values,one)
        weightsE = weight/np.maximum(self.data['Daily_extinct'].values,one)
        weightsR = 0.1*weight/self.data['Recovered'].max()

        errorL2 = ((errorI ** 2)*weightsI + 
                   (errorH ** 2)*weightsH +
                   (errorT ** 2)*weightsT +
                   (errorNP ** 2)*weightsNP +
                   (errorR ** 2)*weightsR +
                   (errorE ** 2)*weightsE).sum() 

        return np.sqrt(errorL2)

    def postprocessRd(self):
        rho_I, rho_H, rho_T = self.params.params_time[:, 6:9].transpose()
        I, H, T = self.Y[-8:-5]
        return rho_I * I + rho_H * H + rho_T * T 

    def computeRt(self):
        nPhases = self.params.nPhases
        nSites = self.params.nSites
        self.Rt = np.zeros((len(self.t_list), nSites)).squeeze()
        for i,t in enumerate(self.t_list):
            beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
                rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = self.params.params_time[t]
            rho_U *= 1 - 8*delta
            r1 = delta + rho_U
            R0_tmp = beta_U / r1
            self.Rt[i] = R0_tmp * (self.Y[0] + self.Y[-3] * self.sigma1 + self.Y[-2] * self.sigma2 + self.Y[-1] * self.sigma2p) / self.Pop
        np.savetxt(self.out_path+'/Rt.csv', self.Rt, delimiter=',')
        return

    def save(self):
        print('Reorganizing and saving results...')
        # Sum undetected from base and variant
        self.Y[1] += self.Y[2]
        self.Y = np.delete(self.Y, 2, 0)
        self.Y = self.Y[:,self.t_list[0]:]

        Nc = self.Y.shape[0]
        dates = pd.date_range(start=self.DPC_start + pd.Timedelta(self.t_list[0], 'days'), periods=self.t_list[-1]-self.t_list[0]+1)
        codes = np.tile(self.codes, len(dates))
        times = np.repeat(self.t_list, len(self.codes))

        results = np.zeros((6+Nc, (len(times))), dtype='O')
        
        results[:3] = codes, dates, times
        results[3:3+Nc] = self.Y.reshape(Nc, len(times))
        results[3+Nc] = self.R_d[self.t_list[0]].flatten() 
        results[4+Nc,0] = self.data[self.data['time'] == self.t_list[0]]['New_positives'].values
        results[4+Nc,1:] = (self.Y[1,:-1] * self.params.params_time[self.t_list[0]+1:,2]).flatten() 
        results[5+Nc,0] = self.data[self.data['time'] == self.t_list[0]]['New_threatened'].values
        results[5+Nc,1:] = (self.Y[3,:-1] * self.params.params_time[self.t_list[0]:-1,4]).flatten() 
        results_df = pd.DataFrame(results.T,columns=['Geocode','date','time','Suscept','Undetected','Isolated',
            'Hospitalized','Threatened','Extinct','Recovered','First_dose','Second_dose','Second_dose_plus','Recovered_detected','New_positives','New_threatened'])
        
        Code = "Age" if self.by_age else "Geocode"
        if not self.by_age:
            results_df = results_df.astype({Code: int,"time": 'float64'})
        else:
            results_df = results_df.astype({Code: str,"time": 'float64'})

        results_df = results_df.sort_values(by=[Code,'date'])
        results_df = results_df.astype(dict(zip(['Suscept','Undetected','Isolated','Hospitalized','Threatened','Extinct','Recovered','First_dose','Second_dose','Second_dose_plus','Recovered_detected','New_positives','New_threatened'],['float64']*13)))

    
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
