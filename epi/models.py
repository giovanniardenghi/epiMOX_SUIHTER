import numpy as np

def SUIHTERmodel(t,y,params,Pop,DO,map_to_prov,dV1vec,dV2vec,R_d,CFR=0):
    beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.params_time[int(np.floor(t))]
    
    sigma1 =0.3
    sigma2 =0.12 
    
    m1 = 0.21
    m2 = 0.042    

    Ns = Pop.shape[0]
    S = y[0:Ns]
    U = y[Ns:2*Ns]
    I = y[2*Ns:3*Ns]
    H = y[3*Ns:4*Ns]
    T = y[4*Ns:5*Ns]
    E = y[5*Ns:6*Ns]
    R = y[6*Ns:7:Ns]
    V1 = y[7*Ns:8:Ns]
    V2 = y[8*Ns:]
   
    rho_U *= 1 - 8*delta

    dV1 = dV1vec[int(np.floor(t))]
    dV2 = dV2vec[int(np.floor(t))]
    dV1S = dV1 * S / (S+R-R_d[int(np.floor(t))])
    dV2S = dV2 * S / (S+R-R_d[int(np.floor(t))])
       
    dSdt = -S * beta_U * U / Pop - np.min((dV1S,S-S * beta_U * U / Pop)) 
    dUdt = (S + sigma1 * V1 + sigma2 * V2) * beta_U * U / Pop  - (delta + rho_U) * U #+ (beta_U*U*R/Pop)*(t>=132)
    dIdt = delta * U - (omega_I + rho_I +theta_H) * I# + theta_H * H
    dHdt = omega_I * I - (rho_H + omega_H + gamma_I) * H + theta_T * T
    dTdt = omega_H * H - (rho_T + gamma_T + theta_T) * T
    dEdt = gamma_T * T + gamma_I * H + theta_H *I 
    dRdt = rho_U * U + rho_I * I + rho_H * H + rho_T * T # - dV1 + dV1S #- (beta_U*U*R/Pop)*(t>=132)
    dV1dt = - min(dV2S,V1 - V1 * sigma1 * beta_U * U/ Pop) - sigma1 * V1 * beta_U * U / Pop + np.min((dV1S,S-S * beta_U * U / Pop)) 
    dV2dt = min(dV2S,V1 - V1 * sigma1 * beta_U * U/ Pop) - sigma2 * V2 * beta_U * U / Pop
 
    return np.concatenate((dSdt, dUdt, dIdt, dHdt, dTdt, dEdt, dRdt, dV1dt, dV2dt), axis=None)

def SUIHTERmodel_variant(t,y,params,Pop,DO,map_to_prov,dV1vec,dV2vec,R_d,variant_perc,variant_factor,kappa1,kappa2):
    beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_H,gamma_I,theta_T = params.params_time[int(np.floor(t))]

    sigma1 = 0.3   
    sigma2 = 0.12     
    # https://assets.publishing.service.gov.uk/government/uploads/system/uploads/attachment_data/file/1000512/Vaccine_surveillance_report_-_week_27.pdf

    # PFIZER https://www.cdc.gov/mmwr/volumes/70/wr/mm7013e3.htm (new data with higher efficacy 81% and 91%

    sigma1v = 1 - kappa1 + kappa1 * sigma1
    sigma2v = 1 - kappa2 + kappa2 * sigma2

    alpha1 = 0.55
    alpha2 = 0.4

    h1 = 0.2
    h2 = 0.05

    t1 = 0.62
    t2 = 0.5

    m1 = 0.21
    m2 = 0.042    

    Ns = Pop.shape[0]
    S = y[0]#:Ns]
    U_base = y[Ns]#:2*Ns]
    U_variant = y[2]#*Ns:3*Ns]
    U = U_base + U_variant
    I = y[3]#*Ns:4*Ns]
    H = y[4]#*Ns:5*Ns]
    T = y[5]#*Ns:6*Ns]
    E = y[6]#*Ns:7*Ns]
    R = y[7]#*Ns:8*Ns]
    V1 = y[8]#*Ns:9*Ns]
    V2 = y[9]#*Ns:]
    
    rho_U *= 1 - 8*delta

    beta_Ubase = beta_U / (1+(variant_factor-1)*variant_perc)

    beta_Uvariant = variant_factor * beta_Ubase

    dV1 = dV1vec[int(np.floor(t))]
    dV2 = dV2vec[int(np.floor(t))]
    dV1S = dV1 * S / (S+R-R_d[int(np.floor(t))])
    dV2S = dV2 * S / (S+R-R_d[int(np.floor(t))])
    totV1 = dV1vec[:int(np.floor(t))].sum()
    totV2 = dV2vec[:int(np.floor(t))].sum()
    totV1ini = dV1vec[:int(params.dataEnd)].sum()
    totV2ini = dV2vec[:int(params.dataEnd)].sum()

    maxV = 54009901

    popS = (maxV - totV1)/maxV
    popV1 = (totV1-totV2)/maxV
    popV2 = totV2/maxV

    popSini = (maxV - totV1ini)/maxV
    popV1ini = (totV1ini-totV2ini)/maxV
    popV2ini = totV2ini/maxV

    cases = popS + sigma1*popV1 + sigma2*popV2 
    casesS = popS/cases
    casesV1 = sigma1*popV1/cases
    casesV2 = sigma2*popV2/cases

    casesini = popSini + sigma1*popV1ini + sigma2*popV2ini 
    casesSini = popSini/casesini
    casesV1ini = sigma1*popV1ini/casesini
    casesV2ini = sigma2*popV2ini/casesini

    StoV1 = np.min((dV1S,S-S * (beta_Ubase * U_base + beta_Uvariant * U_variant) / Pop))
    StoUb = S * beta_Ubase * U_base / Pop
    StoUv = S * beta_Uvariant * U_variant / Pop
    V1toUb = sigma1 * V1 * beta_Ubase * U_base / Pop
    V1toUv = sigma1v * V1 * beta_Uvariant * U_variant / Pop
    V2toUb = sigma2 * V2 * beta_Ubase * U_base / Pop
    V2toUv = sigma2v * V2 * beta_Uvariant * U_variant / Pop

    UbtoI = delta * U_base
    UvtoI = delta * U_variant
    UbtoR = rho_U * U_base
    UvtoR = rho_U * U_variant
    ItoH = omega_I * (casesS + h1 * casesV1 + h2 * casesV2)/(casesSini + h1 * casesV1ini + h2 * casesV2ini) * I
    ItoR = rho_I*I
    ItoE = gamma_I * (casesS + m1 * casesV1 + m2 * casesV2)/(casesSini + m1 * casesV1ini + m2 * casesV2ini) * I
    HtoR = rho_H*H
    HtoT = omega_H * (casesS + t1 * casesV1 + t2 * casesV2)/(casesSini + t1 * casesV1ini + t2 * casesV2ini) * H
    HtoE = gamma_H * H 
    TtoH = theta_T * T
    TtoE = gamma_T * T
    V1toV2 = min(dV2S,V1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop)

    dSdt = - StoUb - StoUv - StoV1
    dUbdt = StoUb + V1toUb + V2toUb - UbtoI - UbtoR
    dUvdt = StoUv + V1toUv + V2toUv - UvtoI - UvtoR
    dIdt = UbtoI + UvtoI - ItoH - ItoR - ItoE 
    dHdt = ItoH - HtoR - HtoT - HtoE + TtoH
    dTdt = HtoT - TtoE - TtoH
    dEdt = TtoE + HtoE + ItoE
    dRdt = UbtoR + UvtoR + ItoR + HtoR #- np.min((dV1R,R)) #- R * beta_Uvariant * U_variant / Pop
    dV1dt = StoV1 - V1toUb - V1toUv - V1toV2
    dV2dt = - V2toUb - V2toUv + V1toV2

    return np.concatenate((dSdt, dUbdt, dUvdt, dIdt, dHdt, dTdt, dEdt, dRdt, dV1dt, dV2dt), axis=None)


def SUIHTERmodel_age(t,y,params,Pop,DO,map_to_prov):
    beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.atTime(t).dot(map_to_prov.transpose())
    
    beta_I *= beta_U

    Ns = Pop.shape[0]
    S = y[0:Ns]
    U = y[Ns:2*Ns]
    I = y[2*Ns:3*Ns]
    H = y[3*Ns:4*Ns]
    T = y[4*Ns:5*Ns]
    E = y[5*Ns:6*Ns]
    R = y[6*Ns:]
    
    dSdt = - S * ( beta_U  * np.dot(DO, U / Pop) + beta_I  * np.dot(DO, I / Pop))
    dUdt =   S * ( beta_U  * np.dot(DO, U / Pop) + beta_I  * np.dot(DO, I / Pop)) - (delta + rho_U) * U
    dIdt = delta * U - (omega_I + rho_I ) * I + theta_H * H
    dHdt = omega_I * I - (rho_H + omega_H + theta_H + gamma_I) * H + theta_T * T
    dTdt = omega_H * H - (rho_T + gamma_T + theta_T) * T
    dEdt = gamma_T * T + gamma_I * H 
    dRdt = rho_U * U + rho_I * I + rho_H * H + rho_T * T
    return np.concatenate((dSdt, dUdt, dIdt, dHdt, dTdt, dEdt, dRdt), axis=None)

def SEIRDmodel(t,y,params,Pop,DO,map_to_prov):
    beta, alpha, gamma, f = params.atTime(t).dot(map_to_prov.transpose())
    PopIn = DO.sum(axis=1)
    Ns = Pop.shape[0]
    S = y[0:Ns]
    E = y[Ns:2*Ns]
    I = y[2*Ns:3*Ns]
    R = y[3*Ns:4*Ns]
    D = y[4*Ns:]
    dSdt = - (beta * S * (I + np.dot(DO,I/Pop)))/(Pop+PopIn)
    dEdt = (beta * S * (I + np.dot(DO,I/Pop)))/(Pop+PopIn) - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = (1-f) * gamma * I
    dDdt = f * gamma * I
    return np.concatenate((dSdt, dEdt, dIdt, dRdt, dDdt), axis=None)
