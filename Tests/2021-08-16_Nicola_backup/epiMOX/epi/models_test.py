import numpy as np

def SUIHTERmodel(t,y,params,Pop,DO,map_to_prov,dV1vec,dV2vec):
    beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.atTime(t).dot(map_to_prov.transpose())
    
    sigma1 = 0.375
    sigma2 = 0.2

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
   
   # if t>=132:
     #   beta_U*=2
        #sigma1=sigma2=1
    delta = params.delta(t)
    rho_U *= 1 - 8*delta
    omega_I = np.clip(params.omegaI(t),0,1)
    omega_H = np.clip(params.omegaH(t),0,1)
    #gamma_I = (1-theta_H)*np.clip(params.gammaH(t)/H,0,1)
    #theta_H *= np.clip(params.gammaH(t)/I,0,1)
    #gamma_T = np.clip(params.gammaT(t),0,1)
    #rho_I = np.clip(params.rhoI(t)-theta_H,0,1)
    #rho_H = np.clip(params.rhoH(t)+theta_H*I/H,0,1)
    #theta_T *= 1 - (T/8799)*0.55+0.25
    #gamma_I = np.clip(params.gammaH(t)+theta_T*T/H,0,1)
    #gamma_T = np.clip(params.gammaT(t)-theta_T,0,1)
    #theta_T = np.clip(params.thetaT(t),0,1)

    dV1 = dV1vec[int(np.floor(t))]
    dV2 = dV2vec[int(np.floor(t))]
    dV1S = dV1 * S / (S+R)
   
    dSdt = -S * beta_U * U / Pop  - dV1S
    dUdt = (S + sigma1 * V1 + sigma2 * V2) * beta_U * U / Pop  - (delta + rho_U) * U #+ (beta_U*U*R/Pop)*(t>=132)
    dIdt = delta * U - (omega_I + rho_I +theta_H) * I# + theta_H * H
    dHdt = omega_I * I - (rho_H + omega_H + gamma_I) * H + theta_T * T
    dTdt = omega_H * H - (rho_T + gamma_T + theta_T) * T
    dEdt = gamma_T * T + gamma_I * H + theta_H *I 
    dRdt = rho_U * U + rho_I * I + rho_H * H + rho_T * T # - dV1 + dV1S #- (beta_U*U*R/Pop)*(t>=132)
    dV1dt = dV1S - dV2 - sigma1 * V1 * beta_U * U / Pop 
    dV2dt = dV2 - sigma2 * V2 * beta_U * U / Pop
    return np.concatenate((dSdt, dUdt, dIdt, dHdt, dTdt, dEdt, dRdt, dV1dt, dV2dt), axis=None)

def SUIHTERmodel_vaccines(t,y,params,Pop,DO,map_to_prov,dV1vec,dV2vec,variant_perc,variant_factor,kappa1,kappa2):
    beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.atTime(t)
    

    beta_U *= 1.5
    sigma1 = 0.375
    sigma2 = 0.2
    
    sigma1v = 1 - kappa1 + kappa1 * sigma1
    sigma2v = 1 - kappa2 + kappa2 * sigma2

    alpha1 = 0.7
    alpha2 = 0.4

    h1 = 0.2
    h2 = 0.04

    m1 = 0.2
    m2 = 0.025    

    Ns = Pop.shape[0]
    S = y[0:Ns]
    U_b = y[Ns:2*Ns]
    U_v1 = y[2*Ns:3*Ns]
    U_v2 = y[3*Ns:4*Ns]
    U = U_b + U_v1 + U_v2
    I_b = y[4*Ns:5*Ns]
    I_v1 = y[5*Ns:6*Ns]
    I_v2 = y[6*Ns:7*Ns]
    I = I_b + I_v1 + I_v2
    H = y[7*Ns:8*Ns]
    T = y[8*Ns:9*Ns]
    E = y[9*Ns:10*Ns]
    R = y[10*Ns:11:Ns]
    V1 = y[11*Ns:12:Ns]
    V2 = y[12*Ns:]
    
    delta = params.delta(t)
    rho_U *= 1 - 8*delta
    omega_I = np.clip(params.omegaI(t),0,1)*params.omegaI_vaccines(t)
    omega_H = np.clip(params.omegaH(t),0,1)

    dV1 = dV1vec[int(np.floor(t))]
    dV2 = dV2vec[int(np.floor(t))]
    dV1S = dV1 * S/(S+R)
    dV1R = dV1 - dV1S
    
    dSdt = -S * beta_U * (U_b + alpha1* U_v1 + alpha2 * U_v2) / Pop - np.min((dV1S,S-S * beta_U * (U_b + alpha1* U_v1 + alpha2 * U_v2) / Pop))
    dUbdt = S * beta_U * (U_b + alpha1* U_v1 + alpha2 * U_v2) / Pop - (delta + rho_U) * U_b
    dUv1dt = V1 * beta_U * sigma1v * (U_b + alpha1* U_v1 + alpha2 * U_v2) / Pop - (delta + rho_U) * U_v1
    dUv2dt = V2 * beta_U * sigma2v * (U_b + alpha1* U_v1 + alpha2 * U_v2) / Pop - (delta + rho_U) * U_v2
    dIbdt = delta * U_b - (omega_I + rho_I + theta_H) * I_b
    dIv1dt = delta * U_v1 - (h1 * omega_I + rho_I + m1 * theta_H) * I_v1
    dIv2dt = delta * U_v2 - (h2 * omega_I + rho_I + m2 * theta_H) * I_v2
    dHdt = omega_I * (I_b + h1 * I_v1 + h2 * I_v2) - (rho_H + omega_H + gamma_I) * H + theta_T * T
    dTdt = omega_H * H - (rho_T + gamma_T + theta_T) * T
    dEdt = gamma_T * T + gamma_I * H + theta_H * (I_b + m1 * I_v1 + m2 * I_v2)
    dRdt = rho_U * U + rho_I * I + rho_H * H - np.min((dV1R,R))
    dV1dt = dV1 - dV2 - V1 * beta_U * sigma1v * (U_b + alpha1* U_v1 + alpha2 * U_v2) / Pop
    dV2dt = dV2 - V2 * beta_U * sigma2v * (U_b + alpha1* U_v1 + alpha2 * U_v2) / Pop

    return np.concatenate((dSdt, dUbdt, dUv1dt, dUv2dt, dIbdt, dIv1dt, dIv2dt, dHdt, dTdt, dEdt, dRdt, dV1dt, dV2dt), axis=None)

def SUIHTERmodel_variant(t,y,params,Pop,DO,map_to_prov,dV1vec,dV2vec,variant_perc,variant_factor,kappa1,kappa2):
    beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_H,gamma_I,theta_T = params.atTime(t)

    sigma1 = 0.3 #0.375   
    sigma2 = 0.12 #0.2     
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
    
    delta = params.delta(t)
    rho_U *= 1 - 8*delta
    omega_I = np.clip(params.omegaI(t),0,1)#*params.omegaI_vaccines(t)
    omega_H = np.clip(params.omegaH(t),0,1)

    #dV1R = dV1 - dV1S
    
    #if delta * U > 250/1e5/7*Pop:
    #    beta_U *= 0.61
    #elif delta * U > 150/1e5/7*Pop:
    #    beta_U *= 0.77
    #elif delta * U > 50/1e5/7*Pop:
    #    beta_U *= 0.87

    beta_Ubase = beta_U / (1+(variant_factor-1)*variant_perc)

    beta_Uvariant = variant_factor * beta_Ubase

    dV1 = dV1vec[int(np.floor(t))]
    dV2 = dV2vec[int(np.floor(t))]
    dV1S = dV1 * S/(S+R)
    totV1 = dV1vec[:int(np.floor(t))-20].sum()
    totV2 = dV2vec[:int(np.floor(t))-20].sum()
    totV1ini = dV1vec[:int(params.dataEnd)-20].sum()
    totV2ini = dV2vec[:int(params.dataEnd)-20].sum()

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

    #print(casesS, casesV1, casesV2, casesSini, casesV1ini, casesV2ini, (casesS + h1 * casesV1 + h2 * casesV2)/(casesSini + h1 * casesV1ini + h2 * casesV2ini),H/I)

    #S0 = (Pop - 54009901) *(1-R/Pop)
    #Seff = max(S-S0,0)

    StoV1 = np.min((dV1S,S-S * (beta_Ubase * U_base + beta_Uvariant * U_variant) / Pop))
    StoUb = S * beta_Ubase * U_base / Pop
    StoUv = S * beta_Uvariant * U_variant / Pop
    V1toUb = sigma1 * V1 * beta_Ubase * U_base / Pop
    V1toUv = sigma1v * V1 * beta_Uvariant * U_variant / Pop
    V2toUb = sigma2 * V2 * beta_Ubase * U_base / Pop
    V2toUv = sigma2v * V2 * beta_Uvariant * U_variant / Pop
    #SefftoUb = Seff * beta_Ubase * U_base / Pop
    #SefftoUv = Seff * beta_Uvariant * U_variant / Pop
    #NUtot = SefftoUb + SefftoUv + V1toUb + V1toUv + V2toUb + V2toUv
    ##NUtot = StoUb + StoUv + V1toUb + V1toUv + V2toUb + V2toUv

    #US_b  = SefftoUb/NUtot
    #UV1_b = V1toUb/NUtot
    #UV2_b = V2toUb/NUtot

    #US_v  = SefftoUv/NUtot
    #UV1_v = V1toUv/NUtot
    #UV2_v = V2toUv/NUtot

    #omI = omega_I*(US_b + 2*US_v + h1 * (UV1_b + 2*UV1_v) + h2 * (UV2_b + 2*UV2_v))

    
    #if abs(np.mod(t,1)) < 1e-10:
    #    with open('frac.txt','a') as f:
    #        f.write(str(t)+'\t'+ str(US_b[0])+'\t'+ str(UV1_b[0])+'\t'+ str(UV2_b[0])+'\t'+ str(US_v[0])+'\t'+ str(UV1_v[0])+'\t'+ str(UV2_v[0]) + '\n')# '\t'+ str(omI[0]) + '\n')

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
    V1toV2 = min(dV2,V1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop)

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

    #dSdt = -S * (beta_Ubase * U_base + beta_Uvariant * U_variant) / Pop - np.min((dV1S,S-S * (beta_Ubase * U_base + beta_Uvariant * U_variant) / Pop))
    #dUbdt = (S + sigma1 * V1 + sigma2 * V2) * beta_Ubase * U_base / Pop - (delta + rho_U) * U_base
    #dUvdt = (S + sigma1v * V1 + sigma2v * V2 ) * beta_Uvariant * U_variant / Pop - (delta + rho_U) * U_variant
    #dIdt = delta * U - (omega_I * (US_b + US_v + h1 * (UV1_b + 1.6 * UV1_v) + h2 * (UV2_b + 1.6 * UV2_v)) + rho_I + gamma_I * (US_b + US_v + m1 * (UV1_b + UV1_v) + m2 * (UV2_b + UV2_v))) * I
    #dHdt = omega_I * (US_b +  US_v + h1 * (UV1_b + 1.6 * UV1_v) + h2 * (UV2_b + 1.6 * UV2_v)) * I - (rho_H + omega_H + gamma_H) * H + theta_T * T
    #dTdt = omega_H * H - (rho_T + gamma_T + theta_T) * T
    #dEdt = gamma_T * T + gamma_H * H + gamma_I * (US_b + US_v + m1 * (UV1_b + UV1_v) + m2 * (UV2_b + UV2_v)) * I
    #dRdt = rho_U * U + rho_I * I + rho_H * H #- np.min((dV1R,R)) #- R * beta_Uvariant * U_variant / Pop
    #dV1dt = np.min((dV1S,S-S * (beta_Ubase * U_base + beta_Uvariant * U_variant) / Pop)) \
    #        - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop -\
    #        min(dV2,V1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop)   
    #dV2dt = min(dV2,V1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop) 
    #- V2 * (sigma2 * beta_Ubase * U_base + sigma2v * beta_Uvariant * U_variant) / Pop
   # with open('flussi.txt','a') as f:
   #     f.write(str(omega_I * (US_b + 2 * US_v + h1 * (UV1_b + 2 * UV1_v) + h2 * (UV2_b + 2 * UV2_v)))  + '\t' +str(- (rho_H + omega_H + gamma_I) * H)+'\t' +str(theta_T * T)+'\t'+str(omega_I)+'\n')
        #f.write(str(US_b)+'\t'+str(US_v)+'\t'+str(UV1_b)+'\t'+str(UV1_v)+'\t'+str(UV2_b)+'\t'+str(UV2_v)+'\n')
    return np.concatenate((dSdt, dUbdt, dUvdt, dIdt, dHdt, dTdt, dEdt, dRdt, dV1dt, dV2dt), axis=None)


def SUIHTERmodel_variant_orig(t,y,params,Pop,DO,map_to_prov,dV1vec,dV2vec,variant_perc,variant_factor,kappa1,kappa2):
    beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.atTime(t)

    sigma1 = 0.375
    sigma2 = 0.2
    #kappa = 0.4  #efficacia vaccino contro variante rispetto a base
    
    sigma1v = 1 - kappa1 + kappa1 * sigma1
    sigma2v = 1 - kappa2 + kappa2 * sigma2
    
    alpha1 = 0.7
    alpha2 = 0.4

    h1 = 0.2
    h2 = 0.05

    m1 = 0.2
    m2 = 0.025    

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
    
    delta = params.delta(t)
    rho_U *= 1 - 8*delta
    omega_I = np.clip(params.omegaI(t),0,1)*params.omegaI_vaccines(t)
    omega_H = np.clip(params.omegaH(t),0,1)
    
    dV1 = dV1vec[int(np.floor(t))]
    dV2 = dV2vec[int(np.floor(t))]
    dV1S = dV1 * S/(S+R)
    dV1R = dV1 - dV1S
    
    #if delta * U > 250/1e5/7*Pop:
    #    beta_U *= 0.61
    #elif delta * U > 150/1e5/7*Pop:
    #    beta_U *= 0.77
    #elif delta * U > 50/1e5/7*Pop:
    #    beta_U *= 0.87

    beta_Ubase = beta_U / (1+(variant_factor-1)*variant_perc)

    beta_Uvariant = variant_factor * beta_Ubase

    S_tot_b = beta_Ubase * U_base * (S + sigma1 * V1 + sigma2 * V2)
    S_tot_v = beta_Uvariant * U_variant * (S + sigma1v * V1 + sigma2v * V2)

    US_b  = beta_Ubase * S * U_base/ (S_tot_b +S_tot_v)
    UV1_b = beta_Ubase * sigma1 * V1 * U_base / (S_tot_b + S_tot_v)
    UV2_b = beta_Ubase * sigma2 * V2 * U_base / (S_tot_b + S_tot_v)

    US_v  = beta_Uvariant * S * U_variant / (S_tot_b + S_tot_v)
    UV1_v = beta_Uvariant *sigma1v * V1 * U_variant / (S_tot_b + S_tot_v)
    UV2_v = beta_Uvariant *sigma2v * V2 * U_variant / (S_tot_b + S_tot_v)

    dSdt = -S * (beta_Ubase * U_base + beta_Uvariant * U_variant) / Pop - np.min((dV1S,S-S * (beta_Ubase * U_base + beta_Uvariant * U_variant) / Pop))
    dUbdt = (S + sigma1 * V1 + sigma2 * V2) * beta_Ubase * U_base / Pop - (delta + rho_U) * U_base
    dUvdt = (S + sigma1v * V1 + sigma2v * V2 ) * beta_Uvariant * U_variant / Pop - (delta + rho_U) * U_variant
    dIdt = delta * U - (omega_I * (US_b + US_v + h1 * (UV1_b + 1 * UV1_v) + h2 * (UV2_b + 1 * UV2_v)) + rho_I + theta_H * (US_b + US_v + m1 * (UV1_b + UV1_v) + m2 * (UV2_b + UV2_v))) * I
    dHdt = omega_I * (US_b +  US_v + h1 * (UV1_b + 1 * UV1_v) + h2 * (UV2_b + 1 * UV2_v)) * I - (rho_H + omega_H + gamma_I) * H + theta_T * T
    dTdt = omega_H * H - (rho_T + gamma_T + theta_T) * T
    dEdt = gamma_T * T + gamma_I * H + theta_H * (US_b + US_v + m1 * (UV1_b + UV1_v) + m2 * (UV2_b + UV2_v)) * I
    dRdt = rho_U * U + rho_I * I + rho_H * H - np.min((dV1R,R)) #- R * beta_Uvariant * U_variant / Pop
    dV1dt = dV1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop -\
    min(dV2,V1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop)   
    dV2dt = min(dV2,V1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop) 
    - V2 * (sigma2 * beta_Ubase * U_base + sigma2v * beta_Uvariant * U_variant) / Pop
   # with open('flussi.txt','a') as f:
   #     f.write(str(omega_I * (US_b + 2 * US_v + h1 * (UV1_b + 2 * UV1_v) + h2 * (UV2_b + 2 * UV2_v)))  + '\t' +str(- (rho_H + omega_H + gamma_I) * H)+'\t' +str(theta_T * T)+'\t'+str(omega_I)+'\n')
        #f.write(str(US_b)+'\t'+str(US_v)+'\t'+str(UV1_b)+'\t'+str(UV1_v)+'\t'+str(UV2_b)+'\t'+str(UV2_v)+'\n')
    return np.concatenate((dSdt, dUbdt, dUvdt, dIdt, dHdt, dTdt, dEdt, dRdt, dV1dt, dV2dt), axis=None)

def SUIHTERmodel_original(t,y,params,Pop,DO,map_to_prov):
    beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.atTime(t).dot(map_to_prov.transpose())
    
    beta_I *= beta_U

    PopIn = DO.sum(axis=1)
    Ns = Pop.shape[0]
    S = y[0:Ns]
    U = y[Ns:2*Ns]
    I = y[2*Ns:3*Ns]
    H = y[3*Ns:4*Ns]
    T = y[4*Ns:5*Ns]
    E = y[5*Ns:6*Ns]
    R = y[6*Ns:]
    
    dSdt = (-S * (beta_U * U + beta_I * I) - beta_U * S * (np.dot(DO, U / Pop))) / (Pop + PopIn)
    dUdt = ( S * (beta_U * U + beta_I * I) + beta_U * S * (np.dot(DO, U / Pop))) / (Pop + PopIn) - (delta + rho_U) * U
    dIdt = delta * U - (omega_I + rho_I + gamma_I) * I + theta_H * H
    dHdt = omega_I * I - (rho_H + omega_H + theta_H) * H + theta_T * T
    dTdt = omega_H * H - (rho_T + gamma_T + theta_T) * T
    dEdt = gamma_T * T + gamma_I * I 
    dRdt = rho_U * U + rho_I * I + rho_H * H + rho_T * T
    return np.concatenate((dSdt, dUdt, dIdt, dHdt, dTdt, dEdt, dRdt), axis=None)

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
