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
    dRdt = rho_U * U + rho_I * I + rho_H * H + rho_T * T - dV1 + dV1S #- (beta_U*U*R/Pop)*(t>=132)
    dV1dt = dV1 - dV2 - sigma1 * V1 * beta_U * U / Pop 
    dV2dt = dV2 - sigma2 * V2 * beta_U * U / Pop
    return np.concatenate((dSdt, dUdt, dIdt, dHdt, dTdt, dEdt, dRdt, dV1dt, dV2dt), axis=None)


def SUIHTERmodel_variant(t,y,params,Pop,DO,map_to_prov,dV1vec,dV2vec,variant_perc,variant_factor,kappa1,kappa2):
    beta_U,beta_I,delta,omega_I,omega_H,rho_U,\
        rho_I,rho_H,rho_T,gamma_T,gamma_I,theta_H,theta_T = params.atTime(t)

    sigma1 = 0.375
    sigma2 = 0.2
    #kappa = 0.4  #efficacia vaccino contro variante rispetto a base
    
    sigma1v = 1 - kappa1 + kappa1 * sigma1
    sigma2v = 1 - kappa2 + kappa2 * sigma2

    beta_Ubase = beta_U / (1+(variant_factor-1)*variant_perc)

    beta_Uvariant = variant_factor * beta_Ubase

    Ns = Pop.shape[0]
    S = y[0:Ns]
    U_base = y[Ns:2*Ns]
    U_variant = y[2*Ns:3*Ns]
    U = U_base + U_variant
    I_base = y[3*Ns:4*Ns]
    I_variant = y[4*Ns:5*Ns]
    I = I_base + I_variant
    H = y[5*Ns:6*Ns]
    T = y[6*Ns:7*Ns]
    E = y[7*Ns:8*Ns]
    R = y[8*Ns:9*Ns]
    V1 = y[9*Ns:10*Ns]
    V2 = y[10*Ns:]
    
    delta = params.delta(t)
    rho_U *= 1 - 8*delta
    omega_I = np.clip(params.omegaI(t),0,1)*params.omegaI_vaccines(t)
    omega_H = np.clip(params.omegaH(t),0,1)
    
    dV1 = dV1vec[int(np.floor(t))]
    dV2 = dV2vec[int(np.floor(t))]
    dV1S = dV1 * S/(S+R)
    dV1R = dV1 - dV1S
    
    dSdt = -S * (beta_Ubase * U_base + beta_Uvariant * U_variant) / Pop - np.min((dV1S,S-S * (beta_Ubase * U_base + beta_Uvariant * U_variant) / Pop))
    dUbdt = (S + sigma1 * V1 + sigma2 * V2) * beta_Ubase * U_base / Pop - (delta + rho_U) * U_base
    dUvdt = (S + sigma1v * V1 + sigma2v * V2 ) * beta_Uvariant * U_variant / Pop - (delta + rho_U) * U_variant
    dIbdt = delta * U_base - (omega_I + rho_I + theta_H) * I_base
    dIvdt = delta * U_variant - (2*omega_I + rho_I + theta_H) * I_variant
    dHdt = omega_I * (I_base+2*I_variant) - (rho_H + omega_H + gamma_I) * H + theta_T * T
    dTdt = omega_H * H - (rho_T + gamma_T + theta_T) * T
    dEdt = gamma_T * T + gamma_I * H + theta_H * I
    dRdt = rho_U * U + rho_I * I + rho_H * H - np.min((dV1R,R)) #- R * beta_Uvariant * U_variant / Pop
    dV1dt = dV1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop - dV2
    #min(dV2,dV1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop)   
    dV2dt = dV2
    #min(dV2,dV1 - V1 * (sigma1 * beta_Ubase * U_base + sigma1v *  beta_Uvariant * U_variant)/ Pop) 
    - V2 * (sigma2 * beta_Ubase * U_base + sigma2v * beta_Uvariant * U_variant) / Pop

    return np.concatenate((dSdt, dUbdt, dUvdt, dIbdt, dIvdt, dHdt, dTdt, dEdt, dRdt, dV1dt, dV2dt), axis=None)

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
