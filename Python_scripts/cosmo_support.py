###############################################
### Functions for the cosmological analysis ###
###############################################

from config import *
from support import normalise

###############################################

def Hubble_function(z, H0, Omega_m, w):
    """
    Hubble function
    """
    return np.sqrt(H0**2*(Omega_m*(1+z)**3+(1-Omega_m)*(1+z)**(3*(1+w))))


def rate_function(z):
    rate = (1+2*z)*(z<=1)+3/4*(5-z)*(z>1)*(z<5)
    return rate


def D_comoving(z, H0, Omega_m, w):
    return wCDM(H0=H0, Om0=Omega_m, Ode0=1-Omega_m, w0=w).comoving_distance(z).value

###############################################

def redshift_distribution(z_array, H0=HUBBLE, Omega_m=OMEGA_MATTER, w=W_LAMBDA):
    """
    Function that generates random redshifts for our FRB events.

    Input
    ----------
    z_array : redshift range from which to draw samples
    
    H0 : Hubble constant [km/s/Mpc]
    
    Omega_m : Omega matter
    
    method : Choose between `rates` and `uniform`. Defines the method used to draw random samples.
    
    Output
    ---------
    PDF : redshift distribution
    """
    
    Dc_squared = D_comoving(z_array, H0, Omega_m, w)**2
    rate = rate_function(z_array)
    Hz = Hubble_function(z_array, H0, Omega_m, w)

    pdf = normalise(4*np.pi*Dc_squared*rate/(Hz*(1+z_array)))
        
    return pdf
    

def draw_redshift_distribution(z_array, H0=HUBBLE, Omega_m=OMEGA_MATTER, w=W_LAMBDA, N_draws=50):
    """
    Function that generates random redshifts for our FRB events.

    Input
    ----------
    z_array : redshift range from which to draw samples
    
    H0 : Hubble constant [km/s/Mpc]
    
    Omega_m : Omega matter
    
    N_draws : Number of mock redshifts to draw
    
    method : Choose between `rates` and `uniform`. Defines the method used to draw random samples.
    
    Output
    ---------
    redshift_draws : Mock redshift observations
    """
    
    pdf=redshift_distribution(z_array, H0, Omega_m, w)
    
    redshift_draws = rng.choice(z_array, p=pdf, replace=True, size=N_draws)
        
    ## Check if we have many events with the same redshift
    if np.unique(redshift_draws).size/N_draws < 0.8:
        raise Exception("Many replications in redshifts drawn. Retry the sampling!")      
    
    return redshift_draws


def generate_events(N_events, z_min, z_max=2.0, z_res=500,\
                   H0=HUBBLE, Omega_m=OMEGA_MATTER, w=W_LAMBDA, alpha=f_ALPHA, f_IGM_0=f_IGM,\
                   method="Gaussian", error_size=1, data_path='--'):

    redshift_array = np.linspace(z_min, z_max, z_res)
    z_centres = draw_redshift_distribution(redshift_array, H0, Omega_m, w, N_events)

    # Theoretical DM, fiducial cosmo
    DM_centres = dispersion_measure(z_centres, H0, Omega_m, w, alpha, f_IGM_0)

    if method=='Gaussian':
        ## Simple Gaussian pdf - Scatter observations according to errors
        DM_obs_centre = rng.normal(DM_centres, SIGMA_DM)
        s_DM_obs = np.repeat(SIGMA_DM, N_events)

    elif method=='Pdf':
        # DATA_PATH = './interpolation/095_C0mean.npz'
        Sigmas, Errors, C0s, As, sigma_error_inter, C0_sigma_inter, A_sigma_inter = FRBs_load_and_create_interpolators(data_path)
        
        ## Modelling the DM pdf -- Standard HOF
        DM_obs_centre = np.zeros_like(z_centres)
        s_DM_obs = np.zeros_like(z_centres)
        
        for idx, z_val in enumerate(tqdm(z_centres)):
            DM_obs_centre[idx], s_DM_obs[idx], _ = \
                DM_diff_sampling(z=z_val, 
                                S=S_FRB, HOF=HOF,
                                sigma_error_inter=sigma_error_inter,
                                C0_sigma_inter=C0_sigma_inter,
                                A_sigma_inter=A_sigma_inter,
                                H0=H0, f_diff=f_IGM_0, f_diff_alpha=alpha,
                                Om=Omega_m, w=w, N_draws=1,
                                Error_factor = error_size
                                )
            
    else:
        print("Warning! For method choose either 'Gaussian' or 'Pdf'.")


    return z_centres, DM_centres, DM_obs_centre, s_DM_obs


    

###############################################

def FRBs_load_and_create_interpolators(path):
    load_arrays = np.load(path)
    Sigmas = load_arrays['a']
    Errors = load_arrays['d']
    C0s = load_arrays['c'] 
    As = load_arrays['b']
    
    sigma_error_inter = interpolate.interp1d(Errors, Sigmas, kind=1, bounds_error=False,fill_value='extrapolate')
    C0_sigma_inter = interpolate.interp1d(Sigmas, C0s, kind=1, bounds_error=False,fill_value='extrapolate')
    A_sigma_inter = interpolate.interp1d(Sigmas, As, kind=1, bounds_error=False,fill_value='extrapolate')

    return Sigmas, Errors, C0s, As, sigma_error_inter, C0_sigma_inter, A_sigma_inter


def f_IGM_redshift(z, alpha=f_ALPHA, f_IGM_0 = f_IGM):
    return f_IGM_0*(1+alpha*z/(1+z))
    

def dDM_integrand_w(z, Om, w, alpha=f_ALPHA, f_IGM_0 = f_IGM):
    """
    Function of the integrand of the DM formula, 
    eq. (12) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    """
    f_IGM_z = f_IGM_redshift(z, alpha, f_IGM_0)
    
    return f_IGM_z*(1+z)/np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))


def dispersion_measure(z, H0=HUBBLE, Om=OMEGA_MATTER, w=W_LAMBDA, alpha=f_ALPHA, f_IGM_0 = f_IGM):
    """
    Function of the DM formula, 
    eq. (12) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift (can be a scalar or array)
    
    H0 : Hubble constant [km/s/Mpc]
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    
    alpha : Alpha parameter can be 0.11 (default is 0)
    
    Output
    ---------
    DM : Dispersion measure [pc/cm^3]
    """    

    # Convert input to numpy array for uniform handling
    z_array = np.asarray(z)
    is_scalar = z_array.ndim == 0
    
    # If scalar input, convert to 1D array for processing
    if is_scalar:
        z_array = z_array.reshape(1)
    
    # Initialize output array
    DM = np.zeros_like(z_array, dtype=float)
    
    # Calculate DM for each redshift value
    factor = 3*C_LIGHT*(H0*KM_2_MPC)*OMEGA_BARYONS/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    unit_transform = DM_2_PCCM3
    for i, z_val in enumerate(z_array):
        integral = quad(dDM_integrand_w, 0, z_val, args=(Om, w, alpha, f_IGM_0))[0]
        DM[i] = unit_transform*factor*integral
    
    # Return scalar if input was scalar, otherwise return array
    if is_scalar:
        return DM[0]
    else:
        return DM



def PhiDM_wCDM(x, w):
    """ NOTE: Compared to previous definitions, here b_i are in the
    denominator and c_i in the numerator.
    """

    w2 = w**2
    w3 = w**3
    w4 = w**4
    w5 = w**5
    w6 = w**6
    w7 = w**7
    w8 = w**8
    w9 = w**9
    w10 = w**10
    w11 = w**11
    w12 = w**12
    
    b1 = 7*(13554501120*w7 + 2144869632*w6 + 186662880*w5 + 11424240*w4 + 447552*w3 + 10512*w2 + 138*w + 1)/(4*(17420977152*w7 + 2348289792*w6 + 206452800*w5 + 12398832*w4 + 473904*w3 + 10800*w2 + 144*w + 1))
    b2 = 7*(287698065408*w8 + 67349242368*w7 + 7092738432*w6 + 484856928*w5 + 23238576*w4 + 732672*w3 + 14328*w2 + 162*w + 1)/(8*(522629314560*w8 + 87869670912*w7 + 8541873792*w6 + 578417760*w5 + 26615952*w4 + 797904*w3 + 15120*w2 + 174*w + 1))
    b3 = 7*(3862337458176*w9 + 1485633788928*w8 + 219003450624*w7 + 17781674688*w6 + 976307904*w5 + 38712816*w4 + 1042416*w3 + 17892*w2 + 180*w + 1)/(64*(12543103549440*w9 + 2631501416448*w8 + 292874641920*w7 + 22423900032*w6 + 1217200608*w5 + 45765648*w4 + 1160784*w3 + 19296*w2 + 198*w + 1))
    c0 = 6*w
    c1 = 15*w*(113857809408*w8 + 30024815616*w7 + 3631469760*w6 + 274710528*w5 + 14793840*w4 + 525312*w3 + 11556*w2 + 144*w + 1)/(2*(104525862912*w8 + 31510715904*w7 + 3587006592*w6 + 280845792*w5 + 15242256*w4 + 538704*w3 + 11664*w2 + 150*w + 1))
    c2 = 9*w*(48333274988544*w10 + 15147986411520*w9 + 2611359461376*w8 + 276998911488*w7 + 19936155456*w6 + 1037489472*w5 + 39459312*w4 + 1036800*w3 + 17676*w2 + 180*w + 1)/(4*(37629310648320*w10 + 15733943967744*w9 + 2719298304000*w8 + 283269477888*w7 + 20869742016*w6 + 1114953984*w5 + 42066864*w4 + 1082592*w3 + 18324*w2 + 192*w + 1))
    c3 = 3*w*(35039125420572672*w12 + 8629966763753472*w11 + 1619918846263296*w10 + 221082842105856*w9 + 21477950191872*w8 + 1497341832192*w7 + 77109117696*w6 + 3001784832*w5 + 87399648*w4 + 1812672*w3 + 25200*w2 + 216*w + 1)/(32*(16255862200074240*w12 + 8377494841294848*w11 + 1873191824621568*w10 + 252316887183360*w9 + 23632344926208*w8 + 1641458763648*w7 + 85870694592*w6 + 3349442016*w5 + 95451696*w4 + 1935144*w3 + 26820*w2 + 234*w + 1))
    
    return (c0+c1*x+c2*x**2+c3*x**3)/(1.0+b1*x+b2*x**2+b3*x**3)
        
def x_phiDM_wCDM(z, Om, w):
    return (1.-Om)/Om*(1.+z)**(3*w)
    
def DM_pade_wCDM(z, H0, Om, w):
    x0 = x_phiDM_wCDM(0, Om, w)
    x = x_phiDM_wCDM(z, Om, w)
    factor = -1.0/Om**(1/2)*(1/(3*w))

    DM_factor = 3*C_LIGHT*(H0*KM_2_MPC)*OMEGA_BARYONS/(8*PI*G_NEWTON*M_PROTON)*(7/8)*f_IGM
    unit_transform = DM_2_PCCM3
    
    return DM_factor*unit_transform*factor*(PhiDM_wCDM(x0, w)-PhiDM_wCDM(x, w)*(1.+z)**(1/2))           


##################
### PDF COSMIC ###
##################

def pdf_DM_cosmo(Delta, C_0, A, sigma, alpha=3, beta=3):
    
    Delta_array = np.asarray(Delta)
    result = np.zeros_like(Delta_array, dtype=float) 
    non_zero_indices = (Delta_array != 0)
    if np.any(non_zero_indices):
        non_zero_Delta = Delta_array[non_zero_indices]
        result[non_zero_indices] = A*(non_zero_Delta**(-beta))*np.exp(-((non_zero_Delta**(-alpha)-C_0)**2)/(2*(alpha**2)*(sigma**2)))
                    
    return result


def pdf_DM_cosmo_LHD(Delta, C_0, A, sigma, alpha=3, beta=3):
    pdf = A*(Delta**(-beta))*np.exp(-((Delta**(-alpha)-C_0)**2)/(2*(alpha**2)*(sigma**2)))
                    
    return pdf    

  

######################################
### For error-sigma_{diff} version ###
######################################

'''
The following functions are used for calculate the error for ∆ and get the sigma (but in function we do a reverse way which from each sigma to calculate error). This is because we find the \sigma_{diff} in P(∆) is not exactly its error and they also don't show linear relation.
'''
    
def var_z(z, Om = OMEGA_MATTER, w = W_LAMBDA):
    # np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))
    def dDc(x):
        return 1/np.sqrt(Om*(1+x)**3+(1-Om)*(1+x)**(3*(1+w)))
    
    def dDM(x):
        return (1+x)/np.sqrt(Om*(1+x)**3+(1-Om)*(1+x)**(3*(1+w)))
    
    def single_z_calc(z_val):
        int1, _ = quad(dDc, 0, z_val)
        int2, _ = quad(dDM, 0, z_val)
        return int1/int2**2
    
    vectorized_calc = np.vectorize(single_z_calc)
    return vectorized_calc(z)

def f_variance_delta(S, z, Om = OMEGA_MATTER, w = W_LAMBDA, met='num'):
    '''
    A general function Error(S,z) to calculate the error of the dispersion measure of diffuse eletron term.
    please do sigma-error interpolate in code to finish error-sigma convert
    example:
    sigma_error = interpolate.interp1d(Errors, Sigmas, kind=1,bounds_error=False, 
    # fill_value='extrapolate'
    )
    '''
    if (met=='num'):
        return S*var_z(z, Om=Om, w=w)
    else:
        return S/z
    

################### FRB_GW DM sampling ###################

def DM_diff_sampling(z, # redshift
                     sigma_error_inter, C0_sigma_inter, A_sigma_inter, # interpolation functions functions
                     #### if not choose 'standard' mode, use the following parameters ####
                     S, HOF=None, # FRB fitting results
                     #### if choose 'standard' mode, use the following parameters ####
                     H0=HUBBLE, f_diff=f_IGM, f_diff_alpha=f_ALPHA, # FRB standard parameters
                     Om=OMEGA_MATTER, w=W_LAMBDA, # other cosmology parameters
                     N_draws=1, int_N=2000, # sampling settings
                     Error_factor = 1.0
                     ):
    """
    Sampling DM_diff for a given redshift and cosmology.
    """
    DM_th=dispersion_measure(z=z, H0=H0, Om=Om, w=w, alpha=f_diff_alpha, f_IGM_0 = f_diff)

        
    error=Error_factor * np.sqrt(f_variance_delta(S=S, z=z, Om=Om, w=w))
    s_DM_obs = error*DM_th
    
    sigma_diff=sigma_error_inter(error)
    C0=C0_sigma_inter(sigma_diff)
    A=A_sigma_inter(sigma_diff)
    
    dm_range=np.linspace(0.25*DM_th, 500+2.0*DM_th, int_N)
    
    p_range=[
        pdf_DM_cosmo(Delta=dm/DM_th, C_0=C0, A=A, sigma=sigma_diff, alpha=3, beta=3)/DM_th
        for dm in dm_range]
    
    p_range=normalise(p_range)
    
    dm_diff_obs = rng.choice(dm_range, size=int_N, replace=True,\
            p=p_range
            )
    
    return dm_diff_obs[0], s_DM_obs, dm_diff_obs


