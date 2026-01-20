###############################################
### Functions for the cosmological analysis ###
###############################################

from config import *
from support import normalise

def log_prob(x, ivar):
    return -0.5 * np.sum(ivar * x**2)

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

def distribution_redshift_Gaussian(z, epi=-7.553):
    """
    Gaussian time-delay model for NSBH merger delay models in 
    [2011.02717]
    """
    
    f=(1+z)**(3.879*epi)+((1+z)/73.5)**(-0.4901*epi)+((1+z)/3.672)**(-5.691*epi)+((1+z)/3.411)**(-11.46*epi)+((1+z)/3.546)**(-16.38*epi)+((1+z)/3.716)**(-20.66*epi)
    f=f**(1/epi)
    
    return f


def distribution_redshift_LogNormal(z, epi=-5.51):
    """
    LogNormal time-delay model for NSBH merger delay models in 
    [2011.02717]
    """
    
    f=(1+z)**(4.131*epi)+((1+z)/22.37)**(-0.5789*epi)+((1+z)/2.978)**(-4.735*epi)+((1+z)/2.749)**(-10.77*epi)+((1+z)/2.867)**(-17.51*epi)+((1+z)/3.04)**(-(0.08148+z**0.574/0.08682)*epi)
    f=f**(1/epi)
    
    return f


def distribution_redshift_Power(z, epi=-8.161):
    """
    Power-law time-delay model for NSBH merger delay models in 
    [2011.02717]
    """
    
    f=(1+z)**(1.895*epi)+((1+z)/5.722)**(-3.759*epi)+((1+z)/11.55)**(-0.7426*epi)
    f=f**(1/epi)
    
    return f


def redshift_distribution(z_array, H0=HUBBLE, Omega_m=OMEGA_MATTER, w=W_LAMBDA, method='rates'):
    """
    Function that generates random redshifts for our GWs/FRB events.

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
    
    if method == 'rates':
        Dc_squared = D_comoving(z_array, H0, Omega_m, w)**2
        rate = rate_function(z_array)
        Hz = Hubble_function(z_array, H0, Omega_m, w)

        pdf = normalise(4*np.pi*Dc_squared*rate/(Hz*(1+z_array)))
        
    elif method == 'uniform':
        pdf = None
    
    elif method == 'gaussian':
        pdf = distribution_redshift_Gaussian(z_array)
        pdf = normalise(pdf)
        
    elif method == 'lognormal':
        pdf = distribution_redshift_LogNormal(z_array)
        pdf = normalise(pdf)
        
    elif method == 'powerlaw':
        pdf = distribution_redshift_Power(z_array) 
        pdf = normalise(pdf)    
        
    else:
        raise Exception("Wrong method chosen! Choose between 'rates', 'uniform', 'gaussian', 'lognormal' and 'powerlaw'.")
    
    return pdf

def draw_redshift_distribution(z_array, H0=HUBBLE, Omega_m=OMEGA_MATTER, w=W_LAMBDA, N_draws=50, method='rates'):
    """
    Function that generates random redshifts for our GWs/FRB events.

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
    
    pdf=redshift_distribution(z_array, H0, Omega_m, w, method)
    
    redshift_draws = rng.choice(z_array, p=pdf, replace=True, size=N_draws)
        
    ## Check if we have many events with the same redshift
    if np.unique(redshift_draws).size/N_draws < 0.8:
        raise Exception("Many replications in redshifts drawn. Retry the sampling!")      
    
    return redshift_draws


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


def f_IGM_redshift(z, alpha=0.11, f_IGM_0 = f_IGM):
    return f_IGM_0*(1+alpha*z/(1+z))
    

def dDL_integrand_w(z, Om, w):
    """
    Function of the integrand of the dL formula, 
    eq. (5) in [arXiv:1805.12265].
    [https://arxiv.org/abs/2302.10585]
    
    Input
    ----------
    z : redshift
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    """
    return 1/np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))


def dDM_integrand_w(z, Om, w, alpha=0.11, f_IGM_0 = f_IGM):
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

def DM_IGM_O_bh_70(z, O_bh_70, Om=OMEGA_MATTER, w=-1, alpha=0.11, f_IGM_0 = f_IGM):
    """
    Function of the DM formula, 
    eq. (12) in [arXiv:1805.12265].
    
    Update to compare with Macquart+ paper,
    [arXiv:2005.13161], Eq. (2).
    
    Input
    ----------
    z : redshift
    
    H0 : Hubble constant [km/s/Mpc]
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    
    Output
    ---------
    DM : Dispersion measure [pc/cm^3]
    """    
    
    O_bH_0=O_bh_70*70

    factor = 3*C_LIGHT*KM_2_MPC*O_bH_0/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    integral = quad(dDM_integrand_w, 0, z, args=(Om, w, alpha, f_IGM_0))[0]
    
    unit_transform = DM_2_PCCM3
    
    DM = unit_transform*factor*integral
    
    return DM

def luminosity_distance(z, H0=HUBBLE, Om=OMEGA_MATTER, w=W_LAMBDA):
    """
    Function of the dL formula,
    eq. (5) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift (can be a scalar or array)
    H0 : Hubble constant [km/s/Mpc]
    Om : Omega matter
    w : DE EoS parameter (w=-1 for Λ)
    
    Output
    ---------
    dL : Luminosity distance [Mpc] (same type as input z)
    """
    
    # Convert input to numpy array for uniform handling
    z_array = np.asarray(z)
    is_scalar = z_array.ndim == 0
    
    # If scalar input, convert to 1D array for processing
    if is_scalar:
        z_array = z_array.reshape(1)
    
    # Initialize output array
    dL = np.zeros_like(z_array, dtype=float)
    
    # Calculate luminosity distance for each redshift value
    for i, z_val in enumerate(z_array):
        factor = (1 + z_val) * (C_LIGHT * 1e-3) / H0
        integral = quad(dDL_integrand_w, 0, z_val, args=(Om, w))[0]
        dL[i] = factor * integral
    
    # Return scalar if input was scalar, otherwise return array
    if is_scalar:
        return dL[0]
    else:
        return dL

def dLDM_measure(z, H0, Om, w=W_LAMBDA, DM_host=0):
    """
    Function that calculates the dLDM product.
    
    Input
    ----------
    z : redshift
    
    H0 : Hubble constant [km/s/Mpc]
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    
    Output
    ---------
    dLDM : Luminosity distance x Dispersion Measure [Mpc x pc/cm^3]
    """

    dL = luminosity_distance(z, H0, Om, w)
    DM = DM_host+dispersion_measure(z, H0, Om, w)
    
    dLDM = dL*DM
    
    return dLDM


def sigma_dL(z, H0=HUBBLE, Om=OMEGA_MATTER, w=W_LAMBDA, method='Wei'):
    """
    Function that calculates the error of dL,
    eq. (10) in [arXiv:1805.12265].
    
    Input
    ----------
    z : redshift
    
    Om : Omega matter
    
    w : DE EoS parameter (w=-1 for Λ)
    
    Output
    ---------
    s_dL : distance error in Mpc
    """      
    
    dL = luminosity_distance(z, H0, Om, w)
    
    if method == 'Wei':
        
        first_term = 2*dL/SNR_GW
        second_term = 0.05*z*dL
        
        s_dL = np.sqrt(first_term**2+second_term**2)
        
    elif method == 'constant':
        s_dL = dL/DL_ERROR_PERC
        
    else:
        print("Wrong error method!")
    
    return s_dL


def GWs_error(z, H0=HUBBLE, Om=OMEGA_MATTER, w=W_LAMBDA, method='constant'):
    if method=='LVK':
        sigma_ratio = func_lin(z, *LVK_error)/100
        dL=luminosity_distance(z, H0, Om, w)
        return sigma_ratio*dL
    elif method=='CE':
        sigma_ratio = func_lin(z, *CE_error)/100
        dL=luminosity_distance(z, H0, Om, w)
        return sigma_ratio*dL
    elif method=='Wei':
        return sigma_dL(z, H0, Om, w, method=method)
    elif method=='constant':
        return sigma_dL(z, H0, Om, w, method=method)
    else:
        print("Wrong error generation method! Choose from: 'LVK', 'CE', 'Wei', and 'constant'.")    


def sigma_dLDM(dL, DM, error_dL, error_DM=SIGMA_DM):
    """
    Function that calculates the error of dLDM product,
    eq. (14) in [arXiv:1805.12265].
    
    Input
    ----------
    dL : Luminosity distance [Mpc]
    
    DM : Dispersion measure [pc/cm^3]
    
    error_dL : Error in luminosity distance [Mpc]
    
    Output
    ---------
    s_dLDM : Error in the product dLDM [Mpc x pc/cm^3]
    """    
       
    s_dL = error_dL
    s_DM = error_DM
    
    first_term = (DM*s_dL)**2
    second_term = (dL*s_DM)**2
    
    s_dLDM = np.sqrt(first_term+second_term)
    
    return s_dLDM


def sigma_DM_IGM(z, sigma_obs=1.5, sigma_MW=30, sigma_IGM=50, sigma_host=50):
    """
    Function that calculated the total error in the dispersion measure of an FRB,
    eq. (13) in [arXiv:1805.12265].
    """
    
    sigma_total=sigma_obs**2+sigma_MW**2+sigma_IGM**2+(sigma_host/(1+z))**2
    
    return np.sqrt(sigma_total)

################
### PDF HOST ###
################

def pdf_DM_host(DM, e_mu, sigma_host):
    # e^\mu with 20-200 pc cm^{-3} and \sigma_{host} in 0.2-2.0
    
    DM_array = np.asarray(DM)
    result = np.zeros_like(DM_array, dtype=float) # np.full_like(DM_array, 1e-10, dtype=np.float64)
    
    valid_indices = (DM_array > 0) & np.isfinite(DM_array)
    
    if np.any(valid_indices):
        mu = np.log(e_mu)
        valid_DM = DM_array[valid_indices]
        result[valid_indices] = 1.0/(sigma_host*np.sqrt(2*np.pi)*valid_DM) * \
                               np.exp(-(np.log(valid_DM)-mu)**2/(2*(sigma_host**2)))
    
    return result

def Norm_pdf_host(e_mu,sigma_host):
    try:
        int, _=quad_vec(lambda x: pdf_DM_host(x, e_mu, sigma_host), 0, 1e20)
        
        return int
    except:
        print('Normalization pdf_DM_host error')
        
        
##################
### PDF COSMIC ###
##################

def pdf_DM_cosmo(Delta, C_0, A, sigma, alpha=3, beta=3):
    
    Delta_array = np.asarray(Delta)
    result = np.zeros_like(Delta_array, dtype=float) #np.full_like(Delta_array, 1e-10, dtype=np.float64) # np.zeros_like(Delta_array, dtype=float)
    non_zero_indices = Delta_array != 0
    if np.any(non_zero_indices):
        non_zero_Delta = Delta_array[non_zero_indices]
        result[non_zero_indices] = A*(non_zero_Delta**(-beta))*np.exp(-((non_zero_Delta**(-alpha)-C_0)**2)/(2*(alpha**2)*(sigma**2)))
                    
    return result

    # Old version
    # pdf=A*(Delta**(-beta))*np.exp(-((Delta**(-alpha)-C_0)**2)/(2*(alpha**2)*(sigma**2)))
    # return pdf

def DM_diff_HOf(z, HOf, Om=OMEGA_MATTER, w=W_LAMBDA):
    
    def integrand(z, Om, w):
        return (1+z)/np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))

    factor = 3*C_LIGHT*HOf/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    
    integral, _ = quad(integrand, 0, z, args=(Om, w))
    
    unit_transform = DM_2_PCCM3*KM_2_MPC
    
    DM = unit_transform*factor*integral
    
    return DM
    
######################################################################
### For Macquart way to find C0 and A, which \sigma_diff=f/sqrt(z) ###
######################################################################

def f_sigma_DM(F, z, met="Mac"):
    if (met=="log"):
        return F/np.log(1+z)
    else:
        return F/np.sqrt(z)

def find_C0(F, z, sigma_met="Mac", alpha=3, beta=3, x_min=0, x_max=np.inf):
    """
    Use fsolve to find C_0 when to_C_0 = 1
    
    Parameters:
    -----------
    F: float - Structure factor parameter 
    z: float - Redshift
    alpha: float - Alpha parameter
    initial_guess: float - C_0 initial guess
    
    Returns:
    --------
    float: C_0 or None if solution not found
    """
        
    sigma=f_sigma_DM(F,z,met=sigma_met)
    C0 = find_C0_sigma(sigma, x_min=x_min, x_max=x_max, alpha=alpha, beta=beta)
    
    return C0    
    

def find_A(C_0, F, z, alpha=3, beta=3, x_min=0, x_max=np.inf, sigma_met="Mac"):
    sigma=f_sigma_DM(F,z,met=sigma_met)
    
    pdf, error = quad(lambda x: pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta),  x_min, x_max)
    
    try:
        return 1/pdf
            
    except Exception as e:
        print(f"find_A error，pdf={pdf}, C_0={C_0}, F={F}, z={z}, error: {e}")
        return None    

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
    
def f_sqrtvar_delta(F_tilde, z, Om = OMEGA_MATTER, w = W_LAMBDA , met='num'):
    '''
    A general function Error(F,z) to calculate the error of the dispersion measure of diffuse eletron term.
    please do sigma-error interpolate in code to finish error-sigma convert
    example:
    sigma_error = interpolate.interp1d(Errors, Sigmas, kind=1,bounds_error=False, 
    # fill_value='extrapolate'
    )
    '''
    if (met=='num'):
        return F_tilde*np.sqrt(var_z(z, Om=Om, w=w))
    else:
        return F_tilde/np.sqrt(z)
    
def f_sqrtvar_delta_Mac(F_tilde,z):
    '''
    A function Error(F,z) from Macquart method to calculate the error of the dispersion measure of diffuse eletron term.
    please do sigma-error interpolate in code to finish error-sigma convert
    example:
    sigma_error = interpolate.interp1d(Errors, Sigmas, kind=1,bounds_error=False, 
    # fill_value='extrapolate'
    )
    '''

    return F_tilde/np.sqrt(z) 

def find_C0_sigma(sigma, x_min=0, x_max=np.inf, alpha=3, beta=3, condition='mean',initial_guess=1.0):
    """
    Use fsolve to find C_0 when to_C_0 = 1
    
    Parameters:
    -----------
    F: float - Structure factor parameter 
    z: float - Redshift
    alpha: float - Alpha parameter
    initial_guess: float - C_0 initial guess
    condition: condition choose from "mean", "median" or "mode" value equal to 1. For the median recommend minimum sigma from 0.3.
    
    Returns:
    --------
    float: C_0 or None if solution not found
    """
    
    if (condition=="mean"):
        def objective_function(C_0):
            result1,_= quad(lambda x: x*pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta), x_min, x_max)
            result2,_= quad(lambda x: pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta), x_min, x_max)
      
            return result1-result2

    elif (condition=="median"):
        def objective_function(C_0):
        
            result,_= quad(lambda x: pdf_DM_cosmo(x, C_0, 1.0, sigma, alpha, beta), x_min, 1.0)
            int,_= quad(lambda x: pdf_DM_cosmo(x, C_0, 1.0, sigma, alpha, beta), 1.0, x_max)
    
            return result-int

    elif (condition=="mode"):
        def objective_function(C_0):
        
            criteria=C_0**2+4*alpha*beta*sigma**2
            d3=2/(C_0+np.sqrt(criteria))
    
            return d3-1

    else:
        print('Choose condition from["mean","median","mode"]')
        
    try:
        solution = fsolve(objective_function, [initial_guess], full_output=True)
    
        if solution[2] == 1:  # Check if solution is found
            return solution[0][0]
            
    except Exception as e:
        print(f"find_C0 error ({condition}), sigma={sigma}, error: {e}, initial_guess={initial_guess}, check input or change initial guess")
        return None

def find_A_sigma(C_0, sigma, alpha=3, beta=3, x_min=0, x_max=np.inf):
    
    pdf, error = quad(lambda x: pdf_DM_cosmo(x, C_0, 1, sigma, alpha, beta),  x_min, x_max)
    
    try:
        return 1.0/pdf
            
    except Exception as e:
        print(f"find_A error，pdf={pdf}, C_0={C_0}, sigma={sigma}, error: {e}")
        return None 
    
def calculate_var(C0, A, sigma_DM, alpha=3, beta=3, x_min=0, x_max=np.inf, error=1e-20, limit=500):
    '''
    One way is to take error as the sqaure root of the variance
    '''
    
    def first_moment_integrand(delta):
        return delta * pdf_DM_cosmo(delta, C0, A, sigma_DM, alpha, beta)
    
    def second_moment_integrand(delta):
        return delta**2 * pdf_DM_cosmo(delta, C0, A, sigma_DM, alpha, beta)
    
    if (x_max!=np.inf):
        x_max1=int_limit(first_moment_integrand, init=x_max, error=error)
        x_max2=int_limit(second_moment_integrand, init=x_max, error=error)
        x_max=np.max([x_max1,x_max2])
        #print(x_max)
    
    mean, _ = quad(first_moment_integrand, x_min, x_max,limit=limit)

    second_moment, _ = quad_vec(second_moment_integrand, x_min, x_max,limit=limit)
    
    variance = second_moment - mean**2
    
    return variance

def calc_confidence_interval_width(cdf, target_prob=0.6827, x_log_min=-3, x_log_max=2):
    
    # For 2σ confidence interval, we need 0.9545 mass (95.45%)
    # target_prob = 0.6827 # 0.9545 #0.9973
    # Find the central interval
    lower_target = (1 - target_prob) / 2
    upper_target = 1 - lower_target
    
    # Find the values at these probability levels using a more robust approach
    def find_quantile(prob_level, cdf=cdf):
        # First, sample points to get rough range
        test_points = np.logspace(x_log_min, x_log_max, 100)  # Log-spaced points
        cdf_values = np.array([cdf(x) for x in test_points])
        
        # Find points that bracket our target probability
        if prob_level <= np.min(cdf_values):
            return test_points[0]
        if prob_level >= np.max(cdf_values):
            return test_points[-1]
        
        # Find the two points that bracket our target
        for i in range(len(cdf_values)-1):
            if cdf_values[i] <= prob_level <= cdf_values[i+1]:
                a, b = test_points[i], test_points[i+1]
                
                # Use bisection method which doesn't rely on sign change
                mid = (a + b) / 2
                for _ in range(20):  # 20 iterations should be enough
                    mid = (a + b) / 2
                    mid_val = cdf(mid)
                    if abs(mid_val - prob_level) < 1e-6:
                        return mid
                    elif mid_val < prob_level:
                        a = mid
                    else:
                        b = mid
                return mid
        
        # Fallback
        return None
    
    lower_limit = find_quantile(lower_target)
    upper_limit = find_quantile(upper_target)
    
    return upper_limit - lower_limit

##############################################
### PDF SRC (Source/Immediate environment) ###
##############################################

import warnings
import sys

class WarningCatcher:
    def __init__(self):
        self.warnings = []
    
    def __call__(self, message, category, filename, lineno, file=None, line=None):
        self.warnings.append(f"{category.__name__}: {message} in {filename} line {lineno}")
        print(f"WARNING: {category.__name__}: {message} in {filename} line {lineno}", file=sys.stderr)

def pdf_DM_src(DM, b, DM_min, DM_max, debug=True):
    '''
    Assume a power-law distribution for DM_src:
    DM=C*t^{-b}
    If p\propto t, one have:
    P(DM)=C*DM^{-1/b}
    where C is the normalization parameter. a is the free parameter we do the grid search. Note C should be also related with the integrate limitation.
    b within -2/5 to 3 according to Yang & Zhang 2017
    '''
        
    warning_catcher = WarningCatcher()
    old_showwarning = warnings.showwarning
    warnings.showwarning = warning_catcher
    
    if (b == 1.0):
        C = 1.0 / (np.log(DM_max) - np.log(DM_min))
    else:
        index = 1 - 1/b
        if abs(index) < 1e-10:
            C = 1.0 / (np.log(DM_max) - np.log(DM_min))
        else:
            numerator = DM_max**index - DM_min**index # np.exp(index * np.log(DM_max)) - np.exp(index * np.log(DM_min))
            C = index / numerator
    
    warnings.showwarning = old_showwarning
    
    if warning_catcher.warnings and debug:
        print("Warnings captured during pdf_DM_src execution:")
        for warning in warning_catcher.warnings:
            print(f"{warning}, b={b}, DM_min={DM_min}, DM_max={DM_max}, index={index}")
            
    DM_array = np.asarray(DM, dtype=np.float64)
    result = np.zeros_like(DM_array, dtype=np.float64)
    
    valid_indices = (DM_array > 0) & (DM_array >= DM_min) & (DM_array <= DM_max)
    if np.any(valid_indices):
        valid_DM = DM_array[valid_indices]
        log_result = np.log(C) - (1/b) * np.log(valid_DM)
        result[valid_indices] = np.exp(log_result)
    
    return result

def pdf_DM_src_index(DM, index, DM_min, DM_max):
    '''
    Assume a power-law distribution for DM_src:
    DM=C*t^{-b}
    If p\propto t, one have:
    P(DM)=C*DM^{-1/b}
    define the index as -1/b
    where C is the normalization parameter. a is the free parameter we do the grid search. Note C should be also related with the integrate limitation.
    b within -2/5 to 3 according to Yang & Zhang 2017
    
    '''
    
    if (index == -1.0):
        C = 1.0 / (np.log(DM_max) - np.log(DM_min))
    else:
        if abs(index-1.0) < 1e-10:
            C = 1.0 / (np.log(DM_max) - np.log(DM_min))
        else:
            numerator = DM_max**(index+1) - DM_min**(index+1)
            C = (index+1) / numerator
            
    DM_array = np.asarray(DM, dtype=np.float64)
    result = np.zeros_like(DM_array, dtype=np.float64)
    
    valid_indices = (DM_array > 0) & (DM_array >= DM_min) & (DM_array <= DM_max)
    if np.any(valid_indices):
        valid_DM = DM_array[valid_indices]
        log_result = np.log(C) + index * np.log(valid_DM)
        result[valid_indices] = np.exp(log_result)
    
    return result

##################################################################
##################### Automate Analysis ##########################
##################################################################

########## Normal version ##########

def DM_IGM_H0_O_b_f_IGM(z, H0_O_b_f_IGM, Om=OMEGA_MATTER, w=-1):
    
    def integrand(z, Om, w):
        return (1+z)/np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))

    factor = 3*C_LIGHT*H0_O_b_f_IGM/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    
    integral, _ = quad(integrand, 0, z, args=(Om, w))
    
    unit_transform = DM_2_PCCM3*KM_2_MPC
    
    DM = unit_transform*factor*integral
    
    return DM

def calculate_dm_probability_num_HOf(DM_frb_max, z, # Data
                                     S, HOf, e_mu, sigma_host, # parameters
                                     f_sigma_error, # sigma(error) function
                                     space='Delta', # which space to convolution
                                     dropna=False, # drop nan value
                                     error_calculator=None # custom error calculator
                                     ):
    
    '''
    ######### Interpolation version, make sure already do the interpolation #######
    ######### For example, use the code below to do the interpolation ############
    load_arrays=np.load('./interpolation/068_C0median.npz')
    Sigmas=load_arrays['a']
    Vars_sqrt=load_arrays['d']

    sigma_var_inter= interpolate.interp1d(Vars_sqrt, Sigmas, kind=1,bounds_error=False, 
        # fill_value='extrapolate'
        )
    '''
    
    if error_calculator is None:
        error = np.sqrt(f_variance_delta(S, z))
    else:
        # custom error calculator
        error = error_calculator(S, z)
        
    ## Cosmic calculation    
    DM_th = DM_IGM_H0_O_b_f_IGM(z=z, H0_O_b_f_IGM=HOf, Om=OMEGA_MATTER, w=-1)
    
    ## parameters calculation
    sigma=sigma=f_sigma_error(error) # f_sigma_error(np.sqrt(f_variance_delta(S,z)))
    
    C_0=find_C0_sigma(sigma=sigma, alpha=3, beta=3, condition='mean')
    A = find_A_sigma(C_0=C_0, sigma=sigma, alpha=3, beta=3)
    
    ## integration
    if (space=='Delta'):
        
        ## variable=Delta
        varable_array = np.linspace(0, DM_frb_max / DM_th, 5000)

        ## Cosmic calculation
        p_cosmic = pdf_DM_cosmo(varable_array, C_0=C_0, A=A, sigma=sigma)
    
        # print([f_sqrtvar_delta(F,z),sigma,C_0, A])
    
        ## Host calculation
        p_host = pdf_DM_host((1+z)*(DM_frb_max - DM_th * varable_array), e_mu, sigma_host)
        
        ## factor
        factor=1+z
        
    elif (space=='DM'):
        ## variable=DM
        varable_array = np.linspace(0, DM_frb_max * (1+z), 5000)
        
        ## Cosmic calculation
        Deltas = (DM_frb_max-varable_array/(1+z))/DM_th
        p_cosmic = pdf_DM_cosmo(Deltas, C_0=C_0, A=A, sigma=sigma)
    
        # print([f_sqrtvar_delta(F,z),sigma,C_0, A])
    
        ## Host calculation
        p_host = pdf_DM_host(varable_array, e_mu, sigma_host)
        
        ## factor
        factor=1.0/DM_th
        
    else:
        raise ValueError("Invalid space parameter. Choose 'Delta' or 'DM'.")
    
    if (dropna==True):
        p_host[np.isnan(p_host)] = 0
        p_cosmic[np.isnan(p_cosmic)] = 0
        
    ## Combine together    
    prob = np.trapz(p_host*p_cosmic, x=varable_array)
    
    ## Transform to probabilities
    # dDM = np.abs(np.diff(DM_frb_array)[0])/DM_IGM_O_bh_70(z=z, O_bh_70=O_bh_70, Om=OMEGA_MATTER, w=-1,alpha=0)/(1+z)
    
    return prob*factor

########## Fast version ##########

def DM_diff_HOF_fast(z, H0_O_b_f_IGM, Om=OMEGA_MATTER, w=W_LAMBDA, int_N=4000):
    
    def integrand(z, Om, w):
        return (1+z)/np.sqrt(Om*(1+z)**3+(1-Om)*(1+z)**(3*(1+w)))

    factor = 3*C_LIGHT*H0_O_b_f_IGM/(8*PI*G_NEWTON*M_PROTON)*(7/8)
    unit_transform = DM_2_PCCM3*KM_2_MPC
    
    z_array = np.linspace(0, z, int_N)
    integral = np.trapz(integrand(z_array, Om, w), x=z_array)
    
    DM = unit_transform*factor*integral
    
    return DM

def calculate_dm_probability_num_HOf_fast(DM_frb_max, z, # Data
                                     S, HOf, e_mu, sigma_host, # parameters
                                     f_sigma_error, # sigma(error) function
                                     # If in Macquart way, try to define a y=x function as input
                                     f_C0_sigma, f_A_sigma, # C0(sigma) and A(sigma) function
                                     space='Delta', # which space to do convolution
                                     dropna=False, # drop nan value
                                     error_calculator=None, # custom error calculator
                                     # One can use default sqrt(f_variance_delta)~sqrt(S/z) or Macquart's one 
                                     # f_sqrtvar_delta~F/sqrt(z)
                                     Om=OMEGA_MATTER, w=W_LAMBDA, # other cosmology parameters
                                     int_N=5000, # number of integration points
                                     Error_factor = 1.0
                                     ):
    '''
    ######### Interpolation version, make sure already do the interpolation #######
    ######### For example, use the code below to do the interpolation ############
load_arrays=np.load('./interpolation/068_C0median.npz')
Sigmas=load_arrays['a']
Vars_sqrt=load_arrays['d']
C0s=load_arrays['c']
As=load_arrays['b']

sigma_var_inter= interpolate.interp1d(Vars_sqrt, Sigmas, kind=1,bounds_error=False, 
    # fill_value='extrapolate'
    )
    
C0_sigma_inter = interpolate.interp1d(Sigmas, C0s, kind=1,bounds_error=False, 
    # fill_value='extrapolate'
    )
    
A_sigma_inter = interpolate.interp1d(Sigmas, As, kind=1,bounds_error=False, 
    # fill_value='extrapolate'
    )
    '''
    
    if error_calculator is None:
        # Our method
        error = Error_factor * np.sqrt(f_variance_delta(S, z, Om=Om, w=w))
    else:
        # custom error calculator
        error = Error_factor * error_calculator(S, z)
    
    ## Cosmic calculation    
    DM_th = DM_diff_HOF_fast(z=z, H0_O_b_f_IGM=HOf, Om=Om, w=w)
    sigma=f_sigma_error(error) #sigma_var_inter(f_sqrtvar_delta(S,z))# sigma_var_inter(np.sqrt(f_variance_delta(F,z)))
    
    C_0 = f_C0_sigma(sigma)
    A = f_A_sigma(sigma)

    ## integration
    if (space=='Delta'):
        
        ## variable=Delta
        varable_array = np.linspace(0, DM_frb_max / DM_th, int_N)

        ## Cosmic calculation
        p_cosmic = pdf_DM_cosmo(varable_array, C_0=C_0, A=A, sigma=sigma)
    
        # print([f_sqrtvar_delta(F,z),sigma,C_0, A])
    
        ## Host calculation
        p_host = pdf_DM_host((1+z)*(DM_frb_max - DM_th * varable_array), e_mu, sigma_host)
        
        ## factor
        factor=1+z
        
    elif (space=='DM'):
        ## variable=DM
        varable_array = np.linspace(0, DM_frb_max * (1+z), int_N)
        
        ## Cosmic calculation
        Deltas = (DM_frb_max-varable_array/(1+z))/DM_th
        p_cosmic = pdf_DM_cosmo(Deltas, C_0=C_0, A=A, sigma=sigma)
    
        # print([f_sqrtvar_delta(F,z),sigma,C_0, A])
    
        ## Host calculation
        p_host = pdf_DM_host(varable_array, e_mu, sigma_host)
        
        ## factor
        factor=1.0/DM_th
        
    else:
        raise ValueError("Invalid space parameter. Choose 'Delta' or 'DM'.")
    
    if (dropna==True):
        p_host[np.isnan(p_host)] = 0
        p_cosmic[np.isnan(p_cosmic)] = 0
        
    ## Combine together    
    prob = np.trapz(p_host*p_cosmic, x=varable_array)
    
    ## Transform to probabilities
    # dDM = np.abs(np.diff(DM_frb_array)[0])/DM_IGM_O_bh_70(z=z, O_bh_70=O_bh_70, Om=OMEGA_MATTER, w=-1,alpha=0)/(1+z)
    
    return prob*factor

def posterior_analysis(Dv_4D, Dv_array, HOf_array, sigma_host_array, e_mu_array, FRB_data):
    log_posterior_4D = np.zeros_like(Dv_4D, dtype= np.float64)
    total_iterations = len(FRB_data['FRB'])

    # Create flattened parameter combinations
    param_combinations = [
        (Dv, HOf, sigma_host, e_mu)
        for Dv in Dv_array
        for HOf in HOf_array
        for sigma_host in sigma_host_array
        for e_mu in e_mu_array
    ]

    # Initialize progress bar
    pbar = tqdm(total=total_iterations, desc='Computing posteriors')

    # Calculate and accumulate probabilities for each FRB
    for _, row in FRB_data.iterrows():
        # Iterate through parameter combinations
        for idx, (Dv, HOf, sigma_host, e_mu) in enumerate(param_combinations):
            prob = calculate_dm_probability_num_HOf_fast(
                DM_frb_max=row['DM_ext'],
                z=row['z'],
                F=Dv,
                HOf=HOf,
                sigma_host=sigma_host,
                e_mu=e_mu
            )

            # Calculate indices for 4D array
            i = idx // (len(HOf_array) * len(sigma_host_array) * len(e_mu_array))
            j = (idx % (len(HOf_array) * len(sigma_host_array) * len(e_mu_array))) // (len(sigma_host_array) * len(e_mu_array))
            k = (idx % (len(sigma_host_array) * len(e_mu_array))) // len(e_mu_array)
            l = idx % len(e_mu_array)

            # Store result
            if prob > 0:
                log_posterior_4D[i,j,k,l] += np.log(prob)
            else:
                log_posterior_4D[i,j,k,l] = -np.inf

            # Update progress bar
            pbar.update(1)

            # Optional: Add parameter values to progress bar description
            pbar.set_description(f"idx={_}, FRB={row['FRB']}, z={row['z']:.2f}, Dv={Dv:.2f}, HOf={HOf:.2f}, σ={sigma_host:.2f}, μ={e_mu:.2f}, prob={prob:.2f}")

        # # log_posterior = np.log(posterior_4D)
        log_sum = np.logaddexp.reduce(log_posterior_4D.ravel())
        if np.isfinite(log_sum):
            log_posterior_4D = log_posterior_4D - log_sum
        else:
            print(f"Warning: posterior sum is zero for FRB={row['FRB']}")
            break

        pbar.update(1)
        pbar.set_description(f"FRB={row['FRB']}")

    # Close progress bar
    pbar.close()
    
    log_sum = np.logaddexp.reduce(log_posterior_4D.ravel())
    if np.isfinite(log_sum):
        posterior_4D = np.exp(log_posterior_4D - log_sum)
    else:
        print(f"Warning: posterior sum is zero")
    
    return posterior_4D

################### FRB_GW DM sampling ###################

def DM_diff_sampling(z, # redshift
                     sigma_error_inter, C0_sigma_inter, A_sigma_inter, # interpolation functions functions
                     #### if not choose 'standard' mode, use the following parameters ####
                     S, HOF=None, # FRB fitting results
                     #### if choose 'standard' mode, use the following parameters ####
                     H0=HUBBLE, f_diff=0.84, f_diff_alpha=0, # FRB standard parameters
                     Om=OMEGA_MATTER, w=W_LAMBDA, # other cosmology parameters
                     N_draws=1, int_N=2000, # sampling settings
                     mode='standard', # generate events from standard cosmology parameters, else from FRB MCMC fitting results
                     Error_factor = 1.0
                     ):
    """
    Sampling DM_diff for a given redshift and cosmology.
    """
    if (mode=='standard'):
        DM_th=dispersion_measure(z=z, H0=H0, Om=Om, w=w, alpha=f_diff_alpha, f_IGM_0 = f_diff)
    else:
        if (HOF is None):
            raise ValueError("HOF must be provided when not using standard mode.")
        DM_th=DM_diff_HOf(z, HOF, Om=Om, w=w)
        
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

def p_dm_ext_fast(DM_ext, z, # Data
                S, e_mu, sigma_host, # parameters
                f_sigma_error, # sigma(error) function
                # If in Macquart way, try to define a y=x function as input
                f_C0_sigma, f_A_sigma, # C0(sigma) and A(sigma) function
                space='Delta', # which space to do convolution
                dropna=False, # drop nan value
                error_calculator=None, # custom error calculator
                # One can use default sqrt(f_variance_delta)~sqrt(S/z) or Macquart's one f_sqrtvar_delta~F/sqrt(z)
                H0=HUBBLE, f_diff=0.84, f_diff_alpha=0, # FRB standard parameters
                Om=OMEGA_MATTER, w=W_LAMBDA, # other cosmology parameters
                int_N=5000 # integration points
                ):
    
    if error_calculator is None:
        # Our method
        error = np.sqrt(f_variance_delta(S=S, z=z, Om=Om, w=w))
    else:
        # custom error calculator
        error = error_calculator(S, z)
    
    ## Cosmic calculation    
    DM_th = dispersion_measure(z=z, H0=H0, Om=Om, w=w, alpha=f_diff_alpha, f_IGM_0 = f_diff)
    #DM_diff_HOF_fast(z=z, H0_O_b_f_IGM=HOf, Om=Om, w=w)
    sigma=f_sigma_error(error)
    
    C_0=f_C0_sigma(sigma)
    A = f_A_sigma(sigma)

    ## integration
    if (space=='Delta'):
        
        ## variable=Delta
        varable_array = np.linspace(0, DM_ext / DM_th, int_N)

        ## Cosmic calculation
        p_cosmic = pdf_DM_cosmo(varable_array, C_0=C_0, A=A, sigma=sigma)
    
        # print([f_sqrtvar_delta(F,z),sigma,C_0, A])
    
        ## Host calculation
        p_host = pdf_DM_host((1+z)*(DM_ext - DM_th * varable_array), e_mu, sigma_host)
        
        ## factor
        factor=1+z
        
    elif (space=='DM'):
        ## variable=DM
        varable_array = np.linspace(0, DM_ext * (1+z), int_N)
        
        ## Cosmic calculation
        Deltas = (DM_ext-varable_array/(1+z))/DM_th
        p_cosmic = pdf_DM_cosmo(Deltas, C_0=C_0, A=A, sigma=sigma)
    
        # print([f_sqrtvar_delta(F,z),sigma,C_0, A])
    
        ## Host calculation
        p_host = pdf_DM_host(varable_array, e_mu, sigma_host)
        
        ## factor
        factor=1.0/DM_th
        
    else:
        raise ValueError("Invalid space parameter. Choose 'Delta' or 'DM'.")
    
    if (dropna==True):
        p_host[np.isnan(p_host)] = 0
        p_cosmic[np.isnan(p_cosmic)] = 0
        
    ## Combine together    
    prob = np.trapz(p_host*p_cosmic, x=varable_array)
    
    return prob*factor

def DM_ext_sampling(z, # redshift
                     S, HOF, EXP_MU, SIGMA_HOST, # FRB fitting results
                     sigma_error_inter, C0_sigma_inter, A_sigma_inter, # interpolation functions functions
                     # H0=HUBBLE, f_diff=0.84, f_diff_alpha=0, # FRB standard parameters
                     Om=OMEGA_MATTER, w=W_LAMBDA, # cosmology parameters
                     N_draws=1, int_N=4000, # sampling settings,
                     Error_factor = 1.0
                     ):
    """
    Sampling DM_ext for a given redshift and cosmology.
    """
    DM_th=DM_diff_HOf(z, HOF, Om=Om, w=w)
    dm_range=np.linspace(0.25*DM_th, 1000+2.0*DM_th, int_N)
    # np.linspace(0.01, 200+5*DM_th, int_N)
    
    p_range=[
        calculate_dm_probability_num_HOf_fast(
        DM_frb_max=dm,
        z=z,
        S=S,
        HOf=HOF,
        sigma_host=SIGMA_HOST,
        e_mu=EXP_MU,
        f_sigma_error=sigma_error_inter,f_C0_sigma=C0_sigma_inter,f_A_sigma=A_sigma_inter,
        Error_factor = Error_factor
        ) for dm in dm_range]
    
    p_range=normalise(p_range)
    
    dm_ext_obs = rng.choice(dm_range, size=N_draws, replace=True,\
            p=p_range
            )
    
    cdf_num=np.cumsum(p_range)
    cdf = interpolate.interp1d(dm_range, cdf_num, kind=1, bounds_error=False, fill_value='extrapolate')
    error4=calc_confidence_interval_width(cdf, target_prob=0.9545, x_log_min=-2, x_log_max=1+np.log10(dm_range[-1]))
    s_DM_obs = error4/4
    
    return dm_ext_obs[0], s_DM_obs

def DM_ext_sampling_fast(z, # redshift
                     sigma_error_inter, C0_sigma_inter, A_sigma_inter, # interpolation functions functions
                     #### if not choose 'standard' mode, use the following parameters ####
                     S, EXP_MU, SIGMA_HOST, HOF=None, # FRB fitting results
                     #### if choose 'standard' mode, use the following parameters ####
                     H0=HUBBLE, f_diff=f_IGM, f_diff_alpha=f_ALPHA, # FRB standard parameters
                     Om=OMEGA_MATTER, w=W_LAMBDA, # other cosmology parameters
                     N_draws=1, int_N=2000, # sampling settings
                     mode='standard', # generate events from standard cosmology parameters, else from FRB MCMC fitting results
                     Error_factor = 1.0
                     ):
    """
    Sampling DM_ext for a given redshift and cosmology.
    """
    if (mode=='standard'):
        DM_th=dispersion_measure(z=z, H0=H0, Om=Om, w=w, alpha=f_diff_alpha, f_IGM_0 = f_diff)
    else:
        if (HOF is None):
            raise ValueError("HOF must be provided when not using standard mode.")
        DM_th=DM_diff_HOf(z, HOF, Om=Om, w=w)
        
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
    
    dm_diff_obs = rng.choice(dm_range, size=N_draws, replace=True,\
            p=p_range
            )

    LOG_SIGMA = np.sqrt((np.exp(SIGMA_HOST**2)-1)*np.exp(2*np.log(EXP_MU)+SIGMA_HOST**2))  # The standard deviation of the LogNormal
    dm_host_obs = rng.lognormal(mean=np.log(EXP_MU), sigma=SIGMA_HOST, size=N_draws)/(1+z)
    
    return dm_diff_obs[0]+dm_host_obs[0], np.sqrt(s_DM_obs**2+(LOG_SIGMA/(1+z))**2)