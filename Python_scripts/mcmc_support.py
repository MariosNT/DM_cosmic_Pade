############################
### Packages and Modules ###
############################

import emcee
from config import *
from cosmo_support import *

# Useful when running the PDF Pade + host + ...

from functools import lru_cache

@lru_cache(maxsize=None)
def _get_interpolators(data_path):
    return FRBs_load_and_create_interpolators(data_path)

####################
### Prior Ranges ###
####################

H0_MIN, H0_MAX = 20, 120
Om_MIN, Om_MAX = 0.0, 1.0
W_MIN, W_MAX   = -2.0, 0.0
SH_MIN, SH_MAX = 0.1, 1.4
EM_MIN, EM_MAX = 20, 350
SD_MIN, SD_MAX = 0, 900 # Limits for Gaussian prior #50, 400


##############################
##### Priors & Likelihoods ###
##############################


def log_prior_frb(theta, H0_min=H0_MIN, H0_max=H0_MAX, Om_min=Om_MIN, Om_max=Om_MAX, W_min=W_MIN, W_max=W_MAX):
    """
    Calculate the log of the prior probability for a set of parameters.

    Args:
        theta: Array of parameters [hubble, omega, w]

    Returns:
        Log prior probability
    """
    hubble, omega, w = theta
    
    # Check if parameters are within prior ranges
    if (H0_min <= hubble <= H0_max and 
        Om_min <= omega <= Om_max and 
        W_min <= w <= W_max ):
        return 0.0  # Log(1) = 0, flat prior
    else:
        return -np.inf

def log_prior_frb_host(theta, H0_min=H0_MIN, H0_max=H0_MAX,\
                       Om_min=Om_MIN, Om_max=Om_MAX,\
                       W_min=W_MIN, W_max=W_MAX,\
                       Shost_min=SH_MIN, Shost_max=SH_MAX,\
                       e_mu_min=EM_MIN, e_mu_max=EM_MAX):
    """
    Calculate the log of the prior probability for a set of parameters.

    Args:
        theta: Array of parameters [hubble, omega, w, Shost, e_mu], 
                Cosmology + Host properties

    Returns:
        Log prior probability
    """
    hubble, omega, w, Shost, e_mu = theta
    
    # Check if parameters are within prior ranges
    if (H0_min <= hubble <= H0_max and 
        Om_min <= omega <= Om_max and 
        W_min <= w <= W_max and
        Shost_min <= Shost <= Shost_max and
        e_mu_min <= e_mu <= e_mu_max
       ):
        return 0.0  # Log(1) = 0, flat prior
    else:
        return -np.inf

def log_prior_frb_host_SD(theta, H0_min=H0_MIN, H0_max=H0_MAX,\
                       Om_min=Om_MIN, Om_max=Om_MAX,\
                       W_min=W_MIN, W_max=W_MAX,\
                       Shost_min=SH_MIN, Shost_max=SH_MAX,\
                       e_mu_min=EM_MIN, e_mu_max=EM_MAX,\
                       SD_min=SD_MIN, SD_max=SD_MAX):
    """
    Calculate the log of the prior probability for a set of parameters.

    Args:
        theta: Array of parameters [hubble, omega, w, Shost, e_mu], 
                Cosmology + Host properties

    Returns:
        Log prior probability
    """
    hubble, omega, w, Shost, e_mu, SD = theta

    ## Gaussian prior for SD (ignoring the constant normalisation)
    mu_SD, sigma_SD = 180.0, 40.0
    lp_SD = -0.5 * ((SD - mu_SD) / sigma_SD) ** 2 # - np.log(np.sqrt(2 * np.pi) * sigma_SD)
    
    # Check if parameters are within prior ranges
    if (H0_min <= hubble <= H0_max and 
        Om_min <= omega <= Om_max and 
        W_min <= w <= W_max and
        Shost_min <= Shost <= Shost_max and
        e_mu_min <= e_mu <= e_mu_max and
        SD_min <= SD <= SD_max
       ):
        return 0.0 + lp_SD # Log(1) = 0, flat prior
    else:
        return -np.inf 

###########################
### Standard Gaussian model

def log_likelihood_frb_gauss(theta, z_o, DM_o, s_DM_o):
    hubble, omega, w = theta
    model = dispersion_measure(z=z_o, H0=hubble, Om=omega, w=w, alpha=f_ALPHA, f_IGM_0 = f_IGM)
    sigma2 = s_DM_o**2
    return -0.5 * np.sum((DM_o - model) ** 2 / sigma2)

def log_probability_frb_gauss(theta, z_o, DM_o, s_DM_o,\
                              H0_min=H0_MIN, H0_max=H0_MAX,\
                              Om_min=Om_MIN, Om_max=Om_MAX,\
                              W_min=W_MIN, W_max=W_MAX):
    
    lp = log_prior_frb(theta, H0_min, H0_max, Om_min, Om_max, W_min, W_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_frb_gauss(theta, z_o, DM_o, s_DM_o)  


###########################
####### Pade Gaussian model

def log_likelihood_frb_Pade_gauss(theta, z_o, DM_o, s_DM_o):
    hubble, omega, w = theta
    model = DM_pade_wCDM(z=z_o, H0=hubble, Om=omega, w=w)
    sigma2 = s_DM_o**2
    return -0.5 * np.sum((DM_o - model) ** 2 / sigma2)

def log_probability_frb_Pade_gauss(theta, z_o, DM_o, s_DM_o,\
                                   H0_min=H0_MIN, H0_max=H0_MAX,\
                                   Om_min=Om_MIN, Om_max=Om_MAX,\
                                   W_min=W_MIN, W_max=W_MAX):
    lp = log_prior_frb(theta, H0_min, H0_max, Om_min, Om_max, W_min, W_max)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_frb_Pade_gauss(theta, z_o, DM_o, s_DM_o)


#######################################################################

###########################
######## Standard PDF model

def log_likelihood_frb_pdf(theta, z_o, DM_o, s_DM_o, data_path='--'):
    hubble, omega, w = theta

    model = dispersion_measure(z=z_o, H0=hubble, Om=omega, w=w, alpha=f_ALPHA, f_IGM_0 = f_IGM)

    error=s_DM_o/model

    error_inter_min = 1e-3
    error_inter_max = 0.71

    error_clipped = np.clip(error, error_inter_min, error_inter_max)

    _, _, _, _, sigma_error_inter, C0_sigma_inter, A_sigma_inter = FRBs_load_and_create_interpolators(data_path)
    sigma_diff=sigma_error_inter(error_clipped)
    C0=C0_sigma_inter(sigma_diff)
    A=A_sigma_inter(sigma_diff)

    p_obs = pdf_DM_cosmo_LHD(Delta=DM_o/model, C_0=C0, A=A, sigma=sigma_diff, alpha=3, beta=3)/model
   
    return np.sum(np.log10(p_obs))

def log_probability_frb_pdf(theta, z_o, DM_o, s_DM_o, data_path='--',\
                            H0_min=H0_MIN, H0_max=H0_MAX,\
                            Om_min=Om_MIN, Om_max=Om_MAX,\
                            W_min=W_MIN, W_max=W_MAX):
    
    lp = log_prior_frb(theta, H0_min, H0_max, Om_min, Om_max, W_min, W_max)
    
    if (not np.isfinite(lp)):
        return -np.inf
    
    return lp + log_likelihood_frb_pdf(theta, z_o, DM_o, s_DM_o, data_path)


###########################
############ Pade PDF model

def log_likelihood_frb_Pade_pdf(theta, z_o, DM_o, s_DM_o, data_path='--'):
    hubble, omega, w = theta

    model = DM_pade_wCDM(z=z_o, H0=hubble, Om=omega, w=w)

    error=s_DM_o/model

    error_inter_min = 1e-3
    error_inter_max = 0.71

    error_clipped = np.clip(error, error_inter_min, error_inter_max)
    
    _, _, _, _, sigma_error_inter, C0_sigma_inter, A_sigma_inter = FRBs_load_and_create_interpolators(data_path)
    
    sigma_diff=sigma_error_inter(error_clipped)
    C0=C0_sigma_inter(sigma_diff)
    A=A_sigma_inter(sigma_diff)

    p_obs = pdf_DM_cosmo_LHD(Delta=model/DM_o, C_0=C0, A=A, sigma=sigma_diff, alpha=3, beta=3)/model
    log_prob = np.sum(np.log10(p_obs))

    if np.any(np.isnan(log_prob)):
        return -np.inf
    else:
        return log_prob

def log_probability_frb_Pade_pdf(theta, z_o, DM_o, s_DM_o, data_path='--',\
                                 H0_min=H0_MIN, H0_max=H0_MAX,\
                                 Om_min=Om_MIN, Om_max=Om_MAX,\
                                 W_min=W_MIN, W_max=W_MAX):
    
    lp = log_prior_frb(theta, H0_min, H0_max, Om_min, Om_max, W_min, W_max)
    
    if (not np.isfinite(lp)):
        return -np.inf
    
    return lp + log_likelihood_frb_Pade_pdf(theta, z_o, DM_o, s_DM_o, data_path)



############################################
############ Pade PDF model (including host)
  

def log_likelihood_frb_Pade_pdf_host(theta, z_o, DM_o, s_DM_o, data_path='--'):
    hubble, omega, w, S_host, e_mu = theta

    model = DM_pade_wCDM(z=z_o, H0=hubble, Om=omega, w=w)

    error=s_DM_o/model

    error_inter_min = 1e-3
    error_inter_max = 0.71

    error_clipped = np.clip(error, error_inter_min, error_inter_max)

    _, _, _, _, sigma_error_inter, C0_sigma_inter, A_sigma_inter = _get_interpolators(data_path) #FRBs_load_and_create_interpolators(data_path)
    sigma_diff=sigma_error_inter(error_clipped)
    C0=C0_sigma_inter(sigma_diff)
    A=A_sigma_inter(sigma_diff)

    p_obs = pdf_DM_joint_LHD(DM_o, z_o, model, e_mu, S_host, sigma_diff, C0, A,\
                     alpha=3, beta=3, space='Delta', dropna=True, int_N=1500)
    log_prob = np.sum(np.log10(p_obs))

    if np.any(np.isnan(log_prob)):
        return -np.inf
    else:
        return log_prob



def log_probability_frb_Pade_pdf_host(theta, z_o, DM_o, s_DM_o, data_path='--',\
                        H0_min=H0_MIN, H0_max=H0_MAX,\
                        Om_min=Om_MIN, Om_max=Om_MAX,\
                        W_min=W_MIN, W_max=W_MAX,\
                        S_host_min=SH_MIN, S_host_max=SH_MAX,\
                        e_mu_min=EM_MIN, e_mu_max=EM_MAX):
    
    lp = log_prior_frb_host(theta, H0_min, H0_max, Om_min, Om_max, W_min, W_max,\
                            S_host_min, S_host_max, e_mu_min, e_mu_max)
    
    if (not np.isfinite(lp)):
        return -np.inf
    
    return lp + log_likelihood_frb_Pade_pdf_host(theta, z_o, DM_o, s_DM_o, data_path)


######################################################
############ Pade PDF model (including host) & S error

    
def log_likelihood_frb_Pade_pdf_host_SD(theta, z_o, DM_o, data_path='--'):
    hubble, omega, w, S_host, e_mu, SD = theta

    model = DM_pade_wCDM(z=z_o, H0=hubble, Om=omega, w=w)

    error=SD/model

    error_inter_min = 1e-3
    error_inter_max = 0.71

    error_clipped = np.clip(error, error_inter_min, error_inter_max)

    _, _, _, _, sigma_error_inter, C0_sigma_inter, A_sigma_inter = _get_interpolators(data_path) 
    sigma_diff=sigma_error_inter(error_clipped)
    C0=C0_sigma_inter(sigma_diff)
    A=A_sigma_inter(sigma_diff)

    p_obs = pdf_DM_joint_LHD(DM_o, z_o, model, e_mu, S_host, sigma_diff, C0, A,\
                     alpha=3, beta=3, space='Delta', dropna=True, int_N=1500)
    log_prob = np.sum(np.log10(p_obs))

    if np.any(np.isnan(log_prob)):
        return -np.inf
    else:
        return log_prob

def log_probability_frb_Pade_pdf_host_SD(theta, z_o, DM_o, data_path='--',\
                        H0_min=H0_MIN, H0_max=H0_MAX,\
                        Om_min=Om_MIN, Om_max=Om_MAX,\
                        W_min=W_MIN, W_max=W_MAX,\
                        S_host_min=SH_MIN, S_host_max=SH_MAX,\
                        e_mu_min=EM_MIN, e_mu_max=EM_MAX,\
                        SD_min=SD_MIN, SD_max=SD_MAX):
    
    lp = log_prior_frb_host_SD(theta, H0_min, H0_max, Om_min, Om_max, W_min, W_max,\
                            S_host_min, S_host_max, e_mu_min, e_mu_max, SD_min, SD_max)
    
    if (not np.isfinite(lp)):
        return -np.inf
    
    return lp + log_likelihood_frb_Pade_pdf_host_SD(theta, z_o, DM_o, data_path) 


##############################################
############# Run MCMC analysis ##############
##############################################


def run_mcmc_analysis(z_c, DM_obs_c, s_DM_obs_c, log_probability,\
                      H0_min=H0_MIN, H0_max=H0_MAX,\
                      Om_min=Om_MIN, Om_max=Om_MAX, W_min=W_MIN, W_max=W_MAX,\
                      H0_init=70, Om_init=0.5, w_init=-1.0, nwalk=20, N_samples=3000, burn_in=100):
    
    ## Initialising MCMC
    initial_params = np.array([H0_init, Om_init, w_init])
    ndim = len(initial_params)
    pos = initial_params + np.array([1, 5e-2, 1e-1]) * rng.normal(0, 1, size=(nwalk, ndim))

    
    sampler = emcee.EnsembleSampler(nwalk, ndim, log_probability, args=(z_c, DM_obs_c, s_DM_obs_c, H0_min, H0_max, Om_min, Om_max, W_min, W_max))
    sampler.run_mcmc(pos, N_samples, progress=True);

    samples_all = sampler.get_chain()
    samples_flat = sampler.get_chain(discard=burn_in, thin=15, flat=True)

    return samples_all, samples_flat


def run_mcmc_analysis_pdf(z_c, DM_obs_c, s_DM_obs_c, log_probability, data_path='--',\
                      H0_min=H0_MIN, H0_max=H0_MAX,\
                      Om_min=Om_MIN, Om_max=Om_MAX, W_min=W_MIN, W_max=W_MAX,\
                      H0_init=70, Om_init=0.5, w_init=-1.0, nwalk=20, N_samples=3000, burn_in=100):
    
    ## Initialising MCMC
    initial_params = np.array([H0_init, Om_init, w_init])
    ndim = len(initial_params)
    pos = initial_params + np.array([1, 5e-2, 1e-1]) * rng.normal(0, 1, size=(nwalk, ndim))

    
    sampler = emcee.EnsembleSampler(nwalk, ndim, log_probability, args=(z_c, DM_obs_c, s_DM_obs_c, data_path, H0_min, H0_max, Om_min, Om_max, W_min, W_max))
    sampler.run_mcmc(pos, N_samples, progress=True);

    samples_all = sampler.get_chain()
    samples_flat = sampler.get_chain(discard=burn_in, thin=15, flat=True)

    return samples_all, samples_flat


def run_mcmc_analysis_pdf_host(z_c, DM_obs_c, s_DM_obs_c, log_probability, data_path='--',\
                       H0_min=H0_MIN, H0_max=H0_MAX,\
                       Om_min=Om_MIN, Om_max=Om_MAX,\
                       W_min=W_MIN, W_max=W_MAX,\
                       S_host_min=SH_MIN, S_host_max=SH_MAX,\
                       e_mu_min=EM_MIN, e_mu_max=EM_MAX,\
                       H0_init=70, Om_init=0.5, w_init=-1.0, S_host_init=1.0, e_mu_init=100,\
                       nwalk=20, N_samples=3000, burn_in=100):
    
    ## Initialising MCMC
    initial_params = np.array([H0_init, Om_init, w_init, S_host_init, e_mu_init])
    ndim = len(initial_params)
    pos = initial_params + np.array([1, 5e-2, 1e-1, 1e-2, 5]) * rng.normal(0, 1, size=(nwalk, ndim))

    
    sampler = emcee.EnsembleSampler(nwalk, ndim, log_probability, args=(z_c, DM_obs_c, s_DM_obs_c, data_path, H0_min, H0_max, Om_min, Om_max, W_min, W_max, S_host_min, S_host_max, e_mu_min, e_mu_max))
    sampler.run_mcmc(pos, N_samples, progress=True);

    tau_time = sampler.get_autocorr_time(quiet=True)

    samples_all = sampler.get_chain()
    samples_flat = sampler.get_chain(discard=burn_in, thin=15, flat=True)

    return samples_all, samples_flat, tau_time.max()
    

def run_mcmc_analysis_pdf_host_SD(z_c, DM_obs_c, log_probability, data_path='--',\
                       H0_min=H0_MIN, H0_max=H0_MAX,\
                       Om_min=Om_MIN, Om_max=Om_MAX,\
                       W_min=W_MIN, W_max=W_MAX,\
                       S_host_min=SH_MIN, S_host_max=SH_MAX,\
                       e_mu_min=EM_MIN, e_mu_max=EM_MAX,\
                       SD_min=SD_MIN, SD_max=SD_MAX,\
                       H0_init=70, Om_init=0.5, w_init=-1.0, S_host_init=1.0, e_mu_init=100, SD_init=100,\
                       nwalk=20, N_samples=3000, burn_in=100):
    
    ## Initialising MCMC
    initial_params = np.array([H0_init, Om_init, w_init, S_host_init, e_mu_init, SD_init])
    ndim = len(initial_params)
    pos = initial_params + np.array([1, 5e-2, 1e-1, 1e-2, 5, 5]) * rng.normal(0, 1, size=(nwalk, ndim))

    
    sampler = emcee.EnsembleSampler(nwalk, ndim, log_probability, args=(z_c, DM_obs_c, data_path, H0_min, H0_max, Om_min, Om_max, W_min, W_max, S_host_min, S_host_max, e_mu_min, e_mu_max, SD_min, SD_max))
    sampler.run_mcmc(pos, N_samples, progress=True);

    tau_time = sampler.get_autocorr_time(quiet=True)

    samples_all = sampler.get_chain()
    samples_flat = sampler.get_chain(discard=burn_in, thin=15, flat=True)

    return samples_all, samples_flat, tau_time.max()

###############################################
############### Analyse Results ###############
###############################################


def mcmc_analyze_results(sampler, burn_in=10, thin=15, target_prob=0.6827):
    """Analyze MCMC results (unchanged from original)"""
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    
    params_median = np.median(flat_samples, axis=0)
    params_lower = np.percentile(flat_samples, 50-target_prob*100/2, axis=0)
    params_upper = np.percentile(flat_samples, 50+target_prob*100/2, axis=0)
    
    params_errors = [(params_upper[i] - params_lower[i]) / 2 for i in range(len(params_median))]
    
    return flat_samples, params_median, params_errors


def mcmc_samples_quantiles(flat_samples, target_prob=0.6827, labels=['H0', 'Om', 'w']):
    """Analyze MCMC results (unchanged from original)"""
    
    params_median = np.median(flat_samples, axis=0)
    params_lower = np.percentile(flat_samples, 50-target_prob*100/2, axis=0)
    params_upper = np.percentile(flat_samples, 50+target_prob*100/2, axis=0)
    
    params_errors = [(params_upper[i] - params_lower[i]) / 2 for i in range(len(params_median))]

    for idx, params in enumerate(params_median):
        print(f"For {labels[idx]} the median is: {round(params,2)} with 68% error range: {round(params_errors[idx], 2)}")
    
    return params_median, params_errors

def mcmc_plot_results(samples, param_names, truths, savetitle=None, bins=30, target_prob=0.6827, font_size=15):
    """
    Plot the MCMC results.
    
    Args:
        samples: MCMC samples
        param_names: Names of the parameters
    """
    
    # Create corner plot
    
    fig = corner.corner(
        samples, 
        labels=param_names,
        truths=truths,
        truth_color='tab:orange',
        quantiles=[0.5-target_prob/2, 0.5, 0.5+target_prob/2], ### [0.16, 0.5, 0.84],
        q_ls=['--', ' ', '--'],
        show_titles=True,
        title_kwargs={"fontsize": font_size-2},
        label_kwargs={"fontsize": font_size},
        title_fmt='.3f',
        bins=bins,
        smooth=True,
        color='tab:blue'
    )

    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=font_size)
    
    if savetitle is not None:
        plt.savefig(savetitle+"_corner_plot.pdf", dpi=300, bbox_inches='tight')
    
    
    plt.show()
    plt.close()