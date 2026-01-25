###############################################
### Functions for the cosmological analysis ###
###############################################

from config import *
from cosmo_support import *
# from support import normalise

def log_prob(x, ivar):
    return -0.5 * np.sum(ivar * x**2)


def log_likelihood(theta, x, y, yerr):
    m, b, log_f = theta
    model = m * x + b
    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))



def log_prior(theta):
    m, b, log_f = theta
    if -5.0 < m < 0.5 and 0.0 < b < 10.0 and -10.0 < log_f < 1.0:
        return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)    




def log_prior_frb(theta):
    """
    Calculate the log of the prior probability for a set of parameters.

    Args:
        theta: Array of parameters [F, HOf, sigma_host, e_mu]

    Returns:
        Log prior probability
    """
    hubble, omega, w = theta

    # Define your prior ranges here
    hubble_min, hubble_max = 40, 100 
    omega_min, omega_max = 0.0, 1.0  
    w_min, w_max = -2.0, 0.0 
    
    # Check if parameters are within prior ranges
    if (hubble_min <= hubble <= hubble_max and 
        omega_min <= omega <= omega_max and 
        w_min <= w <= w_max ):
        return 0.0  # Log(1) = 0, flat prior
    else:
        return -np.inf   

def log_likelihood_frb(theta, z_o, DM_o, s_DM_o):
    hubble, omega, w = theta
    model = dispersion_measure(z=z_o, H0=hubble, Om=omega, w=w, alpha=0, f_IGM_0 = 0.84)
    sigma2 = s_DM_o**2
    return -0.5 * np.sum((DM_o - model) ** 2 / sigma2)


def log_probability_frb(theta, z_o, DM_o, s_DM_o):
    lp = log_prior_frb(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_frb(theta, z_o, DM_o, s_DM_o)  



def log_likelihood_frb_Pade(theta, z_o, DM_o, s_DM_o):
    hubble, omega, w = theta
    model = DM_pade_wCDM(z=z_o, H0=hubble, Om=omega, w=w)
    sigma2 = s_DM_o**2
    return -0.5 * np.sum((DM_o - model) ** 2 / sigma2)


def log_probability_frb_Pade(theta, z_o, DM_o, s_DM_o):
    lp = log_prior_frb(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_frb_Pade(theta, z_o, DM_o, s_DM_o)  





###############################################
############### Analyse Results ###############
###############################################


def mcmc_analyze_results(sampler, burn_in=10, thin=15, target_prob=0.6827):
    """Analyze MCMC results (unchanged from original)"""
    flat_samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
    
    params_median = np.median(flat_samples, axis=0)
    params_lower = np.percentile(flat_samples, 50-target_prob*50, axis=0)
    params_upper = np.percentile(flat_samples, 50+target_prob*50, axis=0)
    
    params_errors = [(params_upper[i] - params_lower[i]) / 2 for i in range(len(params_median))]
    
    return flat_samples, params_median, params_errors


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