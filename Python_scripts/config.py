################
### PACKAGES ###
################

### Arrays, vectorisation & special functions ###
import numpy as np
from numpy import linalg as LA
from numpy.random import default_rng
rng = default_rng(seed=1234)
import pandas as pd

from scipy import interpolate
import scipy.constants as const
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.integrate import quad_vec
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.optimize import root_scalar, minimize

### Astropy package
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
import astropy.constants as astc
from astropy.cosmology import Planck18
from astropy.coordinates import SkyCoord
from astropy.cosmology import wCDM

### Plotting
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import cm, ticker
import corner

### MCMC and loops
from tqdm import tqdm
import warnings


################################
### CONSTANTS and PARAMETERS ###
################################

### Defining useful constants (SI)
### Updated values according to Planck18
OMEGA_MATTER = 0.30966 
W_LAMBDA = -1
HUBBLE = 67.66
OMEGA_BARYONS = 0.04897
PI = const.pi
C_LIGHT = const.c
G_NEWTON = const.G
M_PROTON = const.m_p


### FRB params
S_FRB=0.133
EXP_MU=182.937
SIGMA_HOST=0.605
DM_MWHALO=30
f_IGM = 0.84
f_ALPHA = 0
HOF=HUBBLE*f_IGM*OMEGA_BARYONS   ####2.813

### Transformations
KM_2_MPC = 3.240779e-20
HUBBLE_TIME = HUBBLE*KM_2_MPC  # Hubble in 1/seconds
DM_2_PCCM3 = 3.240779e-23  # Multiply with this to get from the SI result (1/m**2) to pc/cm^3
MPC_2_M = 3.085678e+22  


### GWs and FRB errors - used only for test cases
SNR_GW = 8
DL_ERROR_PERC = 10
SIGMA_DM = 105