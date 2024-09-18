import numpy as np
import radvel
import george
from george import kernels
from astropy.table import Table
from datetime import datetime as date
from PyAstronomy.pyasl import foldAt
import matplotlib.pyplot as plt
from matplotlib import cm
from astropy.timeseries import LombScargle
from scipy.stats import truncnorm
import copy
from scipy.stats import norm

### For Juliet ###
def create_priors(params_list, instruments = ['NIRPS']): 
    
    params = []
    dists = []
    hyperps = []
    
    for instrument in instruments:
        for param in params_list:
            
            # Add the parameter's  name
            params.append(param['name'] + '_' + instrument)
            
            # Add the parameter's distribution
            dists.append(param['dist'])

            # Add the parameter's hyperparameters
            if param['dist'] == 'Uniform' or param['dist'] == 'loguniform':
                hyperps.append([param['min'], param['max']])
            
            elif param['dist'] == 'Normal':
                hyperps.append([param['mean'], param['std']])
                
            elif param['dist'] == 'TruncatedNormal':
                hyperps.append([param['mean'], param['std'], param['min'], param['max']])
                
            elif param['dist'] == 'fixed':
                hyperps.append(param['value'])
                
            else: 
                print('Error: Distribution not recognized')
                return None
                
            
    return params, dists, hyperps

def create_common_priors(params_list): 
    
    params = []
    dists = []
    hyperps = []
    
    
    for param in params_list:
        
        # Add the parameter's  name
        params.append(param['name'])
        
        # Add the parameter's distribution
        dists.append(param['dist'])

        # Add the parameter's hyperparameters
        if param['dist'] == 'Uniform' or param['dist'] == 'loguniform':
            hyperps.append([param['min'], param['max']])
        
        elif param['dist'] == 'Normal':
            hyperps.append([param['mean'], param['std']])
            
        elif param['dist'] == 'TruncatedNormal':
            hyperps.append([param['mean'], param['std'], param['min'], param['max']])
            
        elif param['dist'] == 'fixed':
            hyperps.append(param['value'])
            
        else: 
            print('Error: Distribution not recognized')
            return None
                
            
    return params, dists, hyperps

######################################################################################################################
### Generic objects and functions ####################################################################################
######################################################################################################################

class DataLoader:
    """
    A class to load and preprocess radial velocity (RV) and stellar activity data for multiple instruments.
    """

    def __init__(self, data, raw=False, no_FFp=False):
        self.data = data
        self.star_info = self.data.get('star', [{}])  # Default to a list with one empty dictionary
        self.star = self.star_info['name']  # Name of the star
        self.instruments = list(self.data['instruments'])  # List of instruments used for observations
        self.instruments_info = self.data['instruments']  # Information about each instrument
        self.activity_priors = self.data['activity_priors']  # Priors for the activity indicators
        self.RV_priors = self.data['RV_priors']  # Priors for the RV measurements
        self.nplanets = self.data['nplanets']  # Number of planets in the model
        self.fit_ecc = self.data['fit_ecc']  # Whether to fit eccentricity or not
        self.use_indicator = self.data['use_indicator']  # Whether to use activity indicators
        self.version = 'DRS-3-0-0'  # HARPS pipeline version
        self.rjd_bjd_off = 2400000.5  # Offset to convert RJD to Julian Date

        self.tbl = {}  # Dictionary to hold data tables for each instrument
        self.ref_star = {}  # Reference star for each instrument
        self.t_rv, self.y_rv, self.yerr_rv = {}, {}, {}  # RV data for each instrument
        self.i_good_times = {}  # Indices of good observation times for each instrument
        self.d2v, self.sd2v, self.Dtemp, self.sDtemp = {}, {}, {}, {}  # Stellar activity indicators for each instrument
        self.contrast, self.sig_contrast = {}, {}
        self.fwhm, self.sig_fwhm = {}, {}
        self.Dtemp_suffix = {}  # Suffix for temperature difference indicator columns
        self.raw_file_path = {}  # Paths to the raw data files for each instrument
        self.file_path = {}  # Paths to the preprocessed data files for each instrument
        self.med_rv_nirps = {}  # Median RV values for each instrument
        self.t_mod = {}  # Time data for model predictions
        self.raw = raw  # Whether to load raw data or preprocessed data
        self.no_FFP = no_FFp # Whether to use the FFp corrected data or not
        
        # What are we running?
        self.run_activity = self.data['run_activity']  # Whether to run activity analysis
        self.run_RV = self.data['run_RV']  # Whether to run RV analysis
        self.sampler = self.data['sampler']  # Sampler type (MCMC or nested sampling)
        
        # Whether to fit eccentricity or not
        if self.data['fit_ecc']:
            self.n_planet_params = 5  # Number of planet parameters
        else: 
            self.n_planet_params = 3

        self._load_data()  # Load the data

    def _load_data(self):
        """
        Load data for each instrument and preprocess it.
        """
        for instrument in self.instruments:
            self.ref_star[instrument] = self.instruments_info[instrument].get('ref_star', '')  # Reference star for each instrument
            self.Dtemp_suffix[instrument] = self.instruments_info[instrument].get('dtemp_suffix', '')  # Suffix for temperature difference indicator columns
            bin_label = self.instruments_info[instrument].get('bin_label', '')  # Bin label for the data
            pca_label = self.instruments_info[instrument].get('pca_label', '')  # PCA label for the data
            FFp_label = self.instruments_info[instrument].get('FFp_label', '')  # PCA label for the data
            self.t_min = radvel.utils.date2jd(date(*self.instruments_info[instrument].get('start_time', '')))  # Start time for the data
            self.t_max = radvel.utils.date2jd(date(*self.instruments_info[instrument].get('end_time', '')))  # End time for the data
            # Paths to data files
            self.file_path[instrument] = f'CRAPresults/{self.star}/data/lbl{bin_label}_{instrument}_{self.star}_{self.ref_star[instrument]}{pca_label}{FFp_label}_preprocessed.rdb'
            
            # Read data file
            if self.raw:
                self.file_path[instrument] = f'CRAPresults/{self.star}/data/lbl{bin_label}_{instrument}_{self.star}_{self.ref_star[instrument]}{pca_label}.rdb'
                self.tbl[instrument] = Table.read(self.file_path[instrument], format='rdb')  # Read raw data
                self.tbl[instrument]['rjd'] += self.rjd_bjd_off
            elif self.no_FFP:
                self.file_path[instrument] = f'CRAPresults/{self.star}/data/lbl{bin_label}_{instrument}_{self.star}_{self.ref_star[instrument]}{pca_label}_preprocessed.rdb'
                self.tbl[instrument] = Table.read(self.file_path[instrument], format='rdb')  # Read raw data
            else:
                self.file_path[instrument] = f'CRAPresults/{self.star}/data/lbl{bin_label}_{instrument}_{self.star}_{self.ref_star[instrument]}{pca_label}{FFp_label}_preprocessed.rdb'
                self.tbl[instrument] = Table.read(self.file_path[instrument], format='rdb')  # Read raw data
            
            # Select desired times
            self.i_good_times[instrument] = (self.tbl[instrument]['rjd'] > self.t_min) & (self.tbl[instrument]['rjd'] < self.t_max)  # Good observation times
            self.tbl[instrument] = self.tbl[instrument][self.i_good_times[instrument]]  # Filter data by good times
            
            # RV data
            self.t_rv[instrument], self.y_rv[instrument], self.yerr_rv[instrument] = self.tbl[instrument]['rjd'], self.tbl[instrument]['vrad'], self.tbl[instrument]['svrad']
            # Stellar activity indicators
            # Second derivative
            self.d2v[instrument], self.sd2v[instrument] = self.tbl[instrument]['d2v'], self.tbl[instrument]['sd2v']
            
            # Contrast
            try:
                self.contrast[instrument], self.sig_contrast[instrument] = self.tbl[instrument]['contrast'], self.tbl[instrument]['sig_contrast']
            except: 
                print(f'No contrast data for {instrument}')
            try:
                self.fwhm[instrument], self.sig_fwhm[instrument] = self.tbl[instrument]['fwhm'], self.tbl[instrument]['sig_fwhm']
            except:
                print(f'No FWHM data for {instrument}')
            # Temperature difference indicators
            try: 
                self.Dtemp[instrument], self.sDtemp[instrument] = self.tbl[instrument]['DTEMP' + self.Dtemp_suffix[instrument]], self.tbl[instrument]['sDTEMP' + self.Dtemp_suffix[instrument]]
            except: 
                print(f'No DTEMP{self.Dtemp_suffix[instrument]} data for {instrument}')

            # Median of the RVs
            self.med_rv_nirps[instrument] = np.rint(np.median(self.tbl[instrument]['vrad'].data))  # Median RV value
            
            # Time data for model predictions
            self.t_mod[instrument] = np.linspace(np.min(self.t_rv[instrument]), np.max(self.t_rv[instrument]), 1000)  # Model time data


# Functions for photometric data
def bin_tess_data(times, flux, flux_error, num_points):
    # Calculate the number of data points in each bin
    num_bins = len(flux) // num_points
    
    # Calculate the remainder data points
    remainder = len(flux) % num_points
    
    # Create empty arrays to store the binned times, flux, and flux error
    binned_times = np.zeros(num_points)
    binned_flux = np.zeros(num_points)
    binned_flux_error = np.zeros(num_points)
    
    # Bin the data
    for i in range(num_points):
        start = i * num_bins + min(i, remainder)
        end = start + num_bins + (i < remainder)
        binned_times[i] = np.mean(times[start:end])
        binned_flux[i] = np.mean(flux[start:end])
        binned_flux_error[i] = np.sqrt(np.mean(flux_error[start:end]**2))
    
    return binned_times, binned_flux, binned_flux_error

def bkjd_to_rjd(bkjd):
    return bkjd + 2457000.0


# Functions to deal with directories
# From strings to yaml inputs
def transform_shared_params(shared_params_str):
    """
    Transform a string like 'share_params_3,4,5' into a list [3, 4, 5]
    """
    try: 
        params = shared_params_str.split('_')[2]
        return [int(x) for x in params.split(',')]
    except:
        empty_list = [] 
        return empty_list

def transform_nplanets(nplanets_str):
    """
    Transform a string like '2_planet' into an integer 2
    """
    return int(nplanets_str.split('_')[0])

def transform_fit_ecc(fit_ecc_str):
    """
    Transform a string like 'no_ecc' into a boolean False
    """
    return fit_ecc_str == "fit_ecc"

# From yaml inputs to strings

def create_shared_params_str(shared_params_list):
    """
    Transform a list like [3, 4, 5] into a string 'share_params_3,4,5'
    """
    params_str = ','.join(map(str, shared_params_list))
    return f'share_params_{params_str}'

def create_nplanets_str(nplanets_int):
    """
    Transform an integer like 2 into a string '2_planet'
    """
    return f'{nplanets_int}_planet'

def create_fit_ecc_str(fit_ecc_bool):
    """
    Transform a boolean like False into a string 'no_ecc' and True into 'fit_ecc'
    """
    return "fit_ecc" if fit_ecc_bool else "no_ecc"


def gaussian_logp(x: float, mu: float, sigma: float) -> float:
    # Copied from radvel for convenience
    return -0.5 * ((x - mu) / sigma) ** 2 - 0.5 * np.log((sigma**2) * 2.0 * np.pi)

def truncated_normal_logp(x: float, mu: float, sigma: float, lower: float, upper: float) -> float:
    # Calculate the standard deviation for the truncnorm function
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    
    # Calculate the log probability density of the truncated normal distribution
    logp = truncnorm.logpdf(x, a, b, loc=mu, scale=sigma)
    
    return logp


def uniform_logp(x: float, minval: float, maxval: float) -> float:
    # Copied from radvel for convenience
    if x <= minval or x >= maxval:
        #print('Uniform prior violated')
        return -np.inf
    else:
        return -np.log(maxval - minval)


def jeffreys_logp(x: float, minval: float, maxval: float) -> float:
    # Copied from radvel for convenience
    normalization = 1.0 / np.log(maxval / minval)
    if x < minval or x > maxval:
        return -np.inf
    else:
        return np.log(normalization) - np.log(x)


def mod_jeffreys_logp(x: float, minval: float, maxval: float, kneeval: float) -> float:
    normalization = 1.0 / np.log((maxval - kneeval) / (minval - kneeval))
    if (x > maxval) or (x < minval):
        return -np.inf
    else:
        return np.log(normalization) - np.log(x - kneeval)



def gp_log_prior(p: np.ndarray, priors) -> float:
    log_prob = 0.0

    # Mean with wide prior around 0
    if priors['mu']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[0], priors['mu']['min'], priors['mu']['max'])
    elif priors['mu']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[0], priors['mu']['min'], priors['mu']['max'])
    elif priors['mu']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[0], priors['mu']['mean'], priors['mu']['std'])
    elif priors['mu']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[0], priors['mu']['mean'], priors['mu']['std'], priors['mu']['min'], priors['mu']['max'])
    else:
        raise ValueError(f'Distribution not recognized for mu')
    #print(log_prob)
    
    # Log White noise: Uniform
    if priors['noise']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[1], np.log(priors['noise']['min']**2), np.log(priors['noise']['max']**2))
    elif priors['noise']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[1], np.log(priors['noise']['min']**2), np.log(priors['noise']['max']**2))
    elif priors['noise']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[1], np.log(priors['noise']['mean']**2), (2/priors['noise']['mean'])*priors['noise']['std'])
    elif priors['noise']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[1], np.log(priors['noise']['mean']**2), (2/priors['noise']['mean'])*priors['noise']['std'], np.log(priors['noise']['min']**2), np.log(priors['noise']['max']**2))
    else:
        raise ValueError('Distribution not recognized for white noise')
    #print(log_prob)
    
    # Log Variance: Uniform
    if priors['GP_sigma']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[2], np.log(priors['GP_sigma']['min']**2), np.log(priors['GP_sigma']['max']**2))
    elif priors['GP_sigma']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[2], np.log(priors['GP_sigma']['min']**2), np.log(priors['GP_sigma']['max']**2))
    elif priors['GP_sigma']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[2], np.log(priors['GP_sigma']['mean']**2), (2/priors['GP_sigma']['mean'])*priors['GP_sigma']['std'])
    elif priors['GP_sigma']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[2], np.log(priors['GP_sigma']['mean']**2), (2/priors['GP_sigma']['mean'])*priors['GP_sigma']['std'], np.log(priors['GP_sigma']['min']**2), np.log(priors['GP_sigma']['max']**2))
    else: 
        raise ValueError('Distribution not recognized for amplitude')
    #print(log_prob)
    
    # Log metric (lambda**2): Uniform
    #print(p[3])
    if priors['GP_length']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[3], np.log(priors['GP_length']['min']**2), np.log(priors['GP_length']['max']**2))
    elif priors['GP_length']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[3], np.log(priors['GP_length']['min']**2), np.log(priors['GP_length']['max']**2))
    elif priors['GP_length']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[3], np.log(priors['GP_length']['mean']**2), (2/priors['GP_length']['mean'])*priors['GP_length']['std'])
    elif priors['GP_length']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[3], np.log(priors['GP_length']['mean']**2), (2/priors['GP_length']['mean'])*priors['GP_length']['std'], np.log(priors['GP_length']['min']**2), np.log(priors['GP_length']['max']**2))
    else:
        raise ValueError('Distribution not recognized for length scale')
    #print(log_prob)
    
    # Gamma: Jeffreys prior
    if priors['GP_gamma']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[4], priors['GP_gamma']['min'], priors['GP_gamma']['max'])
    elif priors['GP_gamma']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[4], priors['GP_gamma']['min'], priors['GP_gamma']['max'])
    elif priors['GP_gamma']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[4], priors['GP_gamma']['mean'], priors['GP_gamma']['std'])
    elif priors['GP_gamma']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[4], priors['GP_gamma']['mean'], priors['GP_gamma']['std'], priors['GP_gamma']['min'], priors['GP_gamma']['max'])
    else:
        raise ValueError('Distribution not recognized for gamma')
    #print(log_prob)
    
    # Log Period: Uniform
    if priors['GP_Prot']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[5], np.log(priors['GP_Prot']['min']), np.log(priors['GP_Prot']['max']))
    elif priors['GP_Prot']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[5], np.log(priors['GP_Prot']['min']), np.log(priors['GP_Prot']['max']))
    elif priors['GP_Prot']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[5], np.log(priors['GP_Prot']['mean']), (1/priors['GP_Prot']['mean'])*priors['GP_Prot']['std'])
    elif priors['GP_Prot']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[5], np.log(priors['GP_Prot']['mean']), (1/priors['GP_Prot']['mean'])*priors['GP_Prot']['std'], np.log(priors['GP_Prot']['min']), np.log(priors['GP_Prot']['max']))
    else:
        raise ValueError('Distribution not recognized for log period')
    #print(log_prob)
    
    return log_prob

def log_prior_planet(p: np.ndarray, priors, idx = 0, n_planet_params=3) -> float:
    log_prob = 0.0
    
    # Period of planet
    if priors['per'+str(idx+1)]['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[0+idx*n_planet_params], priors['per'+str(idx+1)]['min'], priors['per'+str(idx+1)]['max'])
    elif priors['per'+str(idx+1)]['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[0+idx*n_planet_params], priors['per'+str(idx+1)]['min'], priors['per'+str(idx+1)]['max'])
    elif priors['per'+str(idx+1)]['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[0+idx*n_planet_params], priors['per'+str(idx+1)]['mean'], priors['per'+str(idx+1)]['std'])
    else:
        raise ValueError(f'Distribution not recognized for per'+str(idx+1))

    # Time of conjunction
    if priors['tc'+str(idx+1)]['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[1+idx*n_planet_params], priors['tc'+str(idx+1)]['min'], priors['tc'+str(idx+1)]['max'])
    elif priors['tc'+str(idx+1)]['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[1+idx*n_planet_params], priors['tc'+str(idx+1)]['min'], priors['tc'+str(idx+1)]['max'])
    elif priors['tc'+str(idx+1)]['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[1+idx*n_planet_params], priors['tc'+str(idx+1)]['mean'], priors['tc'+str(idx+1)]['std'])
    else:
        raise ValueError(f'Distribution not recognized for tc'+str(idx+1))
    
    # Fitting eccentricity or not? 
    if n_planet_params == 5: 
        # e
        if priors['e'+str(idx+1)]['distribution'] == 'Uniform':
            log_prob += uniform_logp(p[2+idx*n_planet_params], priors['e'+str(idx+1)]['min'], priors['e'+str(idx+1)]['max'])
        elif priors['e'+str(idx+1)]['distribution'] == 'loguniform':
            log_prob += jeffreys_logp(p[2+idx*n_planet_params], priors['e'+str(idx+1)]['min'], priors['e'+str(idx+1)]['max'])
        elif priors['e'+str(idx+1)]['distribution'] == 'Normal':
            log_prob += gaussian_logp(p[2+idx*n_planet_params], priors['e'+str(idx+1)]['mean'], priors['e'+str(idx+1)]['std'])
        elif priors['e'+str(idx+1)]['distribution'] == 'fixed':
            log_prob += 0.0
        else:
            raise ValueError(f'Distribution not recognized for e'+str(idx))
        
        # w
        if priors['w'+str(idx+1)]['distribution'] == 'Uniform':
            log_prob += uniform_logp(p[3+idx*n_planet_params], priors['w'+str(idx+1)]['min'], priors['w'+str(idx+1)]['max'])
        elif priors['w'+str(idx+1)]['distribution'] == 'loguniform':
            log_prob += jeffreys_logp(p[3+idx*n_planet_params], priors['w'+str(idx+1)]['min'], priors['w'+str(idx+1)]['max'])
        elif priors['w'+str(idx+1)]['distribution'] == 'Normal':
            log_prob += gaussian_logp(p[3+idx*n_planet_params], priors['w'+str(idx+1)]['mean'], priors['w'+str(idx+1)]['std'])
        elif priors['w'+str(idx+1)]['distribution'] == 'fixed':
            log_prob += 0.0
        else:
            raise ValueError(f'Distribution not recognized for w'+str(idx))
        
        # K
        if priors['k'+str(idx+1)]['distribution'] == 'Uniform':
            log_prob += uniform_logp(p[4+idx*n_planet_params], priors['k'+str(idx+1)]['min'], priors['k'+str(idx+1)]['max'])
        elif priors['k'+str(idx+1)]['distribution'] == 'loguniform':
            log_prob += jeffreys_logp(p[4+idx*n_planet_params], priors['k'+str(idx+1)]['min'], priors['k'+str(idx+1)]['max'])
        elif priors['k'+str(idx+1)]['distribution'] == 'Normal':
            log_prob += gaussian_logp(p[4+idx*n_planet_params], priors['k'+str(idx+1)]['mean'], priors['k'+str(idx+1)]['std'])
        elif priors['k'+str(idx+1)]['distribution'] == 'fixed':
            log_prob += 0.0
        else:
            raise ValueError(f'Distribution not recognized for k'+str(idx+1))
        
    else: 
        # K
        if priors['k'+str(idx+1)]['distribution'] == 'Uniform':
            log_prob += uniform_logp(p[2+idx*n_planet_params], priors['k'+str(idx+1)]['min'], priors['k'+str(idx+1)]['max'])
        elif priors['k'+str(idx+1)]['distribution'] == 'loguniform':
            log_prob += jeffreys_logp(p[2+idx*n_planet_params], priors['k'+str(idx+1)]['min'], priors['k'+str(idx+1)]['max'])
        elif priors['k'+str(idx+1)]['distribution'] == 'Normal':
            log_prob += gaussian_logp(p[2+idx*n_planet_params], priors['k'+str(idx+1)]['mean'], priors['k'+str(idx+1)]['std'])
        else:
            raise ValueError(f'Distribution not recognized for k'+str(idx+1))
   
    
    return log_prob
    
############################################
# emcee implementation
############################################
def emcee_act_log_post(p, gp_models, act, data, i_shared) -> float:
    """
    Compute the log-posterior probability for the Gaussian Process (GP) activity model parameters.

    Parameters:
        p (array): Combined parameter vector.
        gp_models (dict): Dictionary of GP models for each instrument.
        act (dict): Dictionary of activity data for each instrument.
        data (object): Data object containing activity priors.
        i_shared (list): Indices of shared GP parameters.

    Returns:
        float: Log-posterior probability of the GP activity model parameters.
    """
    
    # Separate the parameters for each instrument
    separated_params_list, separated_params_dict = separate_gp_params(p, i_shared, gp_models.keys())

    log_prob_tot = 0
    for instrument, gp_model in gp_models.items():
        gp_params = separated_params_dict[instrument]
        log_prob = gp_log_prior(gp_params, data.activity_priors)

        if np.isfinite(log_prob):
            gp_model.gp.set_parameter_vector(gp_params)
            log_prob_tot += log_prob + gp_model.gp.log_likelihood(act[instrument])
        else:
            return -np.inf
    return log_prob_tot


def emcee_log_post(p_combined, model, data, priors, i_shared, num_planets, n_planet_params, n_gp_params = 6) -> float:
    """
    Compute the log-posterior probability for the combined planetary and GP model parameters.

    Parameters:
        p_combined (array): Combined parameter vector.
        model (object): Combined planetary and GP model.
        data (object): Data object containing RV and activity priors.
        priors (dict): Dictionary of priors for each instrument.
        i_shared (list): Indices of shared GP parameters.
        num_planets (int): Number of planets in the model.
        n_planet_params (int): Number of planetary parameters per planet (default is 3).
        n_gp_params (int): Number of GP parameters per instrument (default is 6).

    Returns:
        float: Log-posterior probability of the combined planetary and GP model parameters.
    """
    num_planets = model.num_planets
    planet_params = p_combined[:n_planet_params*num_planets]
    gp_params_combined = p_combined[n_planet_params*num_planets:]
    
    # Separate the GP parameters for each instrument 
    separated_gp_params_list, separated_gp_params_dict = separate_gp_params(gp_params_combined, i_shared, data.instruments)

    # Update the planet and GP parameters in the model
    model.update_params(np.concatenate([planet_params, separated_gp_params_list]))

    # Calculate log prior for the GP hyperparameters and the planet parameters
    gp_log_prob = 0
    planet_log_prior = 0
    
    for instrument, gp_params in separated_gp_params_dict.items():
        gp_log_prob += gp_log_prior(gp_params, priors[instrument])
        
    for p in range(num_planets):
        planet_log_prior += log_prior_planet(planet_params, priors[data.instruments[0]], idx = p, n_planet_params=n_planet_params)
        
        
    if np.isfinite(gp_log_prob) and np.isfinite(planet_log_prior):
        try:
            # print('log likelihood', model.log_likelihood())
            # print('gp log prob', gp_log_prob)
            # print('planet log prior', planet_log_prior)
            log_prob_tot = gp_log_prob + planet_log_prior + model.log_likelihood()
            if np.isnan(log_prob_tot) or not np.isfinite(log_prob_tot):
                return -np.inf
            return log_prob_tot
        except np.linalg.LinAlgError:
            return -np.inf

    return -np.inf


def emcee_log_post_planet_only(planet_params, model, priors, num_planets, n_planet_params) -> float:
    """
    Compute the log-posterior probability for the planetary model parameters only.

    Parameters:
        planet_params (array): Planetary parameter vector.
        model (object): Planetary model.
        data (object): Data object containing RV and activity priors.
        priors (dict): Dictionary of priors for each instrument.
        num_planets (int): Number of planets in the model.

    Returns:
        float: Log-posterior probability of the planetary model parameters.
    """
    
    # Update the planet and GP parameters in the model
    model.update_params(planet_params)

    # Calculate log prior for the planet parameters
    planet_log_prior = 0
        
    for p in range(num_planets):
        planet_log_prior += log_prior_planet(planet_params, priors, idx = p, n_planet_params=n_planet_params)
        
        
    if np.isfinite(planet_log_prior):
        try:
            # print('log likelihood', model.log_likelihood())
            # print('gp log prob', gp_log_prob)
            # print('planet log prior', planet_log_prior)
            log_prob_tot = planet_log_prior + model.log_likelihood()
            if np.isnan(log_prob_tot) or not np.isfinite(log_prob_tot):
                return -np.inf
            return log_prob_tot
        except np.linalg.LinAlgError:
            return -np.inf

    return -np.inf

def emcee_log_post_gp_only(gp_params, model, data, priors, i_shared, activity_indicator = False) -> float:
    """
    Compute the log-posterior probability for the GP model parameters only.

    Parameters:
        gp_params (array): GP parameter vector.
        gp_models (dict): Dictionary of GP models for each instrument.
        act (dict): Dictionary of activity data for each instrument.
        data (object): Data object containing activity priors.
        i_shared (list): Indices of shared GP parameters.

    Returns:
        float: Log-posterior probability of the GP model parameters.
    """
    
    # Separate the parameters for each instrument
    separated_params_list, separated_params_dict = separate_gp_params(gp_params, i_shared, data.instruments)

    log_prob_tot = 0
    for instrument, gp_model in model.gp_models.items():
        gp_params = separated_params_dict[instrument]
        log_prob = gp_log_prior(gp_params, priors[instrument])

        if np.isfinite(log_prob):
            gp_model.gp.set_parameter_vector(gp_params)
            log_prob_tot += log_prob + gp_model.gp.log_likelihood(data.y_rv[instrument])
        else:
            return -np.inf
    return log_prob_tot


def bic_calculator(log_likelihood, n_params, n_data):
    """
    Calculate the Bayesian Information Criterion (BIC).

    Parameters:
        log_likelihood (float): Log-likelihood of the model.
        n_params (int): Number of parameters in the model.
        n_data (int): Number of data points.

    Returns:
        float: BIC value.
    """
    return -2*log_likelihood + n_params*np.log(n_data)


##############################################
# dynesty implementation
##############################################

def dynesty_prior_transform(u, priors, model, i_shared, data, num_planets, n_planet_params, n_gp_params=6):
    """
    Transform the unit cube `u` to the parameter space according to the priors.

    Parameters:
        u (array): Unit cube vector.
        priors (dict): Dictionary of priors for each instrument.
        model (object): Combined planetary and GP model.
        i_shared (list): Indices of shared GP parameters.
        data (object): Data object containing RV and activity priors.
        num_planets (int): Number of planets in the model.
        n_planet_params (int): Number of planetary parameters per planet (default is 3).
        n_gp_params (int): Number of GP parameters per instrument (default is 6).

    Returns:
        array: Transformed parameter vector.
    """
    
    # Check if all elements of u are within [0, 1]
    if not np.all((u >= 0) & (u <= 1)):
        raise ValueError("All elements of u must be within the interval [0, 1]")
    
    num_planets = model.num_planets
    
    priors = copy.deepcopy(priors)
    
    params = np.zeros_like(u)
    
    any_inst = data.instruments[0]
    
    if n_planet_params == 5: 
        planet_params_labels = ['per', 'tc', 'e', 'w', 'k']
    else:
        planet_params_labels = ['per', 'tc', 'k']
        
    # Planet parameters
    for i in range(num_planets):
        for j, param in enumerate(planet_params_labels):
            key = f'{param}{i+1}'
            dist = priors[any_inst][key]['distribution']
            if dist == 'Uniform':
                params[i * n_planet_params + j] = priors[any_inst][key]['min'] + u[i * n_planet_params + j] * (priors[any_inst][key]['max'] - priors[any_inst][key]['min'])
            elif dist == 'Normal':
                params[i * n_planet_params + j] = priors[any_inst][key]['mean'] + norm.ppf(u[i * n_planet_params + j]) * priors[any_inst][key]['std']
            elif dist == 'TruncatedNormal':
                a, b = (priors[any_inst][key]['min'] - priors[any_inst][key]['mean']) / priors[any_inst][key]['std'], (priors[any_inst][key]['max'] - priors[any_inst][key]['mean']) / priors[any_inst][key]['std']
                params[i * n_planet_params + j] = truncnorm.ppf(u[i * n_planet_params + j], a, b, loc=priors[any_inst][key]['mean'], scale=priors[any_inst][key]['std'])
            elif dist == 'loguniform':
                log_min, log_max = np.log(priors[any_inst][key]['min']), np.log(priors[any_inst][key]['max'])
                params[i * n_planet_params + j] = np.exp(log_min + u[i * n_planet_params + j] * (log_max - log_min))
            else:
                raise ValueError(f"Unknown distribution {dist} for parameter {key}")

    # GP hyperparameters
    i_sep = n_planet_params * num_planets
    
    i_param = i_sep
    for k, param in enumerate(['mu', 'noise', 'GP_sigma', 'GP_length', 'GP_gamma', 'GP_Prot']):
        for j, instrument in enumerate(data.instruments):
            dist = priors[instrument][param]['distribution']
            # Changing the priors to log space
            if param == 'noise' or param == 'GP_sigma' or param == 'GP_length': 
                priors[instrument][param]['min'], priors[instrument][param]['max'], priors[instrument][param]['mean'], priors[instrument][param]['std'] = np.log(priors[instrument][param]['min']**2), np.log(priors[instrument][param]['max']**2), np.log(priors[instrument][param]['mean']**2), (2/priors[instrument][param]['mean'])*priors[instrument][param]['std']
            if param == 'GP_Prot': 
                priors[instrument][param]['min'], priors[instrument][param]['max'], priors[instrument][param]['mean'], priors[instrument][param]['std'] = np.log(priors[instrument][param]['min']), np.log(priors[instrument][param]['max']), np.log(priors[instrument][param]['mean']), (1/priors[instrument][param]['mean'])*priors[instrument][param]['std']
            
            # Transform the parameters
            if dist == 'Uniform':
                params[i_param] = priors[instrument][param]['min'] + u[i_param] * (priors[instrument][param]['max'] - priors[instrument][param]['min'])
            elif dist == 'Normal':
                params[i_param] = priors[instrument][param]['mean'] + norm.ppf(u[i_param]) * priors[instrument][param]['std']
            elif dist == 'TruncatedNormal':
                a, b = (priors[instrument][param]['min'] - priors[instrument][param]['mean']) / priors[instrument][param]['std'], (priors[instrument][param]['max'] - priors[instrument][param]['mean']) / priors[instrument][param]['std']
                params[i_param] = truncnorm.ppf(u[i_param], a, b, loc=priors[instrument][param]['mean'], scale=priors[instrument][param]['std'])
            elif dist == 'loguniform':
                log_min, log_max = np.log(priors[instrument][param]['min']), np.log(priors[instrument][param]['max'])
                params[i_param] = np.exp(log_min + u[i_param] * (log_max - log_min))
            else:
                raise ValueError(f"Unknown distribution {dist} for parameter {param}")
            
            i_param+=1
            if k in i_shared: 
                break  
            
    return params

def dynesty_prior_transform_planet_only(u, priors, model, data, num_planets, n_planet_params):
    """
    Transform the unit cube `u` to the parameter space according to the priors for planetary parameters only.

    Parameters:
        u (array): Unit cube vector.
        priors (dict): Dictionary of priors for each instrument.
        model (object): Planetary model.
        data (object): Data object containing RV and activity priors.
        num_planets (int): Number of planets in the model.
        n_planet_params (int): Number of planetary parameters per planet (default is 3).

    Returns:
        array: Transformed parameter vector for planetary parameters only.
    """
    
    
    # Check if all elements of u are within [0, 1]
    if not np.all((u >= 0) & (u <= 1)):
        raise ValueError("All elements of u must be within the interval [0, 1]")
    
    num_planets = model.num_planets
    
    priors = copy.deepcopy(priors)
    
    params = np.zeros_like(u)
    
    if n_planet_params == 5: 
        planet_params_labels = ['per', 'tc', 'e', 'w', 'k']
    else:
        planet_params_labels = ['per', 'tc', 'k']
    
    # Planet parameters
    for i in range(num_planets):
        for j, param in enumerate(planet_params_labels):
            key = f'{param}{i+1}'
            dist = priors[key]['distribution']
            if dist == 'Uniform':
                params[i * n_planet_params + j] = priors[key]['min'] + u[i * n_planet_params + j] * (priors[key]['max'] - priors[key]['min'])
            elif dist == 'Normal':
                params[i * n_planet_params + j] = priors[key]['mean'] + norm.ppf(u[i * n_planet_params + j]) * priors[key]['std']
            elif dist == 'TruncatedNormal':
                a, b = (priors[key]['min'] - priors[key]['mean']) / priors[key]['std'], (priors[key]['max'] - priors[key]['mean']) / priors[key]['std']
                params[i * n_planet_params + j] = truncnorm.ppf(u[i * n_planet_params + j], a, b, loc=priors[key]['mean'], scale=priors[key]['std'])
            elif dist == 'loguniform':
                log_min, log_max = np.log(priors[key]['min']), np.log(priors[key]['max'])
                params[i * n_planet_params + j] = np.exp(log_min + u[i * n_planet_params + j] * (log_max - log_min))
            else:
                raise ValueError(f'Unknown distribution {dist} for parameter {key}')
            
    return params

def dynesty_prior_transform_gp_only(u, priors, model, data, i_shared):
    
    """
    Transform the unit cube `u` to the parameter space according to the priors for GP parameters only.

    Parameters:
        u (array): Unit cube vector.
        priors (dict): Dictionary of priors for each instrument.
        model (object): GP model.
        data (object): Data object containing RV and activity priors.
        i_shared (list): Indices of shared GP parameters.

    Returns:
        array: Transformed parameter vector for GP parameters only.
    """
    
    # Check if all elements of u are within [0, 1]
    if not np.all((u >= 0) & (u <= 1)):
        raise ValueError("All elements of u must be within the interval [0, 1]")
    
    priors = copy.deepcopy(priors)
    
    params = np.zeros_like(u)
    
    # GP hyperparameters
    i_param = 0
    for k, param in enumerate(['mu', 'noise', 'GP_sigma', 'GP_length', 'GP_gamma', 'GP_Prot']):
        for j, instrument in enumerate(data.instruments):
            dist = priors[instrument][param]['distribution']
            # Changing the priors to log space
            if param == 'noise' or param == 'GP_sigma' or param == 'GP_length': 
                priors[instrument][param]['min'], priors[instrument][param]['max'], priors[instrument][param]['mean'], priors[instrument][param]['std'] = np.log(priors[instrument][param]['min']**2), np.log(priors[instrument][param]['max']**2), np.log(priors[instrument][param]['mean']**2), (2/priors[instrument][param]['mean'])*priors[instrument][param]['std']
            if param == 'GP_Prot': 
                priors[instrument][param]['min'], priors[instrument][param]['max'], priors[instrument][param]['mean'], priors[instrument][param]['std'] = np.log(priors[instrument][param]['min']), np.log(priors[instrument][param]['max']), np.log(priors[instrument][param]['mean']), (1/priors[instrument][param]['mean'])*priors[instrument][param]['std']
            
            # Transform the parameters
            if dist == 'Uniform':
                params[i_param] = priors[instrument][param]['min'] + u[i_param] * (priors[instrument][param]['max'] - priors[instrument][param]['min'])
            elif dist == 'Normal':
                params[i_param] = priors[instrument][param]['mean'] + norm.ppf(u[i_param]) * priors[instrument][param]['std']
            elif dist == 'TruncatedNormal':
                a, b = (priors[instrument][param]['min'] - priors[instrument][param]['mean']) / priors[instrument][param]['std'], (priors[instrument][param]['max'] - priors[instrument][param]['mean']) / priors[instrument][param]['std']
                params[i_param] = truncnorm.ppf(u[i_param], a, b, loc=priors[instrument][param]['mean'], scale=priors[instrument][param]['std'])
            elif dist == 'loguniform':
                log_min, log_max = np.log(priors[instrument][param]['min']), np.log(priors[instrument][param]['max'])
                params[i_param] = np.exp(log_min + u[i_param] * (log_max - log_min))
            else:
                raise ValueError(f"Unknown distribution {dist} for parameter {param}")
            
            i_param+=1
            if k in i_shared: 
                break
    
    return params


def dynesty_log_likelihood(p_combined, model, i_shared, data, num_planets, n_planet_params, n_gp_params=6):
    
    """
    Compute the log likelihood for the combined planetary and GP model parameters.

    Parameters:
        p_combined (array): Combined parameter vector.
        model (object): Combined planetary and GP model.
        i_shared (list): Indices of shared GP parameters.
        data (object): Data object containing RV and activity data.
        num_planets (int): Number of planets in the model.
        n_planet_params (int): Number of planetary parameters per planet (default is 3).
        n_gp_params (int): Number of GP parameters per instrument (default is 6).

    Returns:
        float: Log likelihood of the combined planetary and GP model parameters.
    """

    
    
    # Update the model paramters
    planet_params = p_combined[:n_planet_params*num_planets]
    gp_params_combined = p_combined[n_planet_params*num_planets:]
    
    # Separate the GP parameters for each instrument
    separated_gp_params_list, separated_gp_params_dict = separate_gp_params(gp_params_combined, i_shared, data.instruments)

    # Update the planet and GP parameters in the model
    model.update_params(np.concatenate([planet_params, separated_gp_params_list]))
    
    try:
        log_likelihood = model.log_likelihood()
        if np.isnan(log_likelihood) or not np.isfinite(log_likelihood):
            return -np.inf
        return log_likelihood
    except np.linalg.LinAlgError:
        return -np.inf
    
def dynesty_log_likelihood_planet_only(planet_params, model):
    
    """
    Compute the log likelihood for the planetary model parameters only.

    Parameters:
        planet_params (array): Planetary parameter vector.
        model (object): Planetary model.

    Returns:
        float: Log likelihood of the planetary model parameters.
    """
    
    # Update the model paramters
    model.update_params(planet_params)
    
    try:
        log_likelihood = model.log_likelihood()
        if np.isnan(log_likelihood) or not np.isfinite(log_likelihood):
            return -np.inf
        return log_likelihood
    except np.linalg.LinAlgError:
        return -np.inf

 
def dynesty_log_likelihood_gp_only(gp_params, model, data, i_shared):
    
    """
    Compute the log likelihood for the GP model parameters only.

    Parameters:
        gp_params (array): GP parameter vector.
        gp_models (dict): Dictionary of GP models for each instrument.
        act (dict): Dictionary of activity data for each instrument.
        i_shared (list): Indices of shared GP parameters.

    Returns:
        float: Log likelihood of the GP model parameters.
    """
    
    # Separate the parameters for each instrument
    separated_params_list, separated_params_dict = separate_gp_params(gp_params, i_shared, data.instruments)
    
    log_likelihood = 0
    for instrument, gp_model in model.gp_models.items():
        gp_params = separated_params_dict[instrument]
        gp_model.gp.set_parameter_vector(gp_params)
        log_likelihood += gp_model.gp.log_likelihood(data.y_rv[instrument])
    
    return log_likelihood


#############################################################
#############################################################


def get_max_likelihood_params(post_samples, log_post_probs):
    """
    Get the maximum likelihood parameters from posterior samples.
    
    Args:
    - post_samples (np.ndarray): Array of posterior samples (N_samples, N_parameters).
    - log_post_probs (np.ndarray): Array of log-posterior probabilities (N_samples,).
    
    Returns:
    - max_likelihood_params (np.ndarray): Parameters corresponding to the maximum log-posterior probability.
    """
    # Identify the index of the maximum log-posterior probability
    max_log_prob_idx = np.argmax(log_post_probs)
    
    # Extract the parameters corresponding to the maximum log-posterior probability
    max_likelihood_params = post_samples[max_log_prob_idx]
    
    return max_likelihood_params


def juliet_to_george(p):
    """
    Convert parameters from Juliet format to George format.

    Parameters:
        p (list): List of parameters in Juliet format.
            [mu, noise, GP_sigma, GP_length, GP_gamma, GP_Prot]

    Returns:
        list: List of parameters in George format.
            [mu, log(noise^2), log(GP_sigma^2), log(GP_length^2), GP_gamma, log(GP_Prot)]
    """
    return [p[0], np.log(p[1]**2), np.log(p[2]**2), 
                      np.log(p[3]**2), p[4], np.log(p[5])]


def separate_gp_params(comb_params, i_shared, instruments):
    """
    Separate the combined parameter vector into GP parameters for each instrument.

    Parameters:
        comb_params (list): Combined parameter vector.
        i_shared (list): Indices of shared GP parameters.
        instruments (list): List of instruments.

    Returns:
        tuple: (params_list, params_dict)
            - params_list (list): List of separated GP parameters for all instruments.
            - params_dict (dict): Dictionary of GP parameters for each instrument.
    """
    params_dict = {instrument: [] for instrument in instruments}
    params_list = []
    
    param_id = 0 # Index of the parameter
    i = 0 # Index of the combined parameters
    while i < len(comb_params):
        if param_id in i_shared:
            # Shared parameters, assign to all instruments
            for instrument in instruments:
                params_dict[instrument].append(comb_params[i])
            i += 1
        else:
            # Separate parameters for each instrument
            for instrument in instruments:
                params_dict[instrument].append(comb_params[i])
                i += 1
        
        param_id += 1
    
    # Create params_list
    for instrument in instruments:
        params_list += params_dict[instrument]
    
    return params_list, params_dict

def separate_gp_params_samples(post_samples, i_shared, instruments):
    """
    Separate the post samples into a dictionary with samples for each instrument.
    
    Args:
        post_samples (np.ndarray): Post samples with shape (N_samples, N_walkers, ndim).
        i_shared (list): Indices of shared parameters.
        instruments (list): List of instrument names.
        
    Returns:
        dict: Dictionary containing the samples of the parameters for each instrument.
    """
    N_samples, N_walkers, ndim = post_samples.shape
    total_samples = N_samples * N_walkers
    params_dict = {instrument: {} for instrument in instruments}
    shared_samples = {}

    param_id = 0 # Index of the parameter in the combined vector
    i = 0 # Index in the post_samples array
    while i < ndim:
        if param_id in i_shared:
            # Shared parameters, assign the same samples to all instruments
            shared_samples[param_id] = post_samples[:, :, i].flatten()
            for instrument in instruments:
                params_dict[instrument][param_id] = shared_samples[param_id]
            i += 1
        else:
            # Separate parameters for each instrument
            for instrument in instruments:
                params_dict[instrument][param_id] = post_samples[:, :, i].flatten()
                i += 1
        
        param_id += 1

    return params_dict


def generate_param_names(param_names, i_shared, instruments):
    """
    Generate a list of parameter names where shared parameters remain unchanged 
    and non-shared parameters are suffixed with their respective instrument names.

    Args:
        param_names (list): List of parameter names.
        i_shared (list): Indices of shared parameters.
        instruments (list): List of instrument names.

    Returns:
        list: List of parameter names with appropriate suffixes for non-shared parameters.
    """
    combined_param_names = []

    for i, param in enumerate(param_names):
        if i in i_shared:
            # Shared parameter
            combined_param_names.append(param)
        else:
            # Non-shared parameter, append with instrument suffix
            for instrument in instruments:
                combined_param_names.append(f"{param}_{instrument}")
    
    return combined_param_names



# Class to deal with parameters
class params_vector:
    
    '''
    Class to manage parameter vectors for planetary and GP models.

    p = [per1, tc1, k1, ... (other planets), mu_NIRPS, noise_NIRPS, amp_NIRPS, coherence_length_NIRPS, gamma_NIRPS, Prot_NIRPS, ... (other instruments)]
    i_shared = Indices of shared parameters (e.g., GP parameters that are common across instruments).
    '''
    
    def __init__(self, p, instruments, i_shared, num_planets=1, n_planet_params=3, n_gp_params=6, gp_only=False):
        self.p = p
        self.instruments = instruments
        self.i_shared = i_shared
        self.num_planets = num_planets
        self.n_planet_params = n_planet_params
        self.n_gp_params = n_gp_params
        
        if gp_only==False:
            self.planet_params = self.p[:self.n_planet_params * self.num_planets]
            self.gp_params = self.p[self.n_planet_params * self.num_planets:]
            
        else:
            self.planet_params = []
            self.gp_params = self.p
        
        # Initialize dictionaries for separated GP parameters
        self.gp_params_dict = {instrument: [] for instrument in self.instruments}

        # Fill in the separated GP parameters
        for n, instrument in enumerate(self.instruments):
            start_idx = n * self.n_gp_params
            end_idx = (n + 1) * self.n_gp_params
            self.gp_params_dict[instrument] = self.gp_params[start_idx:end_idx]

    def __getitem__(self, i):
        return self.p[i]

    def __setitem__(self, i, value):
        self.p[i] = value

    def __len__(self):
        return len(self.p)
    
    def combine(self):
        '''
        Convert separated parameter vector to combined version.
        
        Returns:
            combined_params (list): Combined parameter vector.
        '''

        # Create the combined parameter vector
        combined_params = self.planet_params + []
        for i in range(self.n_gp_params):
            if i in self.i_shared:
                combined_params.append(self.gp_params_dict[self.instruments[0]][i])
            else:
                for instrument in self.instruments:
                    combined_params.append(self.gp_params_dict[instrument][i])
                
        return combined_params
    

class QP_GP_Model:
    def __init__(self, gp_params, t, y, yerr):
        
        self.gp_params = gp_params
        self.t = t
        self.y = y
        self.yerr = yerr
        self.jitter = 1e-6 # Minimum jitter to avoid numerical issues

        ker_sqexp = kernels.ExpSquaredKernel(metric=np.exp(gp_params[3]))
        ker_per = kernels.ExpSine2Kernel(gamma=gp_params[4], log_period=gp_params[5])
        kernel = np.exp(gp_params[2]) * ker_sqexp * ker_per

        self.gp = george.GP(
            kernel,
            mean=gp_params[0],
            fit_mean=True,
            white_noise=gp_params[1],
            fit_white_noise=True,
        )
        self.gp.compute(t, yerr=yerr+self.jitter)

    def predict(self, y, t):
        
        gp_values = self.gp.predict(y, t)
        return gp_values[0]
    
    def update_params(self, gp_params):
        self.gp.set_parameter_vector(gp_params)
        self.gp.compute(self.t, yerr=self.yerr+self.jitter)
        
    def log_likelihood(self):
        return self.gp.log_likelihood(self.y)


class QP_GP_Model_Group:
    def __init__(self, p, t_dict, y_dict, yerr_dict, n_gp_params=6):
        """
        Initialize a group of QP_GP_Model instances, one for each instrument.
        
        Parameters:
        gp_params_dict (dict): Dictionary of GP parameters for each instrument.
        t_dict (dict): Dictionary of time arrays for each instrument.
        y_dict (dict): Dictionary of RV measurements for each instrument.
        yerr_dict (dict): Dictionary of RV measurement errors for each instrument.
        """
        self.p = p
        self.t_dict = t_dict
        self.y_dict = y_dict
        self.yerr_dict = yerr_dict
        self.models = {}
        self.instruments = t_dict.keys()
        self.n_gp_params = n_gp_params
        self.gp_params = {}
        self.gp_models = {}
        
        # Initialize GP models for each instrument
        for n, instrument in enumerate(self.instruments):
            self.gp_params[instrument] = self.p[n * n_gp_params:(n + 1) * n_gp_params]
            self.gp_models[instrument] = QP_GP_Model(self.gp_params[instrument], self.t_dict[instrument],
                                                     self.y_dict[instrument], self.yerr_dict[instrument])
    
    def predict(self, y, t):

        predictions = {}
        for instrument in t:
            # Get predictions from GP
            predictions[instrument] = self.gp_models[instrument].predict(y[instrument], t[instrument])

        return predictions
    
    def update_params(self, gp_params):
        """
        Update the GP parameters for each instrument.
        
        Parameters:
        gp_params (array): Parameter vector for all instruments.
        """
        # Update GP model parameters for each instrument
        for n, instrument in enumerate(self.instruments):
            self.gp_params[instrument] = gp_params[n * self.n_gp_params:(n + 1) * self.n_gp_params]
            self.gp_models[instrument].gp.set_parameter_vector(self.gp_params[instrument])
    
    def log_likelihood(self):
        """
        Calculate the combined log likelihood for all instruments.
        
        Returns:
        float: Combined log likelihood.
        """
        log_likelihood = 0
        
        for instrument, model in self.models.items():
            log_likelihood += model.log_likelihood()
        
        return log_likelihood


class Planet_Model:
    def __init__(self, planet_params, data, num_planets, n_planet_params=3):
        self.planet_params = planet_params
        self.num_planets = num_planets
        self.data = data
        self.n_planet_params = n_planet_params
        
        self.params = radvel.Parameters(num_planets, basis="per tc e w k")
        index = 0
        for i in range(1, num_planets + 1):
            self.params['per'+str(i)].value = planet_params[index]
            self.params['tc'+str(i)].value = planet_params[index + 1]
            if n_planet_params == 5:
                self.params['e'+str(i)].value = planet_params[index+2]  
                self.params['w'+str(i)].value = planet_params[index+3]  
                self.params['k'+str(i)].value = planet_params[index + 4]
            else: 
                self.params['e'+str(i)].value = 0.0 # Fixed
                self.params['w'+str(i)].value = 1.57 # Fixed
                self.params['k'+str(i)].value = planet_params[index + 2]
            index += n_planet_params  # Move to the next set of planet parameters
            
        self.rad_model = radvel.RVModel(self.params)

    def predict(self, t):
        return self.rad_model(t)
    
    def update_params(self, p):
        index = 0
        for i in range(1, self.num_planets + 1):
            self.params['per'+str(i)].value = p[index]
            self.params['tc'+str(i)].value = p[index + 1]
            if self.n_planet_params == 5:
                self.params['e'+str(i)].value = p[index+2]  # Fixed for now (TODO)
                self.params['w'+str(i)].value = p[index+3]  # Fixed for now (TODO)
                self.params['k'+str(i)].value = p[index + 4]
            else: 
                self.params['e'+str(i)].value = 0.0 # Fixed
                self.params['w'+str(i)].value = 1.57 # Fixed
                self.params['k'+str(i)].value = p[index + 2]
            index += self.n_planet_params  # Move to the next set of planet parameters
            
        self.rad_model = radvel.RVModel(self.params)
        
    def log_likelihood(self):
        log_like = 0
        for instrument in self.data.instruments:
            residuals = self.rad_model(self.data.t_rv[instrument]) - self.data.y_rv[instrument]
            log_like += -0.5 * np.sum((residuals / self.data.yerr_rv[instrument])**2 + np.log(2 * np.pi * self.data.yerr_rv[instrument]**2))
            
        return log_like

class Planet_GP_Model:
    def __init__(self, p, data, num_planets=1, n_planet_params=3, n_gp_params=6):
        # Parameter vector
        self.p = p

        self.data = data

        # Number of planets and parameters
        self.num_planets = num_planets
        self.instruments = data.instruments
        self.n_planet_params = n_planet_params
        self.n_gp_params = n_gp_params

        # Separation index between planet and GP parameters
        self.i_sep = n_planet_params * num_planets

        # Isolate the planet parameters
        planet_params = np.array(p[:self.i_sep])
        self.radvel_model = Planet_Model(planet_params, data, num_planets, n_planet_params=n_planet_params)

        # Dictionaries to hold GP parameters and models for each instrument
        self.gp_params = {}
        self.gp_models = {}

        # Initialize GP models for each instrument
        for n, instrument in enumerate(self.instruments):
            self.gp_params[instrument] = p[self.i_sep + n * n_gp_params: self.i_sep + (n + 1) * n_gp_params]
            self.gp_models[instrument] = QP_GP_Model(self.gp_params[instrument], self.data.t_rv[instrument],
                                                     self.data.y_rv[instrument], self.data.yerr_rv[instrument])

    def predict(self, y, t, return_components=False):

        predictions = {}
        for instrument in t:
            # Get predictions from GP and planetary models
            gp_values = self.gp_models[instrument].predict(y[instrument], t[instrument])
            planet_values = self.radvel_model.predict(t[instrument])
            
            if return_components:
                predictions[instrument] = {'planet': planet_values, 'GP': gp_values}
            else: 
                predictions[instrument] = planet_values + gp_values

        return predictions

    def log_likelihood(self):
        """
        Compute the combined log likelihood for the GP and planetary models.

        Returns:
            float: Combined log likelihood.
        """
        combined_log_likelihood = 0

        for instrument in self.instruments:
            
            # Predict planetary model
            planet_predictions = self.radvel_model.predict(self.data.t_rv[instrument])
            
            # Compute residuals of data minus planetary model
            residuals_planet = self.data.y_rv[instrument] - planet_predictions
            
            # Compute GP log likelihood for the residuals of the planetary model
            self.gp_models[instrument].gp.compute(self.data.t_rv[instrument], self.data.yerr_rv[instrument])
            gp_log_likelihood = self.gp_models[instrument].gp.log_likelihood(residuals_planet)
            
            # # Predict GP model
            # gp_predictions = self.gp_models[instrument].predict(self.t_rv_dict[instrument])
            
            # # Compute residuals of data minus GP model
            # residuals_gp = self.y_rv_dict[instrument] - gp_predictions
            
            # # Compute the log likelihood for the planetary model using the residuals of data minus GP model
            # planet_log_likelihood = -0.5 * np.sum((residuals_gp / self.yerr_rv_dict[instrument])**2 + np.log(2 * np.pi * self.yerr_rv_dict[instrument]**2))
            
            # Add both GP and planetary log likelihoods to the combined log likelihood
            combined_log_likelihood += gp_log_likelihood #+ planet_log_likelihood

        return combined_log_likelihood
        

    def update_params(self, p):
        self.p = p

        # Update planetary model parameters
        planet_params = np.array(p[:self.i_sep])
        self.radvel_model.update_params(planet_params)

        # Update GP model parameters for each instrument
        for n, instrument in enumerate(self.instruments):
            self.gp_params[instrument] = p[self.i_sep + n * self.n_gp_params: self.i_sep + (n + 1) * self.n_gp_params]
            self.gp_models[instrument].gp.set_parameter_vector(self.gp_params[instrument])

    
    
    
    
########################## Plotting functions ######################################

def plot_lombscargle_periodograms(t_rv_dict, y_rv_dict, yerr_rv_dict, target_fap=0.01, max_frequency=1.0, combined=False, file_path = None):
    """
    Plots Lomb-Scargle periodograms along with the window functions for multiple instruments.
    
    Parameters:
    - t_rv_dict: dict, time data for each instrument
    - y_rv_dict: dict, radial velocity data for each instrument
    - yerr_rv_dict: dict, errors in radial velocity data for each instrument
    - target_fap: float, desired false alarm probability level (default is 0.01)
    - max_frequency: float, maximum frequency for the periodogram (default is 1.0)
    - combined: bool, whether to plot a single periodogram for the combined dataset (default is False)

    Returns:
    - None
    """
    instruments = t_rv_dict.keys()
    colors = cm.coolwarm(np.linspace(0, 1, len(instruments)))

    plt.figure(figsize=(10, 6))

    if combined:
        combined_times = np.concatenate([t_rv_dict[instrument] for instrument in instruments])
        combined_y_rv = np.concatenate([y_rv_dict[instrument] for instrument in instruments])
        combined_yerr_rv = np.concatenate([yerr_rv_dict[instrument] for instrument in instruments])
        
        # Calculate the Lomb-Scargle periodogram for the combined dataset
        ls_combined = LombScargle(combined_times, combined_y_rv, dy=combined_yerr_rv)
        freq_combined, power_combined = ls_combined.autopower(maximum_frequency=max_frequency)
        period_combined = 1 / freq_combined
        
        # Calculate the false alarm probability level for the combined dataset
        fap_combined = ls_combined.false_alarm_level(target_fap)
        
        # Calculate the window function for the combined dataset
        ls_window_combined = LombScargle(combined_times, np.ones_like(combined_y_rv), dy=combined_yerr_rv, fit_mean=False, center_data=False)
        power_window_combined = ls_window_combined.power(freq_combined)
        
        # Plot the periodogram and window function for the combined dataset
        plt.plot(period_combined, power_combined, 'black', label="Combined Periodogram")
        plt.plot(period_combined, power_window_combined, "red", linestyle='--', label="Combined Window Function")
        plt.axhline(fap_combined, linestyle="--", color="blue", label=f"{target_fap * 100}% FA level", alpha=0.5)
    else:
        for idx, instrument in enumerate(instruments):
            times = t_rv_dict[instrument]
            y_rv = y_rv_dict[instrument]
            yerr_rv = yerr_rv_dict[instrument]

            # Calculate the Lomb-Scargle periodogram
            ls = LombScargle(times, y_rv, dy=yerr_rv)
            freq, power = ls.autopower(maximum_frequency=max_frequency)
            period = 1 / freq
            
            # Calculate the false alarm probability level
            fap = ls.false_alarm_level(target_fap)
            
            # Calculate the window function
            ls_window = LombScargle(times, np.ones_like(y_rv), dy=yerr_rv, fit_mean=False, center_data=False)
            power_window = ls_window.power(freq)
            
            # Plot the periodogram and window function
            plt.plot(period, power, color=colors[idx], label=f'{instrument} Periodogram')
            plt.plot(period, power_window, color=colors[idx], linestyle='--', label=f'{instrument} Window Function')
        
            plt.axhline(fap, linestyle="--", color="blue", label=f"{target_fap * 100}% FA level", alpha=0.5)
    
    # Set plot labels and title
    plt.xlabel('Period [days]')
    plt.ylabel('Power')
    plt.title('Lomb-Scargle Periodograms')
    plt.legend()
    plt.xscale('log')
    plt.xlim(0, max(period_combined) if combined else max(period))
    plt.grid(True)
    if file_path is not None:
        plt.savefig(file_path)
    plt.show()