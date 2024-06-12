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
### For Goerge + Radvel ##############################################################################################
######################################################################################################################

class DataLoader:
    def __init__(self, data, raw=False):
        self.data = data
        self.star_info = self.data.get('star', [{}])  # Default to a list with one empty dictionary
        self.star = self.star_info['name']
        self.instruments = list(self.data['instruments'])
        self.instruments_info = self.data['instruments']
        self.activity_priors = self.data['activity_priors']
        self.RV_priors = self.data['RV_priors']
        self.nplanets = self.data['nplanets']
        self.use_indicator = self.data['use_indicator']
        self.version = 'DRS-3-0-0'  # HARPS pipeline version
        self.rjd_rjd_off = 2400000.5

        self.tbl = {}
        self.ref_star = {}  
        self.t_rv, self.y_rv, self.yerr_rv = {}, {}, {}
        self.i_good_times = {}
        self.d2v, self.sd2v, self.Dtemp, self.sDtemp = {}, {}, {}, {}
        self.Dtemp_suffix = {}
        self.raw_file_path = {}
        self.file_path = {}
        self.med_rv_nirps = {}
        self.t_mod = {}
        self.no_Dtemp = self.data['no_Dtemp']
        self.raw = raw
        
        # What are we running?
        self.run_activity = self.data['run_activity']
        self.run_RV = self.data['run_RV']
        self.sampler = self.data['sampler'] # MCMC or nested sampling

        self._load_data()

    def _load_data(self):
        for instrument in self.instruments:
            self.ref_star[instrument] = self.instruments_info[instrument].get('ref_star', '')
            self.Dtemp_suffix[instrument] = self.instruments_info[instrument].get('dtemp_suffix', '')
            bin_label = self.instruments_info[instrument].get('bin_label', '')
            pca_label = self.instruments_info[instrument].get('pca_label', '')
            self.t_min = radvel.utils.date2jd(date(*self.instruments_info[instrument].get('start_time', '')))
            self.t_max = radvel.utils.date2jd(date(*self.instruments_info[instrument].get('end_time', '')))

            self.raw_file_path[instrument] = f'stars/{self.star}/data/lbl{bin_label}_{instrument}_{self.star}_{self.ref_star[instrument]}{pca_label}.rdb'
            self.file_path[instrument] = f'stars/{self.star}/data/lbl{bin_label}_{instrument}_{self.star}_{self.ref_star[instrument]}{pca_label}_preprocessed.rdb'
            if self.raw == True: 
                self.tbl[instrument] = Table.read(self.raw_file_path[instrument], format='rdb')
                self.tbl[instrument]['rjd'] += self.rjd_rjd_off
            else: 
                self.tbl[instrument] = Table.read(self.file_path[instrument], format='rdb')
            self.tbl[instrument]['vrad'] -= np.median(self.tbl[instrument]['vrad'])

            # Select desired times
            self.i_good_times[instrument] = (self.tbl[instrument]['rjd'] > self.t_min) & (self.tbl[instrument]['rjd'] < self.t_max)
            self.tbl[instrument] = self.tbl[instrument][self.i_good_times[instrument]]
            
            # RV data    
            self.t_rv[instrument], self.y_rv[instrument], self.yerr_rv[instrument] = self.tbl[instrument]['rjd'], self.tbl[instrument]['vrad'], self.tbl[instrument]['svrad']
            
            # Stellar activity indicators
            self.d2v[instrument], self.sd2v[instrument] = self.tbl[instrument]['d2v'] / np.max(self.tbl[instrument]['d2v']), np.abs(self.tbl[instrument]['sd2v'] / np.max(self.tbl[instrument]['d2v']))
            self.d2v[instrument] -= np.median(self.d2v[instrument])
            
            if not self.no_Dtemp:
                self.Dtemp[instrument], self.sDtemp[instrument] = self.tbl[instrument]['DTEMP' + self.Dtemp_suffix[instrument]], self.tbl[instrument]['sDTEMP' + self.Dtemp_suffix[instrument]]
                self.Dtemp[instrument] -= np.median(self.Dtemp[instrument])

            # Median of the RVs
            self.med_rv_nirps[instrument] = np.rint(np.median(self.tbl[instrument]['vrad'].data))
            self.t_mod[instrument] = np.linspace(np.min(self.t_rv[instrument]), np.max(self.t_rv[instrument]), 1000)



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
        log_prob += gaussian_logp(p[1], np.log(priors['noise']['mean']**2), np.log(priors['noise']['std']**2))
    elif priors['noise']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[1], np.log(priors['noise']['mean']**2), np.log(priors['noise']['std']**2), np.log(priors['noise']['min']**2), np.log(priors['noise']['max']**2))
    else:
        raise ValueError('Distribution not recognized for white noise')
    #print(log_prob)
    
    # Log Variance: Uniform
    if priors['GP_sigma']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[2], np.log(priors['GP_sigma']['min']**2), np.log(priors['GP_sigma']['max']**2))
    elif priors['GP_sigma']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[2], np.log(priors['GP_sigma']['min']**2), np.log(priors['GP_sigma']['max']**2))
    elif priors['GP_sigma']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[2], np.log(priors['GP_sigma']['mean']**2), np.log(priors['GP_sigma']['std']**2))
    elif priors['GP_sigma']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[2], np.log(priors['GP_sigma']['mean']**2), np.log(priors['GP_sigma']['std']**2), np.log(priors['GP_sigma']['min']**2), np.log(priors['GP_sigma']['max']**2))
    else: 
        raise ValueError('Distribution not recognized for amplitude')
    #print(log_prob)
    
    # Log metric (lambda**2): Uniform
    if priors['GP_length']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[3], np.log(priors['GP_length']['min']**2), np.log(priors['GP_length']['max']**2))
    elif priors['GP_length']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[3], np.log(priors['GP_length']['min']**2), np.log(priors['GP_length']['max']**2))
    elif priors['GP_length']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[3], np.log(priors['GP_length']['mean']**2), np.log(priors['GP_length']['std']**2))
    elif priors['GP_length']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[3], np.log(priors['GP_length']['mean']**2), np.log(priors['GP_length']['std']**2), np.log(priors['GP_length']['min']**2), np.log(priors['GP_length']['max']**2))
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
        log_prob += gaussian_logp(p[5], np.log(priors['GP_Prot']['mean']), np.log(priors['GP_Prot']['std']))
    elif priors['GP_Prot']['distribution'] == 'TruncatedNormal':
        log_prob += truncated_normal_logp(p[5], np.log(priors['GP_Prot']['mean']), np.log(priors['GP_Prot']['std']), np.log(priors['GP_Prot']['min']), np.log(priors['GP_Prot']['max']))
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
   
    
    # # Secosw
    # if priors['secosw'+str(idx)]['distribution'] == 'Uniform':
    #     log_prob += uniform_logp(p[2], priors['secosw'+str(idx)]['min'], priors['secosw'+str(idx)]['max'])
    # elif priors['secosw'+str(idx)]['distribution'] == 'loguniform':
    #     log_prob += jeffreys_logp(p[2], priors['secosw'+str(idx)]['min'], priors['secosw'+str(idx)]['max'])
    # elif priors['secosw'+str(idx)]['distribution'] == 'Normal':
    #     log_prob += gaussian_logp(p[2], priors['secosw'+str(idx)]['mean'], priors['secosw'+str(idx)]['std'])
    # elif priors['secosw'+str(idx)]['distribution'] == 'fixed':
    #     log_prob += 0.0
    # else:
    #     raise ValueError(f'Distribution not recognized for secosw'+str(idx))
    
    # # Sesinw
    # if priors['sesinw'+str(idx)]['distribution'] == 'Uniform':
    #     log_prob += uniform_logp(p[3], priors['sesinw'+str(idx)]['min'], priors['sesinw'+str(idx)]['max'])
    # elif priors['sesinw'+str(idx)]['distribution'] == 'loguniform':
    #     log_prob += jeffreys_logp(p[3], priors['sesinw'+str(idx)]['min'], priors['sesinw'+str(idx)]['max'])
    # elif priors['sesinw'+str(idx)]['distribution'] == 'Normal':
    #     log_prob += gaussian_logp(p[3], priors['sesinw'+str(idx)]['mean'], priors['sesinw'+str(idx)]['std'])
    # elif priors['sesinw'+str(idx)]['distribution'] == 'fixed':
    #     log_prob += 0.0
    # else:
    #     raise ValueError(f'Distribution not recognized for sesinw'+str(idx))
    
    # K
    if priors['k'+str(idx+1)]['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[2+idx*n_planet_params], priors['k'+str(idx+1)]['min'], priors['k'+str(idx+1)]['max'])
    elif priors['k'+str(idx+1)]['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[2+idx*n_planet_params], priors['k'+str(idx+1)]['min'], priors['k'+str(idx+1)]['max'])
    elif priors['k'+str(idx+1)]['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[2+idx*n_planet_params], priors['k'+str(idx+1)]['mean'], priors['k'+str(idx+1)]['std'])
    elif priors['k'+str(idx+1)]['distribution'] == 'fixed':
        log_prob += 0.0
    else:
        raise ValueError(f'Distribution not recognized for k'+str(idx+1))
   
    
    return log_prob
    
############################################
# emcee implementation
############################################
def emcee_act_log_post(p, gp_models, act, data, i_shared) -> float:
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


def emcee_log_post(p_combined, model, data, priors, i_shared, num_planets, n_planet_params=3, n_gp_params = 6) -> float:
    
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
        planet_log_prior += log_prior_planet(planet_params, priors[data.instruments[0]], idx = p)
        
        
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

##############################################
##############################################

##############################################
# dynesty implementation
def dynesty_prior_transform(u, priors, model, i_shared, data, num_planets, n_planet_params=3, n_gp_params=6):
    """
    Transform the unit cube `u` to the parameter space according to the priors.
    """
    num_planets = model.num_planets
    planet_params = u[:n_planet_params*num_planets]
    gp_params_combined = u[n_planet_params*num_planets:]
    
    priors = copy.deepcopy(priors)
    
    params = np.zeros_like(u)
    
    any_inst = data.instruments[0]
    # Planet parameters
    for i in range(num_planets):
        for j, param in enumerate(['per', 'tc', 'k']):
            key = f'{param}{i+1}'
            dist = priors[any_inst][key]['distribution']
            if dist == 'Uniform':
                params[i * n_planet_params + j] = priors[any_inst][key]['min'] + u[i * n_planet_params + j] * (priors[any_inst][key]['max'] - priors[any_inst][key]['min'])
            elif dist == 'Normal':
                params[i * n_planet_params + j] = priors[any_inst][key]['mean'] + u[i * n_planet_params + j] * priors[any_inst][key]['std']
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
    instruments = data.instruments
    
    i_param = i_sep
    for k, param in enumerate(['mu', 'noise', 'GP_sigma', 'GP_length', 'GP_gamma', 'GP_Prot']):
        for j, instrument in enumerate(instruments):
            dist = priors[instrument][param]['distribution']
            # Changing the priors to log space
            if param == 'noise' or param == 'GP_sigma' or param == 'GP_length': 
                priors[instrument][param]['min'], priors[instrument][param]['max'], priors[instrument][param]['mean'], priors[instrument][param]['std'] = np.log(priors[instrument][param]['min']**2), np.log(priors[instrument][param]['max']**2), np.log(priors[instrument][param]['mean']**2), np.log(priors[instrument][param]['std']**2)
            if param == 'GP_Prot': 
                priors[instrument][param]['min'], priors[instrument][param]['max'], priors[instrument][param]['mean'], priors[instrument][param]['std'] = np.log(priors[instrument][param]['min']), np.log(priors[instrument][param]['max']), np.log(priors[instrument][param]['mean']), np.log(priors[instrument][param]['std'])
            
            # Transform the parameters
            if dist == 'Uniform':
                params[i_param] = priors[instrument][param]['min'] + u[i_param] * (priors[instrument][param]['max'] - priors[instrument][param]['min'])
            elif dist == 'Normal':
                params[i_param] = priors[instrument][param]['mean'] + u[i_param] * priors[instrument][param]['std']
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


def dynesty_log_likelihood(p_combined, model, i_shared, data, num_planets, n_planet_params=3, n_gp_params=6):
    """
    Compute the log likelihood for the given parameters.
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

# Homemade George + RadVel model

def juliet_to_george(p):
    return [p[0], np.log(p[1]**2), np.log(p[2]**2), 
                      np.log(p[3]**2), p[4], np.log(p[5])]


def separate_gp_params(comb_params, i_shared, instruments):
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

    def predict(self, t_mod=None):
        if t_mod is None:
            t_mod = self.t
        gp_values = self.gp.predict(self.y, t_mod)
        return gp_values[0]


class Planet_Model:
    def __init__(self, planet_params, num_planets):
        self.planet_params = planet_params
        self.num_planets = num_planets
        
        self.params = radvel.Parameters(num_planets, basis="per tc secosw sesinw k")
        index = 0
        for i in range(1, num_planets + 1):
            self.params['per'+str(i)].value = planet_params[index]
            self.params['tc'+str(i)].value = planet_params[index + 1]
            self.params['secosw'+str(i)].value = 0.0  # Fixed for now (TODO)
            self.params['sesinw'+str(i)].value = 0.0  # Fixed for now (TODO)
            self.params['k'+str(i)].value = planet_params[index + 2]
            index += 3  # Move to the next set of planet parameters
            
        self.rad_model = radvel.RVModel(self.params)

    def predict(self, t_mod):
        return self.rad_model(t_mod)
    
    def update_params(self, p):
        index = 0
        for i in range(1, self.num_planets + 1):
            self.params['per'+str(i)].value = p[index]
            self.params['tc'+str(i)].value = p[index + 1]
            self.params['secosw'+str(i)].value = 0.0
            self.params['sesinw'+str(i)].value = 0.0
            self.params['k'+str(i)].value = p[index + 2]
            index += 3
        
        self.rad_model = radvel.RVModel(self.params)

class Planet_GP_Model:
    def __init__(self, p, t_rv_dict, y_rv_dict, yerr_rv_dict, num_planets=1, n_planet_params=3, n_gp_params=6):
        # Parameter vector
        self.p = p

        # Data dictionaries for time, RV measurements, and their errors
        self.t_rv_dict = t_rv_dict
        self.y_rv_dict = y_rv_dict
        self.yerr_rv_dict = yerr_rv_dict

        # List of instruments
        self.instruments = list(t_rv_dict.keys())

        # Number of planets and parameters
        self.num_planets = num_planets
        self.n_planet_params = n_planet_params
        self.n_gp_params = n_gp_params

        # Separation index between planet and GP parameters
        self.i_sep = n_planet_params * num_planets

        # Isolate the planet parameters
        planet_params = np.array(p[:self.i_sep])
        self.radvel_model = Planet_Model(planet_params, num_planets)

        # Dictionaries to hold GP parameters and models for each instrument
        self.gp_params = {}
        self.gp_models = {}

        # Initialize GP models for each instrument
        for n, instrument in enumerate(self.instruments):
            self.gp_params[instrument] = p[self.i_sep + n * n_gp_params: self.i_sep + (n + 1) * n_gp_params]
            self.gp_models[instrument] = QP_GP_Model(self.gp_params[instrument], t_rv_dict[instrument], self.y_rv_dict[instrument], yerr_rv_dict[instrument])

    def predict(self, t_mod_dict=None, return_components=False):
        # If no t_mod_dict is provided, use the RV times
        if t_mod_dict is None:
            t_mod_dict = self.t_rv_dict

        predictions = {}
        for instrument, t_mod in t_mod_dict.items():
            # Get predictions from GP and planetary models
            gp_values = self.gp_models[instrument].predict(t_mod)
            planet_values = self.radvel_model.predict(t_mod)
            predictions[instrument] = planet_values + gp_values
            
            if return_components:
                predictions[instrument] = {'planet': planet_values, 'GP': gp_values}

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
            planet_predictions = self.radvel_model.predict(self.t_rv_dict[instrument])
            
            # Compute residuals of data minus planetary model
            residuals_planet = self.y_rv_dict[instrument] - planet_predictions
            
            # Compute GP log likelihood for the residuals of the planetary model
            self.gp_models[instrument].gp.compute(self.t_rv_dict[instrument], self.yerr_rv_dict[instrument])
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

def plot_lombscargle_periodograms(t_rv_dict, y_rv_dict, yerr_rv_dict, target_fap=0.01, max_frequency=1.0, combined=False):
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
    plt.show()