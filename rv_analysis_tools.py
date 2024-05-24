import numpy as np
import radvel
import george
from george import kernels
from astropy.table import Table
from datetime import datetime as date

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


### For Goerge + Radvel ###

class DataLoader:
    def __init__(self, data):
        self.data = data
        self.star_info = self.data.get('star', [{}])  # Default to a list with one empty dictionary
        self.instruments = list(self.data['instruments'])
        self.instruments_info = self.data['instruments']
        self.activity_priors = self.data['activity_priors']
        self.time_range = self.data['time']
        self.RV_priors = self.data['RV_priors']
        self.ccf = self.data['CCF']
        self.version = 'DRS-3-0-0'  # HARPS pipeline version
        self.rjd_rjd_off = 2400000.5

        self.t_min = radvel.utils.date2jd(date(*self.time_range['start']))
        self.t_max = radvel.utils.date2jd(date(*self.time_range['end']))

        self.tbl = {}
        self.t_rv, self.y_rv, self.yerr_rv = {}, {}, {}
        self.d2v, self.sd2v, self.Dtemp, self.sDtemp = {}, {}, {}, {}
        self.med_rv_nirps = {}
        self.t_mod = {}

        self._load_data()

    def _load_data(self):
        for instrument in self.instruments:
            star_name = self.star_info.get('name', '')  # Accessing the first element of the star_info list
            ref_star = self.instruments_info[instrument].get('ref_star', '')
            suffix = self.instruments_info[instrument].get('dtemp_suffix', '')
            bin_label = self.instruments_info[instrument].get('bin_label', '')
            pca_label = self.instruments_info[instrument].get('pca_label', '')

            file_path = f'stars/{star_name}/data/lbl{bin_label}_{instrument}_{star_name}_{ref_star}{pca_label}_preprocessed.rdb'
            self.tbl[instrument] = Table.read(file_path, format='rdb')
            self.tbl[instrument]['rjd'] += self.rjd_rjd_off

            # Select desired times
            i_good_times = (self.tbl[instrument]['rjd'] > self.t_min) & (self.tbl[instrument]['rjd'] < self.t_max)
            self.tbl[instrument] = self.tbl[instrument][i_good_times]

            # RV data    
            self.t_rv[instrument], self.y_rv[instrument], self.yerr_rv[instrument] = self.tbl[instrument]['rjd'], self.tbl[instrument]['vrad'], self.tbl[instrument]['svrad']

            # Stellar activity indicators
            self.d2v[instrument], self.sd2v[instrument] = self.tbl[instrument]['d2v'] / np.max(self.tbl[instrument]['d2v']), np.abs(self.tbl[instrument]['sd2v'] / np.max(self.tbl[instrument]['d2v']))
            self.d2v[instrument] -= np.median(self.d2v[instrument])
            self.Dtemp[instrument], self.sDtemp[instrument] = self.tbl[instrument]['DTEMP' + suffix], self.tbl[instrument]['sDTEMP' + suffix]
            self.Dtemp[instrument] -= np.median(self.Dtemp[instrument])

            # Median of the RVs
            self.med_rv_nirps[instrument] = np.rint(np.median(self.tbl[instrument]['vrad'].data))
            self.t_mod[instrument] = np.linspace(np.min(self.t_rv[instrument]), np.max(self.t_rv[instrument]), 1000)



def gaussian_logp(x: float, mu: float, sigma: float) -> float:
    # Copied from radvel for convenience
    return -0.5 * ((x - mu) / sigma) ** 2 - 0.5 * np.log((sigma**2) * 2.0 * np.pi)


def uniform_logp(x: float, minval: float, maxval: float) -> float:
    # Copied from radvel for convenience
    if x <= minval or x >= maxval:
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
    else:
        raise ValueError(f'Distribution not recognized for mu')
    
    # Log White noise: Uniform
    if priors['noise']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[1], np.log(priors['noise']['min']**2), np.log(priors['noise']['max']**2))
    elif priors['noise']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[1], np.log(priors['noise']['min']**2), np.log(priors['noise']['max']**2))
    elif priors['noise']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[1], np.log(priors['noise']['mean']**2), np.log(priors['noise']['std']**2))
    else:
        raise ValueError('Distribution not recognized for white noise')
    
    # Log Variance: Uniform
    if priors['GP_sigma']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[2], np.log(priors['GP_sigma']['min']**2), np.log(priors['GP_sigma']['max']**2))
    elif priors['GP_sigma']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[2], np.log(priors['GP_sigma']['min']**2), np.log(priors['GP_sigma']['max']**2))
    elif priors['GP_sigma']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[2], np.log(priors['GP_sigma']['mean']**2), np.log(priors['GP_sigma']['std']**2))
    else: 
        raise ValueError('Distribution not recognized for amplitude')
    
    # Log metric (lambda**2): Uniform
    if priors['GP_length']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[3], np.log(priors['GP_length']['min']**2), np.log(priors['GP_length']['max']**2))
    elif priors['GP_length']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[3], np.log(priors['GP_length']['min']**2), np.log(priors['GP_length']['max']**2))
    elif priors['GP_length']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[3], np.log(priors['GP_length']['mean']**2), np.log(priors['GP_length']['std']**2))
    else:
        raise ValueError('Distribution not recognized for length scale')
    
    # Gamma: Jeffreys prior
    if priors['GP_gamma']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[4], priors['GP_gamma']['min'], priors['GP_gamma']['max'])
    elif priors['GP_gamma']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[4], priors['GP_gamma']['min'], priors['GP_gamma']['max'])
    elif priors['GP_gamma']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[4], priors['GP_gamma']['mean'], priors['GP_gamma']['std'])
    else:
        raise ValueError('Distribution not recognized for gamma')
    
    # Log Period: Uniform
    if priors['GP_Prot']['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[5], np.log(priors['GP_Prot']['min']), np.log(priors['GP_Prot']['max']))
    elif priors['GP_Prot']['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[5], np.log(priors['GP_Prot']['min']), np.log(priors['GP_Prot']['max']))
    elif priors['GP_Prot']['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[5], np.log(priors['GP_Prot']['mean']), np.log(priors['GP_Prot']['std']))
    else:
        raise ValueError('Distribution not recognized for log period')
    
    return log_prob

def planet_log_prior(p: np.ndarray, priors, idx = 1) -> float:
    log_prob = 0.0
    
    # Period of planet
    if priors['per'+str(idx)]['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[0], priors['per'+str(idx)]['min'], priors['per'+str(idx)]['max'])
    elif priors['per'+str(idx)]['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[0], priors['per'+str(idx)]['min'], priors['per'+str(idx)]['max'])
    elif priors['per'+str(idx)]['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[0], priors['per'+str(idx)]['mean'], priors['per'+str(idx)]['std'])
    else:
        raise ValueError(f'Distribution not recognized for per'+str(idx))

    # Time of conjunction
    if priors['tc'+str(idx)]['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[1], priors['tc'+str(idx)]['min'], priors['tc'+str(idx)]['max'])
    elif priors['tc'+str(idx)]['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[1], priors['tc'+str(idx)]['min'], priors['tc'+str(idx)]['max'])
    elif priors['tc'+str(idx)]['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[1], priors['tc'+str(idx)]['mean'], priors['tc'+str(idx)]['std'])
    else:
        raise ValueError(f'Distribution not recognized for tc'+str(idx))
   
    
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
    if priors['k'+str(idx)]['distribution'] == 'Uniform':
        log_prob += uniform_logp(p[2], priors['k'+str(idx)]['min'], priors['k'+str(idx)]['max'])
    elif priors['k'+str(idx)]['distribution'] == 'loguniform':
        log_prob += jeffreys_logp(p[2], priors['k'+str(idx)]['min'], priors['k'+str(idx)]['max'])
    elif priors['k'+str(idx)]['distribution'] == 'Normal':
        log_prob += gaussian_logp(p[2], priors['k'+str(idx)]['mean'], priors['k'+str(idx)]['std'])
    elif priors['k'+str(idx)]['distribution'] == 'fixed':
        log_prob += 0.0
    else:
        raise ValueError(f'Distribution not recognized for k'+str(idx))
   
    
    
    return log_prob


def create_combined_params(params_dict, i_shared, param_names):
    comb_params = []
    comb_params_labels = []

    instruments = list(params_dict.keys())

    for i in range(len(param_names)):
        if i in i_shared:
            # Shared parameters, use the first instrument's parameters
            comb_params.append(params_dict[instruments[0]][i])
            comb_params_labels.append(param_names[i])
        else:
            # Separate parameters for each instrument
            for instrument in instruments:
                comb_params.append(params_dict[instrument][i])
                comb_params_labels.append(f"{param_names[i]}_{instrument}")
            
    return comb_params, comb_params_labels


def separate_params(comb_params, i_shared, instruments):
    params_dict = {instrument: [] for instrument in instruments}
    
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
    
    return params_dict


def interleave(main_list, insert_list):
    result = [elem for pair in zip(main_list, insert_list) for elem in pair]
    # Ensure last element from insert_list is added only once
    if len(main_list) == len(insert_list):
        return result
    else:
        # If main_list is longer, add the remaining elements
        result.extend(main_list[len(insert_list):])
        # If insert_list is longer, add the remaining elements
        result.extend(insert_list[len(main_list):])
    return result

def interleave_params(param_dict):
    # Extract the instrument names and number of parameters
    instruments = list(param_dict.keys())
    num_params = len(param_dict[instruments[0]])

    # Initialize an empty list to hold the interleaved parameters
    interleaved_params = []

    # Loop through each parameter index
    for param_idx in range(num_params):
        # Loop through each instrument
        for instrument in instruments:
            # Append the parameter to the interleaved list
            interleaved_params.append(param_dict[instrument][param_idx])

    return interleaved_params


# Homemade George + RadVel model

def juliet_to_george(p):
    return [p[0], np.log(p[1]**2), np.log(p[2]**2), 
                      np.log(p[3]**2), p[4], np.log(p[5])]

class QP_GP_Model:
    def __init__(self, gp_params, t_rv, y_rv, yerr_rv):
        self.gp_params = gp_params
        self.t_rv = t_rv
        self.y_rv = y_rv
        self.yerr_rv = yerr_rv
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
        self.gp.compute(t_rv, yerr=yerr_rv+self.jitter)

    def predict(self, t_mod=None):
        if t_mod is None:
            t_mod = self.t_rv
        gp_values = self.gp.predict(self.y_rv, t_mod)
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

class Planet_GP_Model:
    def __init__(self, p, t_rv_dict, y_rv_dict, yerr_rv_dict, num_planets=1):
        self.p = p
        self.t_rv_dict = t_rv_dict
        self.y_rv_dict = y_rv_dict
        self.yerr_rv_dict = yerr_rv_dict
        self.num_planets = num_planets

        planet_params = np.array(p[:3*num_planets])
        self.radvel_model = Planet_Model(planet_params, num_planets)
        
        # Create dictionaries to hold GP models for each instrument
        self.gp_models = {}
        
        # Separate and interleave GP parameters
        gp_params = np.array(p[3*num_planets:])
        num_gp_params_per_instrument = len(gp_params) // len(t_rv_dict)

        # Interleave GP parameters for each instrument
        interleaved_gp_params = {instrument: [] for instrument in t_rv_dict.keys()}
        for i in range(num_gp_params_per_instrument):
            for j, instrument in enumerate(t_rv_dict.keys()):
                interleaved_gp_params[instrument].append(gp_params[i * len(t_rv_dict) + j])

        # Initialize GP models
        for instrument, params in interleaved_gp_params.items():
            params = np.array(params)
            self.gp_models[instrument] = QP_GP_Model(params, t_rv_dict[instrument], y_rv_dict[instrument], yerr_rv_dict[instrument])

    def predict(self, t_mod_dict=None, return_components=False):
        if t_mod_dict is None:
            t_mod_dict = self.t_rv_dict

        predictions = {}
        for instrument, t_mod in t_mod_dict.items():
            gp_values = self.gp_models[instrument].predict(t_mod)
            planet_values = self.radvel_model.predict(t_mod)
            predictions[instrument] = planet_values + gp_values
            
            if return_components:
                predictions[instrument] = {'planet': planet_values, 'GP': gp_values}

        return predictions

    def log_likelihood(self):
        predictions = self.predict()
        
        ll_total = 0
        for instrument, pred in predictions.items():
            try:
                residuals = self.y_rv_dict[instrument] - pred
                individual_ll = -0.5 * (np.sum((residuals / self.yerr_rv_dict[instrument])**2 + np.log(2 * np.pi * self.yerr_rv_dict[instrument]**2)))
                ll_total += individual_ll
            except np.linalg.LinAlgError:
                return -np.inf  # Return a very bad likelihood if Cholesky decomposition fails
        
        return ll_total

    def update_params(self, p):
        self.p = p
        planet_params = np.array(p[:3*self.num_planets])
        self.radvel_model.update_params(planet_params)  # Update the planet parameters
        
        # Separate GP parameters
        gp_params = p[3*self.num_planets:]
        
        # Number of GP parameters per instrument
        num_gp_params_per_instrument = len(gp_params) // len(self.t_rv_dict)
        
        # Interleave GP parameters for each instrument
        interleaved_gp_params = {instrument: [] for instrument in self.t_rv_dict.keys()}
        for i in range(num_gp_params_per_instrument):
            for j, instrument in enumerate(self.t_rv_dict.keys()):
                interleaved_gp_params[instrument].append(gp_params[i * len(self.t_rv_dict) + j])
        
        # Convert lists to numpy arrays and update GP models
        for instrument, params in interleaved_gp_params.items():
            params = np.array(params)
            #print(f"{instrument} GP parameters: {params}")
            self.gp_models[instrument].gp.set_parameter_vector(params)
    
