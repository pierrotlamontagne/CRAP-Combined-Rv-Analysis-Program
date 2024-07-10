import os
import time
import logging
from pathlib import Path
import copy
import corner
import emcee
import matplotlib.pyplot as plt
import numpy as np
from george.gp import LinAlgError
import yaml
from matplotlib import cm
import rv_analysis_tools as rv
import importlib
from astropy.timeseries import LombScargle
import dynesty
import radvel
import pickle
from itertools import product

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

dir_template = "{parent_dir}/{star}/{crap_dir}/{model_to_run}/{shared_param}/{nplanet}/{fit_ecc}/{sampler}/"

# Parent directory
parent_dir = ['CRAPresults']

# Stars to run CRAP on 
stars = ['PROXIMA']

# CRAPanalysis directory
crap_dir = ["CRAPanalysis"]

# Which model to run (one per python file)
model_to_run = ['keplerian+GP']

# List of shared parameters to run
#shared_params_list = ["share_params_3", "share_params_4","share_params_5", 
                 #"share_params_3,4", "share_params_3,5","share_params_4,5",  "share_params_3,4,5"]
shared_params_list = ['share_params_3,4,5']

# List of sampler
sampler_list = ['emcee']

# List of numbers of planets to test
nplanets_list = ['1_planet']

# Fit eccentricity or not
fit_ecc_list = ['no_ecc']

# Create combinations of all the run parameters
combinations = product(parent_dir, stars, crap_dir, model_to_run, shared_params_list, nplanets_list, fit_ecc_list, sampler_list)

# How many runs to skip
skip = 0

# Counts the runs
counter = 0

for combo in combinations:
    if counter < skip:
        counter += 1
        continue
    
    # Create the directory path
    working_path = dir_template.format(
        parent_dir=combo[0],
        star=combo[1], 
        crap_dir=combo[2],
        model_to_run=combo[3],
        shared_param=combo[4],
        nplanet=combo[5],
        fit_ecc=combo[6],
        sampler=combo[7]
    )
    counter += 1

    # Create the directory
    os.makedirs(working_path, exist_ok=True)
    
    star = combo[1]
    
    # Log running information
    logger.info(f'Starting run for {star} with parameters:')
    logger.info(f'Model: {combo[3]} | Shared params: {combo[4]} | Number of planets: {combo[5]} | Fit eccentricity: {combo[6]} | Sampler: {combo[7]}')

    # Timing the loading of input files and data
    start_time = time.time()
    logger.info('Loading input file and data...')
    
    with open(f'CRAPresults/{star}/input.yaml', 'r') as file:
        yaml_file = yaml.safe_load(file)
        
    # Change shared params
    i_shared = rv.transform_shared_params(combo[4])
        
    data = rv.DataLoader(yaml_file)
    gp_labels = rv.generate_param_names(['mu', 'log_wn', 'log_amp', 'log_lambda', 'gamma', 'log_Prot'],
                                    i_shared, data.instruments)
    
    elapsed_time = time.time() - start_time
    logger.info(f'Data loaded in {elapsed_time:.2f} seconds.')
    
    # Change number of planets
    data.nplanets = rv.transform_nplanets(combo[5])
    
    # Change fit_ecc
    fit_ecc = rv.transform_fit_ecc(combo[6])
    if fit_ecc: 
        data.n_planet_params = 5
    else: 
        data.n_planet_params = 3
    
    # Change sampler
    data.sampler = combo[7]
    
    # Always run the RV analysis
    data.run_RV = True # TODO: Can change to only recreate the graphs
    
    # Define number of GP params
    n_gp_params = 6
    
    # Matplotlib preferences
    plt.style.use('ggplot')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'legend.fontsize': 'small',
        'axes.labelsize': 'medium',
        'axes.titlesize':'large',
        'xtick.labelsize':'small',
        'ytick.labelsize':'small',
        'figure.autolayout': True,
    })

    colors = plt.cm.jet(np.linspace(1, 0, len(data.instruments)))
    
    # Plot the periodogram of the RV data
    rv.plot_lombscargle_periodograms(data.t_rv, data.y_rv, data.yerr_rv, combined=True, file_path=working_path+'RV_periodogram.png')
    
    if data.use_indicator:
        act_post_samples_walkers = np.load(f'CRAPresults/{star}/CRAPanalysis/activity/act_post_samples.npy')
        act_post_samples = act_post_samples_walkers.reshape(-1, act_post_samples_walkers.shape[2])

        # For first guess
        act_med_params = np.median(act_post_samples, axis=0)
        separated_med_params, separated_med_params_dict = rv.separate_gp_params(act_med_params, i_shared, data.instruments)
        act_samples_dict = rv.separate_gp_params_samples(act_post_samples_walkers, i_shared, data.instruments)  # Contains the samples for the parameters of each instrument
        
    logger.info(f'Creating initial guess for the RV parameters for {data.nplanets} planets and activity...')
    
    # Initialize lists to hold planet parameters and labels
    p0 = []
    planet_labels = []
    priors = {}  # Dictionary to store the priors
    prior_params_names = ['mu', 'noise', 'GP_sigma', 'GP_length', 'GP_gamma', 'GP_Prot']

    # Loop over the number of planets to construct the parameter and label lists
    for i in range(1, data.nplanets + 1):
        if data.n_planet_params == 3: 
            p0.extend([
                data.RV_priors[f'per{i}']['guess'],
                data.RV_priors[f'tc{i}']['guess'],
                data.RV_priors[f'k{i}']['guess']
            ])
            planet_labels.extend([f'per{i}', f'tc{i}', f'k{i}'])
        else: 
            p0.extend([
                data.RV_priors[f'per{i}']['guess'],
                data.RV_priors[f'tc{i}']['guess'],
                data.RV_priors[f'e{i}']['guess'],
                data.RV_priors[f'w{i}']['guess'],
                data.RV_priors[f'k{i}']['guess']
            ])
            planet_labels.extend([f'per{i}', f'tc{i}', f'e{i}', f'w{i}', f'k{i}'])

    # Loop over the instruments to construct the GP parameter list
    for inst in data.instruments: 
        priors[inst] = copy.deepcopy(data.RV_priors)
        
    for instrument in data.instruments: 
        if data.use_indicator:
            # Create the initial vector from the activity indicator
            p0.extend(
                [np.mean(data.y_rv[instrument]), np.log(0.1**2), np.log(np.std(data.y_rv[instrument])**2),
                separated_med_params_dict[instrument][3], separated_med_params_dict[instrument][4], separated_med_params_dict[instrument][5]])
        
            # Add priors for GP parameters
            for idx, param_name in enumerate(prior_params_names):
                if param_name == 'GP_length':
                    priors[instrument][param_name] = {
                        'guess': float(np.median(np.sqrt(np.exp(act_samples_dict[instrument][idx])))),
                        'distribution': 'TruncatedNormal',
                        'mean': float(np.median(np.sqrt(np.exp(act_samples_dict[instrument][idx])))),
                        'std': float(np.std(np.sqrt(np.exp(act_samples_dict[instrument][idx])))),
                        'min': data.RV_priors['GP_length']['min'],
                        'max': data.RV_priors['GP_length']['max']
                    }
                elif param_name == 'GP_gamma':
                    priors[instrument][param_name] = {
                        'guess': float(np.median(act_samples_dict[instrument][idx])),
                        'distribution': 'TruncatedNormal',
                        'mean': float(np.median(act_samples_dict[instrument][idx])),
                        'std': float(np.std(act_samples_dict[instrument][idx])), 
                        'min': data.RV_priors['GP_gamma']['min'], 
                        'max': data.RV_priors['GP_gamma']['max']
                    }
                elif param_name == 'GP_Prot':
                    priors[instrument][param_name] = {
                        'guess': float(np.median(np.exp(act_samples_dict[instrument][idx]))),
                        'distribution': 'TruncatedNormal',
                        'mean': float(np.median(np.exp(act_samples_dict[instrument][idx]))),
                        'std': float(np.std(np.exp(act_samples_dict[instrument][idx]))), 
                        'min': data.RV_priors['GP_Prot']['min'],
                        'max': data.RV_priors['GP_Prot']['max']
                    }
                else: 
                    continue

        else:
            p0.extend(rv.juliet_to_george([
                np.mean(data.y_rv[instrument]), 0.1, np.std(data.y_rv[instrument]),
                data.RV_priors['GP_length']['guess'], data.RV_priors['GP_gamma']['guess'], data.RV_priors['GP_Prot']['guess']
            ]))
            priors[instrument] = copy.deepcopy(data.RV_priors)

    # Create initial parameter vector: 
    p0_vect = rv.params_vector(p0, data.instruments, i_shared,
                            num_planets=data.nplanets, n_planet_params=data.n_planet_params)
    comb_p0 = p0_vect.combine()
    param_labels = np.concatenate([planet_labels, gp_labels])

    # Save the priors to a file
    with open(working_path + 'RV_priors.yaml', 'w') as file:
        yaml.dump(priors, file)
        
    logger.info(f'Creating {data.nplanets} planet + GP model...')
    model = rv.Planet_GP_Model(p0_vect, data,
                            num_planets=data.nplanets,
                            n_planet_params=data.n_planet_params)
    logger.info('Model created successfully.')
        
    if data.sampler == "emcee":
        # MCMC fit of the GP hyperparameters 
        nwalkers, ndim = 3*len(comb_p0), len(comb_p0)
        num_warmup = 50 * ndim
        num_post_samples = 500 * ndim

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            rv.emcee_log_post,
            args=(model, data, priors, i_shared, data.nplanets, data.n_planet_params),
            moves=[emcee.moves.DEMove()]
        )
        
        # Change the dimension of p0 with the number of walkers
        comb_p0_emcee = np.array([comb_p0 + np.random.randn(ndim) * 1e-4 for i in range(nwalkers)])

        if data.run_RV:
            logger.info('Running MCMC for the combined model...')
            start_time = time.time()
            sampler.run_mcmc(comb_p0_emcee, num_post_samples + num_warmup, progress=True)  # Run the MCMC
            
            # Save the samples
            post_samples_walkers = sampler.get_chain(discard=num_warmup)
            post_samples = post_samples_walkers.reshape(-1, post_samples_walkers.shape[2])  # Remove the walkers dimension
            np.save(working_path + 'post_samples_walkers.npy', post_samples_walkers)
            np.save(working_path + 'post_samples.npy', post_samples)  
            
            # Save the log probabilities
            log_prob_samples = sampler.get_log_prob(discard=num_warmup)
            np.save(working_path + 'log_prob_samples.npy', log_prob_samples) 
            
            # Compute the BIC value
            N_data_points = np.sum([len(data.t_rv[instrument]) for instrument in data.instruments])
            bic = rv.bic_calculator(np.max(log_prob_samples), ndim, N_data_points)
            np.savetxt(working_path + f'BIC.txt',
                       np.array([bic]), fmt='%.6f')
            elapsed_time = time.time() - start_time
            logger.info(f'MCMC completed in {elapsed_time:.2f} seconds.')
            logger.info(f'BIC: {bic}')
            
        else: 
            logger.info('Loading previous MCMC fit for the combined model...')
            
    if data.sampler == 'dynesty': 
        # Set up the Nested Sampler
        ndim = len(comb_p0)

        # Define parameters in relation to ndim
        n_live_points = 10 * ndim  # Number of live points, typically 10 * ndim
        dlogz = 0.1  # Stopping criteria for the evidence
        maxiter = 5000 * ndim  # Maximum number of iterations
        num_warmup = 100 * ndim


        if data.run_RV: 
            sampler = dynesty.NestedSampler(rv.dynesty_log_likelihood, rv.dynesty_prior_transform, ndim, 
                                            nlive=n_live_points, 
                                            logl_args=(model, i_shared, data, data.nplanets, data.n_planet_params), 
                                            ptform_args=(priors, model, i_shared, data, data.nplanets, data.n_planet_params))

            # Run the Nested Sampler
            sampler.run_nested(dlogz=dlogz, maxiter=maxiter)
            results = sampler.results
            # Save the results to a .pkl file
            with open(working_path + 'dynesty_results.pkl', 'wb') as f:
                pickle.dump(results, f)
                
            elapsed_time = time.time() - start_time
            logger.info(f'Nested sampling completed in {elapsed_time:.2f} seconds.')
        else: 
            logger.info('Loading previous Nested Sampling fit for the planet + GP model...')
            
            with open(working_path + 'dynesty_results.pkl', 'rb') as f:
                results = pickle.load(f)
            
        all_log_prob_samples = results.logl
        
        # Smooth the log-probability curve using a moving average
        window_size = 50
        smoothed_log_prob = np.convolve(all_log_prob_samples, np.ones(window_size)/window_size, mode='valid')

        # Reverse the smoothed log-probability and calculate the derivative
        smoothed_log_prob_reversed = smoothed_log_prob[::-1]
        derivative_log_prob_reversed = np.abs(np.diff(smoothed_log_prob_reversed))

        # Identify the point where the derivative falls below a threshold and stays there
        threshold = 0.03
        try:
            burn_in_reversed = next((i for i, v in enumerate(derivative_log_prob_reversed) if v > threshold), 0)
        except StopIteration:
            burn_in_reversed = 0
        
        burn_in = len(smoothed_log_prob_reversed) - burn_in_reversed - 1

        logger.info(f"Determined burn-in period: {burn_in}")

        # Gather the results: samples and log_prob of all samples
        samples = np.copy(results.samples[burn_in:])  # Array of shape (nsamples, ndim)
        np.save(working_path + 'post_samples.npy', samples)
        
        log_prob = np.copy(results.logl[burn_in:])  # Log-likelihood values for each sample
        np.save(working_path + 'log_prob_samples.npy', log_prob)


        # Extract the evidence (logZ) and its uncertainty (logZerr)
        log_evidence = results.logz[-1]
        log_evidence_err = results.logzerr[-1]
        evidence_filename = working_path + f'log_evidence.txt'
        np.savetxt(evidence_filename, np.array([[log_evidence, log_evidence_err]]), header="log_evidence log_evidence_err", fmt='%.6f')
        
            
    post_samples = np.load(working_path + 'post_samples.npy')
    if data.sampler == 'emcee': 
        post_samples_walkers = np.load(working_path + 'post_samples_walkers.npy')
    log_prob_samples = np.load(working_path + 'log_prob_samples.npy')
    
    if data.sampler == 'emcee':  
        logger.info('Plotting traceplot...')
        start_time = time.time()

        fig, axes = plt.subplots(ndim, figsize=(10, ndim*2), sharex=True)

        for i in range(ndim):
            ax = axes[i]
            ax.plot(post_samples_walkers[:, :, i], "k", alpha=0.3)
            ax.set_xlim(0, len(post_samples_walkers))
            ax.yaxis.set_label_coords(-0.1, 0.5)
            ax.set_ylabel(param_labels[i])
        axes[-1].set_xlabel("step number")
        plt.suptitle('Joint fit of NIRPS and HARPS planet + GP')
        plt.savefig(working_path + f'traceplot.png')

        elapsed_time = time.time() - start_time
        logger.info(f'Traceplot plotted in {elapsed_time:.2f} seconds.')

    elif data.sampler == 'dynesty': 
        logger.info('Plotting traceplot...')
        start_time = time.time()
        
        # Number of parameters
        ndim = len(comb_p0)

        # Create a figure for the trace plot
        fig, axes = plt.subplots(ndim + 1, 1, figsize=(10, 2 * (ndim + 1)), sharex=True)

        # Plot each parameter
        for i in range(ndim):
            axes[i].plot(post_samples[:, i], color='blue', alpha=0.5)
            axes[i].set_ylabel(param_labels[i])
            axes[i].grid(True)

        # Plot the log probability
        axes[ndim].plot(log_prob_samples, color='red', alpha=0.5)
        axes[ndim].set_ylabel('log_prob')
        axes[ndim].set_xlabel('Iteration')
        axes[ndim].grid(True)

        # Add a title to the figure
        fig.suptitle('Trace Plot', fontsize=16)

        # Adjust layout to avoid overlap
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        plt.savefig(working_path + f'traceplot.png')
        
        elapsed_time = time.time() - start_time
        logger.info(f'Traceplot plotted in {elapsed_time:.2f} seconds.')
        
    else: 
        logger.error('The selected sampler is not valid.')
        
    logger.info('Plotting corner plot...')
    start_time = time.time()

    # Choose which parameters to show in the cornerplot
    corner_samples = post_samples

    # Make a corner plot
    fig = corner.corner(
        corner_samples,
        labels=param_labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_kwargs={"fontsize": 12},
        color="blue",  # Set a color scheme (optional)
        hist_kwargs={"linewidth": 1.5, "alpha": 0.7},  # Adjust histogram appearance
        fill_contours=True,  # Show contours for data density
        fill_kw={"cmap": "Blues"},  # Set colormap for contours (optional)
        smooth=True,  # Improve smoothness of contours
        fontsize=10,  # Set font size for labels and tick marks
        truth_color='orange',
    )

    plt.tight_layout()
    plt.savefig(working_path + f'cornerplot.png')

    elapsed_time = time.time() - start_time
    logger.info(f'Corner plot created in {elapsed_time:.2f} seconds.')
    
    # Get the best parameters
    med_params = np.median(post_samples.flatten().reshape(-1, ndim), axis=0)
    max_params = rv.get_max_likelihood_params(post_samples.flatten().reshape(-1, ndim), log_prob_samples.flatten())

    # Get the uncertainty
    errors = np.percentile(post_samples.flatten().reshape(-1, ndim), [16, 84], axis=0)

    logger.info('Median likelihood parameters:')
    logger.info('--------------------------')
    # Show the median likelihood parameters
    with open(working_path + 'median_likelihood_params.txt', 'w') as text_file:
        for i, label in enumerate(param_labels):
            logger.info(f'{label}: {med_params[i]:.3f} - {errors[0][i]:.3f} + {errors[1][i]:.3f}')
            text_file.write(f'{label}: {med_params[i]:.3f} - {errors[0][i]:.3f} + {errors[1][i]:.3f}\n')
        
    logger.info('Maximum likelihood parameters:')
    logger.info('--------------------------')
    # Show the maximum likelihood parameters
    with open(working_path + 'maximum_likelihood_params.txt', 'w') as text_file:
        for i, label in enumerate(param_labels):
            logger.info(f'{label}: {max_params[i]:.3f} - {errors[0][i]:.3f} + {errors[1][i]:.3f}')
            text_file.write(f'{label}: {max_params[i]:.3f} - {errors[0][i]:.3f} + {errors[1][i]:.3f}\n')

    best_params = np.copy(max_params)
    
    # Separate the parameters for each instrument
    best_gp_params, best_gp_params_dict = rv.separate_gp_params(med_params[data.n_planet_params * data.nplanets:], i_shared, data.instruments)
    separated_best_params = np.concatenate([best_params[:data.nplanets * data.n_planet_params], best_gp_params])
    model.update_params(separated_best_params)

    # Fetch the noise term of each instrument
    noise_terms = {instrument: np.sqrt(data.yerr_rv[instrument]**2 + np.exp(best_gp_params_dict[instrument][1])) for instrument in data.instruments}
    
    logger.info('Plotting the best-fit model and residuals...')
    start_time = time.time()

    # Plot the planets + GP model and the samples
    fig, axes = plt.subplots(5, 1, figsize=(20, 18), sharex=False)
    rjd_off = 245000

    # Dictionaries to store the model predictions
    planet_mod_pred_on_data = {}
    planet_mod_pred_on_mod_times = {}
    gp_mod_pred_on_data = {}
    gp_mod_pred_on_mod_times = {}

    # RV offset for each instrument
    rv_offset = {}

    # Planet residuals
    planet_res = {}

    for instrument in data.instruments: 
        # Evaluate planets
        planet_mod_pred_on_data[instrument] = model.predict(data.y_rv, data.t_rv, return_components=True)[instrument]['planet']
        planet_mod_pred_on_mod_times[instrument] = model.predict(data.y_rv, data.t_mod, return_components=True)[instrument]['planet']
        
        # Calculate planet residuals to evaluate the GP
        planet_res[instrument] = data.y_rv[instrument] - planet_mod_pred_on_data[instrument]

    for idx, instrument in enumerate(data.instruments):
        
        # Offset
        rv_offset[instrument] = best_params[data.n_planet_params * data.nplanets + idx]
        
        # Evaluate GP
        gp_mod_pred_on_data[instrument] = model.predict(planet_res, data.t_rv, return_components=True)[instrument]['GP']
        gp_mod_pred_on_mod_times[instrument] = model.predict(planet_res, data.t_mod, return_components=True)[instrument]['GP']
        
        # Plot data with error bars
        axes[0].errorbar(data.t_rv[instrument] - rjd_off, data.y_rv[instrument], yerr=noise_terms[instrument], fmt="o", ms=5, label=f'{instrument}', color=colors[idx])
        axes[1].errorbar(data.t_rv[instrument] - rjd_off, data.y_rv[instrument] - planet_mod_pred_on_data[instrument], yerr=noise_terms[instrument], fmt="o", ms=5, label=f'{instrument}', color=colors[idx])
        axes[2].errorbar(data.t_rv[instrument] - rjd_off, data.y_rv[instrument] - gp_mod_pred_on_data[instrument], yerr=noise_terms[instrument], fmt="o", ms=5, label=f'{instrument}', color=colors[idx])
        
        # Plot best likelihood
        axes[2].plot(data.t_mod[instrument] - rjd_off, planet_mod_pred_on_mod_times[instrument], color=colors[idx], linewidth=1.5, alpha=0.5)  # Keplerian
        axes[1].plot(data.t_mod[instrument] - rjd_off, gp_mod_pred_on_mod_times[instrument], color=colors[idx], linestyle='--', linewidth=1.5, alpha=0.5)  # GP
        axes[0].plot(data.t_mod[instrument] - rjd_off, planet_mod_pred_on_mod_times[instrument] + gp_mod_pred_on_mod_times[instrument], color=colors[idx], linewidth=1.5, alpha=0.5)  # Full model
        # Plot residuals
        axes[3].errorbar(data.t_rv[instrument] - rjd_off, data.y_rv[instrument] - planet_mod_pred_on_data[instrument] - gp_mod_pred_on_data[instrument], yerr=noise_terms[instrument], fmt='o', color=colors[idx], label=f'{instrument} residuals')
        
        # Plot the periodogram of the residuals
        ls = LombScargle(data.t_rv[instrument] - rjd_off, data.y_rv[instrument] - planet_mod_pred_on_data[instrument] - gp_mod_pred_on_data[instrument], dy=noise_terms[instrument])
        target_fap = 0.01
        fap = ls.false_alarm_level(target_fap)
        freq, power = ls.autopower(maximum_frequency=1.0)
        period = 1 / freq

        ls_window = LombScargle(data.t_rv[instrument] - rjd_off, np.ones_like(data.y_rv[instrument]), dy=noise_terms[instrument], fit_mean=False, center_data=False)
        power_window = ls_window.power(freq)

        axes[4].plot(period, power, color=colors[idx], label=f'{instrument} Periodogram')
        axes[4].plot(period, power_window, color=colors[idx], linestyle='--', label=f'{instrument} Window Function')

    axes[4].axhline(fap, linestyle="--", color="black", label=f"{target_fap * 100}% FA level", alpha=0.5)

    axes[0].set_title('Full model')
    axes[1].set_title('GP model (data - keplerian)')
    axes[2].set_title('Keplerian model (data - GP model)')
    axes[3].set_title('Residuals')
    axes[4].set_title('Periodograms of the residuals')
    axes[3].set_xlabel('Time [rjd]')
    axes[0].set_ylabel('RV [m/s]')
    axes[1].set_ylabel('RV [m/s]')
    axes[2].set_ylabel('RV [m/s]')
    axes[3].set_ylabel('RV [m/s]')
    axes[0].legend(title='Instruments')
    axes[1].legend(title='Instruments')
    axes[2].legend(title='Instruments')
    axes[3].legend(title='Instruments')
    axes[4].set_xlabel('Period [days]')
    axes[4].set_ylabel('Power')
    axes[4].legend()
    axes[4].set_xscale('log')
    axes[4].set_xlim(0, 200)
    plt.suptitle(f'{data.nplanets} planets + GP model', fontsize=20)
    plt.savefig(working_path + f'best_fit.png')
    plt.show()

    elapsed_time = time.time() - start_time
    logger.info(f'Best-fit model and residuals plotted in {elapsed_time:.2f} seconds.')
    
    # Calculate the RMS of the residuals
    rms = np.sqrt(np.mean([np.sum((np.abs(data.y_rv[instrument] - planet_mod_pred_on_data[instrument] - gp_mod_pred_on_data[instrument]))**2) for instrument in data.instruments]))
    np.savetxt(working_path + 'rms.txt', np.array([rms]), fmt='%.6f')
    logger.info(f'RMS of the residuals: {rms:.3f} m/s')

    # Calculate the median residual value
    median_residual = np.median([np.median(np.abs(data.y_rv[instrument] - planet_mod_pred_on_data[instrument] - gp_mod_pred_on_data[instrument])) for instrument in data.instruments])
    np.savetxt(working_path + 'median_residual.txt', np.array([median_residual]), fmt='%.6f')
    logger.info(f'Median residual value: {median_residual:.3f} m/s')
    
    logger.info('Plotting phase-folded RV curves of our planets...')
    start_time = time.time()

    def foldAt(time, period, t0):
        return ((time - t0 + 0.5 * period) % period) / period

    def bin_data(phases, RVs, RV_errs, num_bins=20):
        bins = np.linspace(0, 1, num_bins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        binned_RVs = np.zeros(num_bins)
        binned_errs = np.zeros(num_bins)
        
        for i in range(num_bins):
            in_bin = (phases >= bins[i]) & (phases < bins[i+1])
            if np.any(in_bin):
                weights = 1 / RV_errs[in_bin]**2
                binned_RVs[i] = np.sum(RVs[in_bin] * weights) / np.sum(weights)
                binned_errs[i] = np.sqrt(1 / np.sum(weights))
            else:
                binned_RVs[i] = np.nan
                binned_errs[i] = np.nan
        
        return bin_centers, binned_RVs, binned_errs

    def phase_folded_plot(data_times, model_times, RVs, RV_errs, keplerian_model_RVs, period, t0, instrument_names):
        
        # Plotting the data
        plt.figure(figsize=(10, 6))
        
        for i, instrument in enumerate(instrument_names):
            # Fold the data and model times to get the phases
            phases_data = foldAt(data_times[instrument], period, t0)
            phases_model = foldAt(model_times[instrument], period, t0)

            # Sort the phases for plotting
            sort_indices_data = np.argsort(phases_data)
            sort_indices_model = np.argsort(phases_model)

            phases_data_sorted = phases_data[sort_indices_data]
            RV_sorted = RVs[instrument][sort_indices_data]
            
            phases_model_sorted = phases_model[sort_indices_model]
            keplerian_model_RVs_sorted = keplerian_model_RVs[instrument][sort_indices_model]

            # Plot the observed data
            plt.errorbar(phases_data_sorted, RV_sorted, yerr=RV_errs[instrument], fmt='o', color=colors[i], label=f'{instrument}', zorder=1, alpha=0.2)
            
            # Bin the data and plot the binned points
            bin_centers, binned_RVs, binned_errs = bin_data(phases_data_sorted, RV_sorted, RV_errs[instrument][sort_indices_data])
            plt.errorbar(bin_centers, binned_RVs, yerr=binned_errs, fmt='s', color=colors[i], zorder=2, markersize=6)
        
        # Plot the model
        plt.plot(phases_model_sorted, keplerian_model_RVs_sorted, color='black', label='Keplerian Model', lw=2)

        # Add labels and title
        plt.xlabel('Phase')
        plt.ylabel('Radial Velocity (m/s)')
        plt.title(f'Period = {period:.2f} days')
        plt.legend()
        plt.grid(True)
        
        # Show the plot
        plt.show()

    # Loop through each planet and plot the data
    for idx_planet in range(model.num_planets):
        planet_params = radvel.Parameters(1, basis='per tc e w k')
        planet_params['per1'].value = best_params[data.n_planet_params * idx_planet]
        planet_params['tc1'].value  = best_params[data.n_planet_params * idx_planet + 1]
        
        if data.n_planet_params == 5:
            planet_params['e1'].value = best_params[data.n_planet_params * idx_planet + 2]
            planet_params['w1'].value = best_params[data.n_planet_params * idx_planet + 3]
            planet_params['k1'].value  = best_params[data.n_planet_params * idx_planet + 4]
        else:
            planet_params['e1'].value = 0.0
            planet_params['w1'].value = 1.57
            planet_params['k1'].value  = best_params[data.n_planet_params * idx_planet + 2]

        # Create a Keplerian orbit model for the current planet
        radvel_model = radvel.RVModel(planet_params)

        data_times = {instrument: data.t_rv[instrument] for instrument in data.instruments}
        model_times = {instrument: data.t_mod[instrument] for instrument in data.instruments}
        
        # Subtract the GP model and other planets from the data
        RVs = {}
        for instrument in data.instruments:
            # Predict the RVs for the other planets
            other_planet_rvs = np.zeros_like(data.y_rv[instrument])
            for j in range(model.num_planets):
                if j != idx_planet:
                    other_planet_params = radvel.Parameters(1, basis='per tc e w k')
                    other_planet_params['per1'].value = best_params[data.n_planet_params * j]
                    other_planet_params['tc1'].value = best_params[data.n_planet_params * j + 1]
                    if data.n_planet_params == 5:
                        other_planet_params['e1'].value = best_params[data.n_planet_params * j + 2]
                        other_planet_params['w1'].value = best_params[data.n_planet_params * j + 3]
                        other_planet_params['k1'].value = best_params[data.n_planet_params * j + 4]
                    else: 
                        other_planet_params['e1'].value = 0.0
                        other_planet_params['w1'].value = 1.57
                        other_planet_params['k1'].value = best_params[data.n_planet_params * j + 2]
                    
                    other_planet_model = radvel.RVModel(other_planet_params)
                    other_planet_rvs += other_planet_model(data.t_rv[instrument])

            RVs[instrument] = data.y_rv[instrument] - model.predict(planet_res, data.t_rv, return_components=True)[instrument]['GP'] - other_planet_rvs

        RV_errs = {instrument: data.yerr_rv[instrument] for instrument in data.instruments}
        keplerian_model_RVs = {instrument: radvel_model(data.t_mod[instrument]) for instrument in data.instruments}
        phase_folded_plot(data_times, model_times, RVs, RV_errs, keplerian_model_RVs, best_params[data.n_planet_params * idx_planet], best_params[data.n_planet_params * idx_planet + 1], data.instruments)
        
        plt.savefig(working_path + f'phase_folded_planet_{idx_planet + 1}.png')

    elapsed_time = time.time() - start_time
    logger.info(f'Phase-folded RV curves plotted in {elapsed_time:.2f} seconds.')

    def format_error(value, error):
        """
        Format the value and error to show all digits up to the first non-zero digit of the error.
        """
        error_str = f"{error:.2e}"
        decimal_places = abs(int(error_str.split('e')[-1]))
        format_str = f"{{:.{decimal_places}f}}"
        return format_str.format(value), format_str.format(error)
    
    # Known info about the planets in the system
    planets_info = data.star_info['planets']
    letters = list(data.star_info['planets'].keys())

    for p in range(data.nplanets):
        fig, axes = plt.subplots(1, data.n_planet_params, figsize=(20, 6))
        
        planet_number = p + 1
        
        try:
            # Literature values
            per_lit = planets_info[letters[p]]['period']
            mass_lit = planets_info[letters[p]]['mass']
            ecc_lit = planets_info[letters[p]]['ecc']
            omega_lit = planets_info[letters[p]]['omega']
            
        except: 
            per_lit = 'None'
            mass_lit = 'None'
            ecc_lit = 'None'
            omega_lit = 'None'
        
        # Period
        P_samples = post_samples[:, planet_number * data.n_planet_params - data.n_planet_params]
        P_med = np.median(P_samples)
        P_err = np.percentile(P_samples, [16, 84])
        P_med_str, P_err_str = format_error(P_med, np.mean(np.abs(P_err - P_med)))
        
        # Time of conjunction
        tc_samples = post_samples[:, planet_number * data.n_planet_params - data.n_planet_params + 1]
        tc_med = np.median(tc_samples)
        tc_err = np.percentile(tc_samples, [16, 84])
        tc_med_str, tc_err_str = format_error(tc_med, np.mean(np.abs(tc_err - tc_med)))

        if data.n_planet_params == 5:
            # Eccentricity
            e_samples = post_samples[:, planet_number * data.n_planet_params - data.n_planet_params + 2]
            e_med = np.median(e_samples)
            e_err = np.percentile(e_samples, [16, 84])
            e_med_str, e_err_str = format_error(e_med, np.mean(np.abs(e_err - e_med)))
            
            # Omega
            w_samples = post_samples[:, planet_number * data.n_planet_params - data.n_planet_params + 3]
            w_med = np.median(w_samples)
            w_err = np.percentile(w_samples, [16, 84])
            w_med_str, w_err_str = format_error(w_med, np.mean(np.abs(w_err - w_med)))
            
            k_samples = post_samples[:, planet_number * data.n_planet_params - data.n_planet_params + 4]
        else:
            k_samples = post_samples[:, planet_number * data.n_planet_params - data.n_planet_params + 2]
        
        # Mass    
        M_star = np.random.normal(data.data['star']['M_star'], data.data['star']['M_star_err'], len(k_samples))
        M_samples = radvel.utils.Msini(k_samples, P_samples, M_star, 0)  # Transform into mass 
        M_med = np.median(M_samples)
        M_err = np.percentile(M_samples, [16, 84])
        M_med_str, M_err_str = format_error(M_med, np.mean(np.abs(M_err - M_med)))
        
        # Period histogram
        axes[0].hist(P_samples, bins=50, color='blue', alpha=0.5)
        axes[0].set_xlabel('Period [days]')
        axes[0].set_ylabel('Number of samples')
        axes[0].axvline(P_med, color='red', linestyle='--', label='Median')
        axes[0].axvline(P_err[0], color='green', linestyle='--', label='16th and 84th percentile')
        axes[0].axvline(P_err[1], color='green', linestyle='--')
        axes[0].set_title(f'P = {P_med_str} ± {P_err_str} days')
        if per_lit != 'None': axes[0].axvline(per_lit, color='black', linestyle='--', label='Literature')
        
        # Time of conjunction histogram
        axes[1].hist(tc_samples, bins=50, color='blue', alpha=0.5)
        axes[1].set_xlabel('Time of conjunction [rjd]')
        axes[1].set_ylabel('Number of samples')
        axes[1].axvline(tc_med, color='red', linestyle='--', label='Median')
        axes[1].axvline(tc_err[0], color='green', linestyle='--', label='16th and 84th percentile')
        axes[1].axvline(tc_err[1], color='green', linestyle='--')
        axes[1].set_title(f'tc = {tc_med_str} ± {tc_err_str} rjd')

        # Mass histogram
        axes[2].hist(M_samples, bins=50, color='blue', alpha=0.5)
        axes[2].set_xlabel('Mass (M$_\oplus$)')
        axes[2].set_ylabel('Number of samples')
        axes[2].axvline(M_med, color='red', linestyle='--', label='Median')
        axes[2].axvline(M_err[0], color='green', linestyle='--', label='16th and 84th percentile')
        axes[2].axvline(M_err[1], color='green', linestyle='--')
        axes[2].set_title(f'M = {M_med_str} ± {M_err_str} M$_\oplus$')
        if mass_lit != 'None': axes[2].axvline(mass_lit, color='black', linestyle='--', label='Literature')
        
        axes[0].legend()
        axes[1].legend()
        axes[2].legend()
        
        if data.n_planet_params == 5:
            # Eccentricity histogram
            axes[3].hist(e_samples, bins=50, color='blue', alpha=0.5)
            axes[3].set_xlabel('Eccentricity')
            axes[3].set_ylabel('Number of samples')
            axes[3].set_title(f'e = {e_med_str} ± {e_err_str}')
            axes[3].axvline(e_med, color='red', linestyle='--', label='Median')
            axes[3].axvline(e_err[0], color='green', linestyle='--', label='16th and 84th percentile')
            axes[3].axvline(e_err[1], color='green', linestyle='--')
            if ecc_lit != 'None': axes[3].axvline(ecc_lit, color='black', linestyle='--', label='Literature')
            axes[3].legend()
            
            # Omega histogram
            axes[4].hist(w_samples, bins=50, color='blue', alpha=0.5)
            axes[4].set_xlabel(r'$\omega$ (radians)')
            axes[4].set_ylabel('Number of samples')
            axes[4].set_title(f'$\omega$ = {w_med_str} ± {w_err_str}')
            axes[4].axvline(w_med, color='red', linestyle='--', label='Median')
            axes[4].axvline(w_err[0], color='green', linestyle='--', label='16th and 84th percentile')
            axes[4].axvline(w_err[1], color='green', linestyle='--')
            if omega_lit != 'None': axes[4].axvline(omega_lit, color='black', linestyle='--', label='Literature')
            axes[4].legend()
            
        plt.tight_layout()
        plt.suptitle(f'Planet {planet_number}', fontsize=20, y=1.05)
        plt.savefig(working_path + f'planet_{planet_number}_histogram.png')

    logger.info('All done! Hope you are happy with your results.')
