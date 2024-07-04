# Data Processing with LBL and RV Analysis

This repository provides a guide for using LBL (Line-by-line) analysis for processing astronomical data and conducting RV (Radial Velocity) analysis for star systems. The process involves multiple steps for data extraction, processing, and subsequent RV analysis using different scripts.

## Steps to run a full analysis
### Running lbl
1. Download the target's data from ARI if you want the APERO reduction (Montreal) or from DACE if you want the ESPRESSO reduction (Geneva). If you take the 2D spectra from the ESO archive, make sure you take the ANCILLARY data. 
2. Put the `.tar.gz` file in `run_lbl/data_dir/science/OBJECT/`
3. Unzip the file using this command:
```
tar -xvf filename.tar.gz
```
4. Move all the `.fits` files out of the date directories with this command: 
```
mv */*.fits .
```
5. Comeback in the `/science` directory with `cd ..`
6. Your data will be in directories labeled with the data of each observation. You want to remove all files from these directories so that they are all located directly in the parent directory of the object. 
```
mv */*.fits .
```
7. It is possible that the spectra's file names contain `_A` or `_B`. This denotes the fibers going from the telescope to the detectors. You'll want to only use the data from fiber A. 
```
rm *_B*.fits
``` 
7. The spectra are now ready to be reduces into RV measurements using LBL. Go check the `wrap_{instrument}.py` file and make sure all the info is correct (object name, run params, etc.). Ask Pierrot, Étienne, Neil or Charles for help on how to construct a good wrap file if you are confused.
8. Run the `wrap_{instrument}.py` in a terminal to run lbl on the data. If the reduction takes a long time and you need to close your terminal, use a screen so that LBL carries on even with your computer closed. (See https://linuxize.com/post/how-to-use-linux-screen/ for a tutorial on screens)
9. Once the reduction is over the results should be in the form of a `.rdb` file in the `/lblrdb` directory. The file containing `lbl2` contain per-night bins of the data.  

### Running RV analysis
1. Copy the `/template_star` directory in the `CRAPresults` directory and rename it the name of your star. 
2. Fetch the `.rdb` files in the LBL directory and copy it in the directory `CRAPresults/{star name}/data/`. 
3. Rename the data with the following format: 
```
lbl{bin_label}_{instrument}_{star_name}_{ref_star}{pca_label}
```
- `bin_label` is empty for unbinned data and '2' for binned data
- `pca_label` is empty for non PCA corrected data and '_PCA' for PCA corrected data

4. Fill the `input.yaml` file with target specific information and priors specific for this target. If you want a default large prior, you don't need to change anything to the priors part of the file. 
5. You are now ready to perform a RV analysis of your target with the provided notebooks and python files listed in the section below. 
6. The results of the analysis will be stored in the `CRAPanalysis/` directory of your target. Subfolders with information about your run will be automatically created so that you can save results and compare them together when you change things in the `input.yaml` file. Here is the path architecture of each run: 
```
{model_to_run}/{shared_params}/{nplanets}/{fit_ecc}/{sampler}/
```
- `model_to_run`: - `keplerian+GP`: Family of models containing a keplerian components (planets) AND a GP   component (stellar activity)
                  - `GP_only`: Family of models only fitting a GP. This is to see if the "no planets" scenario is plausible. 
                  - `keplerian_only`: Family of models only fitting planets. Useful for stars with very little to no stellar activity. 

- `shared_params`: Specifies which GP parameters are shared between instruments. For example, the rotation period of the star is contained in parameter number 5 and it may be justified to share it between instruments since it is a physical quantity. Here is the chart to know which digit is connected to which GP parameter: 
    0. `mu` (offset)
    1. `log_wn` (jitter noise)
    2. `log_amp` (amplitude)
    3. `log_lambda` (Coherence length)
    4. `gamma` (Number of sub-structures)
    5. `log_Prot` (Rotation period)

- `nplanets`: Number of planets in the fitted model. 
- `fit_ecc`: If the eccentricity and argument of periastron were fitted for each planet in the model. 
- `sampler`: Which sampler was used to explore the parameter space. Between `emcee` (MCMC) and `dynesty` (Nested Sampling). 
			

## RV analysis scripts
- `activity_indicator_analysis.ipynb`: A notebook used to analyse potential activity signals in the activity indicators (second derivative or differential temperature). If you set the `use_indicator` argument in the input file, the results from this analysis will be used as a prior for the GP fitted in the RV analysis. 
- `keplerian+GP.ipynb`: A notebook that performs the analysis of the RV data with a model containing keplerian signals of planets generated with the `radvel` package and a stellar activity model generated with Quasi-Periodic GP kernels from the `george` package. 
- `GP_only.ipynb`: Only fits a GP model on the RV data. 
- `keplerian_only.ipynb`: Only fits planet models on the RV data. 

*Note: The `CRAPrunner_(...).py` files can be used to lauch many models at the same time, but they can take a long time depending on your needs.*

*Main used packages*: 
- RadVel (https://radvel.readthedocs.io/en/latest/)
- George (https://george.readthedocs.io/en/latest/)
- emcee (https://emcee.readthedocs.io/en/stable/)
- dynesty (https://dynesty.readthedocs.io/en/stable/)


