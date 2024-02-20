# Data Processing with LBL and RV Analysis

This repository provides a guide for using LBL (Line-by-line) analysis for processing astronomical data and conducting RV (Radial Velocity) analysis for star systems. The process involves multiple steps for data extraction, processing, and subsequent RV analysis using different scripts.

## Steps to run a full analysis
### Running lbl
1. Download the target's data from APERO/ESO archive/DACE. Make sure you take ANCILLARY data on ESO archive. 
2. Put the `.tar.gz` file in `lbl_run/data_dir/science/OBJECT/`
3. Unzip the file using this command:
```
tar -xvf filename.tar.gz
```
4. Move all the `.fits` files out of the date directories with this command: 
```
mv */*.fits .
```
5. Comeback in the `/science` directory with `cd ..`
6. Run this command to put all `BLAZE` files in the `calib` directory. 
```
mv */*BLAZE*.fits ../calib
```
7. Go check the `wrap_{instrument}.py` file and make sure all the info is correct (object name, run params, etc.)
8. Run the `wrap_{instrument}.py` in a terminal to run lbl on the data. 
9. The `.rdb` file corresponding to the run should be in the `/lblrdb` directory. 

### Running RV analysis
1. Copy the `/template_star` directory in the `stars` directory and rename it the name of your star. 
2. Fetch the `.rdb` files in the LBL directory and copy it in the directory `stars/{star name}/data/`. 
3. Rename the data with the following format: 
```
lbl{bin_label}_{instrument}_{star_name}_{ref_star}{pca_label}
```
- `bin_label` is empty for unbinned data and '2' for binned data
- `pca_label` is empty for non PCA corrected data and '_PCA' for PCA corrected data

4. Fill the `input.yaml` file with target specific information and priors specific for this target. If you want a default large prior, you don't need to change anything to the file. 
5. Run both `run_julilet.py` and `run_goerge_radvel.py` files to run the RV analysis. The results will be stored in the directories `juliet` and `george_radvel` respectively. 
			

## RV analysis scripts
*Note that all scripts come in `.py` and `.ipynb` format, but the jupyter notebooks are always more up-to-date so export to python if you want to run the script in a terminal.* 

### Juliet
This script uses the `Juliet` package to perform an RV analysis over models with different numbers of planets and different types of activity models. 


Espinoza, N., Kossakowski, D., & Brahm, R. (2019), "juliet: a versatile modelling tool for transiting and non-transiting exoplanetary systems", MNRAS, 490(2), 2262-2283, doi: 10.1093/mnras/stz2688# LBL-reduction-and-RV-analysis

