import numpy as np
import matplotlib.pyplot as plt
#import requests
#from bs4 import BeautifulSoup
from typing import List, Optional, Tuple, Dict
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.timeseries import LombScargle
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import pickle
import requests


### THANKS THOMAS VANDAL
def sigma_clip_tbl(
    tbl: Table, qty_list: List[str], sigma: float = 5.0, **kwargs
) -> np.ndarray[bool]:
    mask = np.zeros(len(tbl), dtype=bool)
    for qty in qty_list:
        mask |= sigma_clip(tbl[qty], sigma=sigma, **kwargs).mask
    return mask

def plot_error_dist(
    tbl: Table, qty_list: List[str],
    qty_labels = {
        "vrad": "RV",
        "d2v": "D2V",
        "d3v": "D3V",
        "DTEMP": "$\Delta$T",
    },
    qty_units = {
        "vrad": "m/s",
        "d2v": "m$^2$ / s$^2$",
        "d3v": "m$^3$ / s$^3$",
        "DTEMP": "K",
    },
    quantile_cut: float = 0.95
) -> Tuple[Figure, Axes]:
    fig, axes = plt.subplots(ncols=len(qty_list), figsize=(24, 6))
    for i, qty in enumerate(qty_list):
        err_cut = np.quantile(tbl[f"s{qty}"], quantile_cut)
        axes[i].hist(tbl[f"s{qty}"], bins=50)
        axes[i].axvline(
            err_cut, color="r", linestyle="--", label=f"Quantile {quantile_cut}"
        )
        axes[i].set_title(f"{qty_labels[qty]} Error Histogram")
        axes[i].set_xlabel(f"{qty_labels[qty]} Error [{qty_units[qty]}]")
    axes[0].legend()
    return fig, axes


def error_quantile_clip(
    tbl: Table, qty_list: List[str], quantile_cut: float = 0.95, inc_label='s'
) -> np.ndarray[bool]:
    mask = np.zeros(len(tbl), dtype=bool)
    for qty in qty_list:
        err_cut = np.quantile(tbl[f"{inc_label}{qty}"], quantile_cut)
        mask |= tbl[f"{inc_label}{qty}"] > err_cut
    return mask


def sigma(tmp):
    # return a robust estimate of 1 sigma
    sig1 = 0.682689492137086
    p1 = (1-(1-sig1)/2)*100
    return (np.nanpercentile(tmp,p1) - np.nanpercentile(tmp,100-p1))/2.0


def find_mass(K, P, e, i, Ms): 
    # K in m/s, P in days, i in degrees, Ms in solar masses
    G = 6.67408e-11 # m3 kg-1 s-2
    P = P*86400 # convert to seconds
    i = np.radians(i) # convert to radians
    Ms = Ms*1.989e30 # convert to kg
    
    M_p = ((((P/(2*np.pi*G))**(1/3))*K*(Ms**(2/3)))/np.sin(i))
    
    print(f"La masse de la planète est de {M_p/5.972e24} masses terrestres")
    return M_p
    
    
### THANKS THOMAS VANDAL 

def plot_timeseries(
    tbl: Table,
    qty_list: List[str] = ["vrad", "d2v", "d3v", "DTEMP"],
    qty_labels = {
        "vrad": "RV",
        "d2v": "D2V",
        "d3v": "D3V",
        "DTEMP": "$\Delta$T",
    },
    qty_units = {
        "vrad": "m/s",
        "d2v": "m$^2$ / s$^2$",
        "d3v": "m$^3$ / s$^3$",
        "DTEMP": "K",
    },
    inc_label='s',
    plot_ls: bool = True,
    ylog: bool = False,
    err_label: Optional[str] = None,
    err_fmt: str = "k.",
    target_fap: float = 0.01,
    fig: Optional[Figure] = None,
    axes: Optional[np.ndarray[Axes]] = None,
    star_rotation = None,
    candidate_planets: Dict[str, float] = None, 
    t_off: float = 0,
    tlabel: str = "rjd",
    color = 'blue',
) -> Tuple[Figure, Axes]:
    ncols = 2 if plot_ls else 1
    num_qty = len(qty_list)

    if fig is None:
        fig, axes = plt.subplots(
            nrows=num_qty,
            ncols=ncols,
            figsize=(12 * ncols, 3.5 * num_qty),
            sharex="col",
            squeeze=False,
        )
    elif axes is None:
        axes = np.array(fig.axes).reshape((num_qty, ncols))

    for i, qty in enumerate(qty_list):
        t = tbl[tlabel].data
        y = tbl[qty].data
        yerr = tbl[f"{inc_label}{qty}"].data
        axes[i, 0].errorbar(t, y, yerr=yerr, fmt=err_fmt, label=err_label, color=color)
        axes[i, 0].set_ylabel(f"{qty_labels[qty]} [{qty_units[qty]}]")

        if plot_ls:
            #Remove NaNs before performing Lomb-Scargle
            nan_mask = ~np.isnan(y)
            y = y[nan_mask]
            t = t[nan_mask]
            yerr = yerr[nan_mask]
            
            ls = LombScargle(t, y, dy=yerr)
            fap = ls.false_alarm_level(target_fap)
            freq, power = ls.autopower(maximum_frequency=1.0)
            period = 1 / freq

            ls_window = LombScargle(
                t, np.ones_like(y), dy=yerr, fit_mean=False, center_data=False
            )
            power_window = ls_window.power(freq)

            axes[i, 1].plot(period, power, color, label="Periodogram" if i == 0 else None)
            axes[i, 1].plot(
                period, power_window, "C1", label="Window Function" if i == 0 else None
            )
            axes[i, 1].axhline(
                fap,
                linestyle="--",
                color="r",
                label=f"{target_fap}% FA level" if i == 0 else None,
                alpha=0.5,
            )

            # Loop through candidate planets and their periods and add them to the plot
            if candidate_planets:
                for planet, period_value in candidate_planets.items():
                    axes[i, 1].axvline(
                        period_value['period'],
                        linestyle="--",
                        label=f"{planet}" if i == 0 else None,
                        alpha=0.5,
                    )
                    
            # Add star rotation period to plot
            if star_rotation:
                axes[i, 1].axvline(
                    star_rotation,
                    linestyle="--",
                    color="C5",
                    label="Rotation" if i == 0 else None,
                    alpha=0.7,
                )

            axes[i, 1].set_xscale("log")
            if ylog:
                axes[i, 1].set_yscale("log")
                axes[i, 1].set_ylim((1e-2, None))
            axes[i, 1].set_title(f"{qty_labels[qty]} LS Periodogram")
            axes[i, 1].set_ylabel("Power")
            # Get handles for periodograms only
            handles = []
            labels = []
            for ax in axes[:, 1].flatten():
                handles_sub, labels_sub = ax.get_legend_handles_labels()
                handles.extend(handles_sub)
                labels.extend(labels_sub)
            # Remove duplicate handles and labels
            handles, labels = zip(
                *sorted(set(zip(handles, labels)), key=lambda x: labels.index(x[1]))
            )

            fig.legend(handles, labels, loc="upper right")
    axes[-1, 0].set_xlabel(f"Time [BJD - {t_off:.0f}]")
    if plot_ls:
        axes[-1, 1].set_xlabel("Period [d]")

    return fig, axes



### Thanks CHATGPT

def remove_nans_from_table(table, last_column = None):
    # Create a new dictionary to store the cleaned arrays
    cleaned_table = Table()
    NaNs_array = np.full(np.shape(table['vrad']), True)
    
    i=0
    for key in table.keys():
        if i<=6:
            inans = np.argwhere(np.isnan(table[key]))
            NaNs_array[inans] = False
            
        i+=1
        
    for key in table.keys():
            cleaned_table[key] = table[key][NaNs_array]
    
    return cleaned_table

### Saving and loading dictionnaries
# SAVE
def save_dict(dictionnary, file_path):
    with open(file_path, "wb") as pkl_handle:
        pickle.dump(dictionnary, pkl_handle)

# LOAD
def load_dict(file_path):
    with open(file_path, "rb") as pkl_handle:
        output = pickle.load(pkl_handle)
        return output


