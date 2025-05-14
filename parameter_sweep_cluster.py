from model import Model, train_model
from itertools import product
import scipy.stats as stats
from tqdm import tqdm
import pandas as pd
from utils import get, vm_like, get_natural_stats_distribution, get_bias_variance
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew
from viz import main_plot, plot_firing_rate_for_stims, plot_cumulative_firing_rate_for_stims
import sys
import os
# %%

def main(i):
    dt = 1e-2
    N = 420
    T = 1
    n_sims = 1

    j0_list = [0]
    j1_list = get(np.linspace(1, 3, 5))

    h0_list = [1]
    h1_list = get(np.linspace(0.5, 2, 7))
    lr_list = [5e-3]
    noise_list = [0.00]
    stim_noise_list = [0.00]
    count_thresh_list = [0.97, 0.9]
    width_scaling_list = [1]
    n_stim_list = [200, 300, 350, 400, 450, 500]
    distribution_kappa_list = [4, 5, 6, 7]
    min_theta_list = [2,3,4]

    nonlinearity = lambda x: np.maximum(x, 0)
    param_combs = [{"j0": p[0],
                    "j1": p[1],
                    "h0": p[2],
                    "h1": p[3],
                    "lr": p[4],
                    "noise": p[5],
                    "stim_noise": p[6],
                    "count_thresh": p[7],
                    "width_scaling": p[8],
                    "n_stim": p[9],
                    "distribution_kappa": p[10],
                    "min_theta":p[11]
                    } for p in product(
        j0_list, j1_list, h0_list, h1_list, lr_list, noise_list, stim_noise_list,
        count_thresh_list, width_scaling_list, n_stim_list, distribution_kappa_list,
        min_theta_list
    )]

    results = []
    results.append(run_param_comb(N, T, dt, n_sims, nonlinearity, param_combs[i]))

    results = pd.DataFrame(results)
    results.to_csv(os.path.join("results",f"model_res_{i}.csv"))
    print("Done with params:\n", param_combs[i])
    

def run_param_comb(N, T, dt, n_sims, nonlinearity, params):
    res = {}
    res.update(params)
    np.random.seed(42)
    stim_list = get_natural_stats_distribution(params["n_stim"],kappa = params['distribution_kappa']) + np.pi
    model_idr, learning_thetas_idr, learning_tuning_widths_idr = train_model(
        stimuli=stim_list, j0=params["j0"], j1=params["j1"], h0=params["h0"], h1=params["h1"], N=N,
        lr=params["lr"], T=T, dt=dt, noise=params["noise"], stim_noise=params["stim_noise"],
        count_thresh=params["count_thresh"], width_scaling=params["width_scaling"], n_sims=n_sims,
        nonlinearity=nonlinearity, tuning_widths=params['min_theta'],
        tuning_func=vm_like, gains=1, update=True, recalculate_connectivity=True, normalize_fr=True,
        limit_width=False
    )
    model_ndr, learning_thetas_ndr, learning_tuning_widths_ndr = train_model(
        stimuli=stim_list, j0=params["j0"], j1=params["j1"], h0=params["h0"], h1=params["h1"], N=N,
        lr=params["lr"], T=T, dt=dt, noise=params["noise"], stim_noise=params["stim_noise"],
        count_thresh=params["count_thresh"], width_scaling=params["width_scaling"], n_sims=n_sims,
        nonlinearity=nonlinearity, tuning_widths=8,
        tuning_func=vm_like, gains=1, update=True, recalculate_connectivity=True, normalize_fr=True,
        limit_width=False
    )
    bias_idr, stimuli = get_bias_and_emd(model_idr, params, res, stim_list, "IDR")
    bias_ndr, _ = get_bias_and_emd(model_ndr, params, res, stim_list, "NDR")
    calculate_corrent_incorrect_bias(bias_idr, bias_ndr, res, stimuli)
    calculate_skews(model_idr, model_ndr, res)
    title = "_".join(f"{k}-{v}" for k, v in params.items())
    try:
        main_plot(stim_list, model_idr, model_ndr, savename = title)
        plt.close('all')
    except Exception as e:
        pass
    try:
        main_plot(stim_list, model_idr, model_ndr, choice_thresh="h0", savename= title)
        plt.close('all')
    except Exception as e:
        pass
    try:
        plot_firing_rate_for_stims(model_idr, model_ndr, savename= title)
        plt.close('all')
    except Exception as e:
        pass
    try:
        plot_cumulative_firing_rate_for_stims(model_idr, model_ndr, savename= title, choice_thresh="h0")
        plt.close('all')
    except Exception as e:
        pass
    try:
        plot_cumulative_firing_rate_for_stims(model_idr, model_ndr, choice_thresh=0,
                                              savename= title)
        plt.close('all')
    except Exception as e:
        pass
    return res


def calculate_skews(model_idr, model_ndr, res):
    oblique_stim = 3 * np.pi / 4
    cardinal_stim = np.pi
    near_oblique_stim = oblique_stim + np.pi / 12
    near_cardinal_stim = cardinal_stim + np.pi / 12
    idr_resps = []
    ndr_resps = []
    for stim in [oblique_stim, near_oblique_stim, cardinal_stim, near_cardinal_stim]:
        idr_resps.append(get(np.squeeze(model_idr.run(stim))))
        ndr_resps.append(get(np.squeeze(model_ndr.run(stim))))
    idr_skews = []
    ndr_skews = []
    idr_n_peaks = []
    ndr_n_peaks = []
    idr_peaks_strengths = []
    ndr_peaks_strengths = []
    idr_skews_h0 = []
    ndr_skews_h0 = []
    for idr_resp, ndr_resp in zip(idr_resps, ndr_resps):
        idr_skews.append(skew(idr_resp, nan_policy='omit'))
        ndr_skews.append(skew(ndr_resp, nan_policy='omit'))
        
        idr_peaks = np.where(np.diff(np.sign(np.gradient(idr_resp)))==-2)[0]
        ndr_peaks = np.where(np.diff(np.sign(np.gradient(ndr_resp)))==-2)[0]
        
        idr_n_peaks.append(idr_peaks.size)
        ndr_n_peaks.append(ndr_peaks.size)

        idr_peaks_strengths.append(idr_resp[idr_peaks+1])
        ndr_peaks_strengths.append(idr_resp[ndr_peaks+1])
        
        idr_resp[idr_resp < model_idr.h0] = 0
        ndr_resp[ndr_resp < model_ndr.h0] = 0

        idr_skews_h0.append(skew(idr_resp, nan_policy='omit'))
        ndr_skews_h0.append(skew(ndr_resp, nan_policy='omit'))
    for i, (idr_skew, idr_skew_h0, ndr_skew, ndr_skew_h0, stim_name) in enumerate(zip(
            idr_skews, idr_skews_h0, ndr_skews, ndr_skews_h0, ["OBLIQUE", "NEAR_OBLIQUE", "CARDINAL", "NEAR_CARDINAL"]
    )):
        res[stim_name + "_IDR_SKEW"] = idr_skew
        res[stim_name + "_IDR_SKEW_H0"] = idr_skew_h0
        res[stim_name + "_NDR_SKEW"] = ndr_skew
        res[stim_name + "_NDR_SKEW_H0"] = ndr_skew_h0
        res[stim_name + "_IDR_N_PEAKS"] = idr_n_peaks[i]
        res[stim_name + "_IDR_PEAK_STRENGTH"] = idr_peaks_strengths[i]
        res[stim_name + "_NDR_N_PEAKS"] = ndr_n_peaks[i]
        res[stim_name + "_NDR_PEAK_STRENGTH"] = ndr_peaks_strengths[i]


def calculate_corrent_incorrect_bias(bias_idr, bias_ndr, res, stimuli):
    max_bias = max(np.abs(np.array(bias_idr)).max().item(), np.abs(np.array(bias_ndr)).max().item())
    if max_bias != 0:
        correct_bias = get(np.squeeze(np.sin(4 * stimuli))) * max_bias
        incorrect_bias = get(np.squeeze(np.sin(np.pi + (4 * stimuli)))) * max_bias
        if np.abs(np.array(bias_idr)).max().item() != 0:
            res["IDR_CORRECT_BIAS_MSE"] = ((correct_bias - get(bias_idr)) ** 2).mean()
            res["IDR_INCORRECT_BIAS_MSE"] = ((incorrect_bias - get(bias_idr)) ** 2).mean()
        else:
            res["IDR_CORRECT_BIAS_MSE"] = np.nan
            res["IDR_INCORRECT_BIAS_MSE"] = np.nan
        if np.abs(np.array(bias_ndr)).max().item() != 0:
            res["NDR_CORRECT_BIAS_MSE"] = ((correct_bias - get(bias_ndr)) ** 2).mean()
            res["NDR_INCORRECT_BIAS_MSE"] = ((incorrect_bias - get(bias_ndr)) ** 2).mean()
        else:
            res["NDR_CORRECT_BIAS_MSE"] = np.nan
            res["NDR_INCORRECT_BIAS_MSE"] = np.nan
    else:
        res["IDR_CORRECT_BIAS_MSE"] = np.nan
        res["IDR_INCORRECT_BIAS_MSE"] = np.nan
        res["NDR_CORRECT_BIAS_MSE"] = np.nan
        res["NDR_INCORRECT_BIAS_MSE"] = np.nan


def get_bias_and_emd(model, params, res, stim_list, model_type):
    try:
        bias, variance, stimuli, bias_ci = get_bias_variance(model, sigma=1, choice_thresh="h0")
        res[f"EMD_{model_type}_STIM"] = stats.wasserstein_distance(get(stim_list), get(model.theta))
        res[f"EMD_{model_type}_UNIFORM"] = stats.wasserstein_distance(
            get(np.random.uniform(0, 2 * np.pi, size=params["n_stim"] * 4)), get(model.theta))
    except Exception as e:
        bias, variance, stimuli, bias_ci = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        res[f"EMD_{model_type}_STIM"] = np.nan
        res[f"EMD_{model_type}_UNIFORM"] = np.nan
    return bias, stimuli


if __name__ == '__main__':
    i = int(sys.argv[1])
    main(i)
