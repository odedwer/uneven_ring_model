from model import Model, train_model
from itertools import product
import scipy.stats as stats
from tqdm import tqdm
import pandas as pd
from utils import get, vm_like, get_natural_stats_distribution, get_bias_variance
import cupy as np

j0_list = np.linspace(0.1, 0.5, 3).get()
j1_list = np.linspace(2, 3, 4).get()

h0_list = np.linspace(0.1, 0.5, 3).get()
h1_list = np.linspace(0.1, 0.5, 5).get()
lr_list = [5e-3]
noise_list = [0.00]
stim_noise_list = [0.00]
count_thresh_list = [0.97, 0.9]
width_scaling_list = [1]
n_stim_list = [300]

dt = 1e-2
N = 420
T = 1
noise = 0.00
n_sims = 1

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
                "n_stim": p[9]
                } for p in product(
    j0_list, j1_list, h0_list, h1_list, lr_list, noise_list, stim_noise_list,
    count_thresh_list, width_scaling_list, n_stim_list
)]
#%%
results = []
for params in tqdm(param_combs):
    res = {}
    res.update(params)

    np.random.seed(42)
    stim_list = get_natural_stats_distribution(params["n_stim"]) + np.pi
    model_idr = train_model(
        stimuli=stim_list, j0=params["j0"], j1=params["j1"], h0=params["h0"], h1=params["h1"], N=N,
        lr=params["lr"], T=T, dt=dt, noise=params["noise"], stim_noise=params["stim_noise"],
        count_thresh=params["count_thresh"], width_scaling=params["width_scaling"], n_sims=n_sims,
        nonlinearity=nonlinearity, tuning_widths=2,
        tuning_func=vm_like, gains=1, update=True, recalculate_connectivity=False, normalize_fr=True,
    )

    model_ndr = train_model(
        stimuli=stim_list, j0=params["j0"], j1=params["j1"], h0=params["h0"], h1=params["h1"], N=N,
        lr=params["lr"], T=T, dt=dt, noise=params["noise"], stim_noise=params["stim_noise"],
        count_thresh=params["count_thresh"], width_scaling=params["width_scaling"], n_sims=n_sims,
        nonlinearity=nonlinearity, tuning_widths=8,
        tuning_func=vm_like, gains=1, update=True, recalculate_connectivity=False, normalize_fr=True,
    )
    try:
        bias_idr, variance_idr, stimuli, bias_ci_idr = get_bias_variance(model_idr, sigma=1)
        res["EMD_IDR_STIM"] = stats.wasserstein_distance(stim_list.get(), model_idr.theta.get())
        res["EMD_IDR_UNIFORM"] = stats.wasserstein_distance(
            np.random.uniform(0, 2 * np.pi, size=params["n_stim"] * 4).get(), model_idr.theta.get())
    except Exception as e:
        bias_idr, variance_idr, stimuli, bias_ci_idr = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        res["EMD_IDR_STIM"] = np.nan
        res["EMD_IDR_UNIFORM"] = np.nan

    try:
        bias_ndr, variance_ndr, stimuli, bias_ci_ndr = get_bias_variance(model_ndr, sigma=1)
        res["EMD_NDR_STIM"] = stats.wasserstein_distance(stim_list.get(), model_ndr.theta.get())
        res["EMD_NDR_UNIFORM"] = stats.wasserstein_distance(
            np.random.uniform(0, 2 * np.pi, size=params["n_stim"] * 4).get(), model_ndr.theta.get())
    except Exception as e:
        bias_ndr, variance_ndr, stimuli, bias_ci_ndr = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
        res["EMD_IDR_STIM"] = np.nan
        res["EMD_IDR_UNIFORM"] = np.nan

    max_bias = max(np.abs(np.array(bias_idr)).max().item(), np.abs(np.array(bias_ndr)).max().item())
    if max_bias != 0:
        correct_bias = np.squeeze(np.sin(4 * stimuli)).get() * max_bias
        incorrect_bias = np.squeeze(np.sin(np.pi + (4 * stimuli))).get() * max_bias
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

    results.append(res)
    pd.DataFrame(results).to_csv("model_res.csv", index=True)

results = pd.DataFrame(results)
results.to_csv("model_res.csv", index=True)