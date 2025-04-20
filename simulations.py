# %%
from model import train_model
from utils import get, vm_like, get_natural_stats_distribution, reload
import cupy as np
from viz import main_plot,plot_firing_rate_for_stims

params = {
    "j0": 0.3,
    "j1": 3,
    "h0": 0.2,
    "h1": 0.25,
    "lr": 1e-2,
    "noise": 0.001,
    "stim_noise": np.radians(5).item(),
    "count_thresh": 0.9,
    "width_scaling": 1,
    "n_stim": 400,
    "N": 420,
    "T": 1,
    "dt": 1e-2,
    "n_sims": 1,
    "nonlinearity": lambda x: np.maximum(x, 0)

}

np.random.seed(42)
stim_list = get_natural_stats_distribution(int(params["n_stim"])) + np.pi

model_idr = train_model(
    stimuli=stim_list, j0=params["j0"], j1=params["j1"], h0=params["h0"], h1=params["h1"], N=params["N"],
    lr=1e-2, T=params["T"], dt=params["dt"], noise=params["noise"], stim_noise=params["stim_noise"],
    count_thresh=params["count_thresh"], width_scaling=params["width_scaling"], n_sims=params["n_sims"],
    nonlinearity=params["nonlinearity"], tuning_widths=3,
    tuning_func=vm_like, gains=1, update=True, recalculate_connectivity=False, normalize_fr=True,
)
model_ndr = train_model(
    stimuli=stim_list, j0=params["j0"], j1=params["j1"], h0=params["h0"], h1=params["h1"], N=params["N"],
    lr=1e-2, T=params["T"], dt=params["dt"], noise=params["noise"], stim_noise=params["stim_noise"],
    count_thresh=params["count_thresh"], width_scaling=params["width_scaling"], n_sims=params["n_sims"],
    nonlinearity=params["nonlinearity"], tuning_widths=8,
    tuning_func=vm_like, gains=1, update=True, recalculate_connectivity=False, normalize_fr=True,
)

main_plot(stim_list, model_idr, model_ndr)
main_plot(stim_list, model_idr, model_ndr, choice_thresh="h0")
# main_plot(stim_list, model_idr, model_ndr, choice_thresh="bayesian")
#%%
plot_firing_rate_for_stims(model_idr, model_ndr)