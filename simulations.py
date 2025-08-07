# %%
from model import train_model
from utils import get, vm_like, get_natural_stats_distribution, reload, circ_distance
import cupy as np
import matplotlib.pyplot as plt
from viz import main_plot, plot_firing_rate_for_stims, plot_cumulative_firing_rate_for_stims, preferred_orientation_plot

params = {
    "j0": 0,
    "j1": 3,
    "h0": 1.5,
    "h1": 1,
    "lr": 0.005,
    "noise": 0.0,
    "stim_noise": 0,
    "count_thresh": 0.95,
    "width_scaling": 1,
    "n_stim": 200,
    "N": 420,
    "T": 1,
    "dt": 1e-2,
    "n_sims": 2,
    "nonlinearity": lambda x: np.maximum(x, 0),
    "recalculate_connectivity": True,
    "limit_width": False
}

np.random.seed(42)
stim_list = get_natural_stats_distribution(int(params["n_stim"]), kappa=4.5, n_sims=params['n_sims']) + np.pi

# model_idr,idr_learning_thetas, idr_learning_tuning_widths = train_model(
#     stimuli=stim_list, j0=params["j0"], j1=params["j1"], h0=params["h0"], h1=params["h1"], N=params["N"],
#     lr=params["lr"], T=params["T"], dt=params["dt"], noise=params["noise"], stim_noise=params["stim_noise"],
#     count_thresh=params["count_thresh"], width_scaling=params["width_scaling"], n_sims=params["n_sims"],
#     nonlinearity=params["nonlinearity"], tuning_widths=3,
#     tuning_func=vm_like, gains=1, update=True, recalculate_connectivity=params["recalculate_connectivity"],
#     normalize_fr=True, limit_width=params["limit_width"],use_tqdm=True,save_process=False
# )
model_ndr, ndr_learning_thetas, ndr_learning_tuning_widths, ndr_learning_connectivity = train_model(
    stimuli=stim_list, j0=params["j0"], j1=params["j1"], h0=params["h0"], h1=params["h1"], N=params["N"],
    lr=params["lr"], T=params["T"], dt=params["dt"], noise=params["noise"], stim_noise=params["stim_noise"],
    count_thresh=params["count_thresh"], width_scaling=params["width_scaling"], n_sims=params["n_sims"],
    nonlinearity=params["nonlinearity"], tuning_widths=8,
    tuning_func=vm_like, gains=1, update=True, recalculate_connectivity=params["recalculate_connectivity"],
    normalize_fr=True, limit_width=params["limit_width"], use_tqdm=True, save_process=False
)

title = "recalc_{}_limit_{}_noise".format(params["recalculate_connectivity"], params["limit_width"])

# main_plot(stim_list, model_idr, model_ndr, savename="main_plot_0_thresh_" + title)
main_plot(stim_list, model_idr, model_ndr, choice_thresh="h0", savename="main_plot_h0_thresh_" + title)
plt.show()
# main_plot(stim_list, model_idr, model_ndr, choice_thresh="bayesian", savename="main_plot_bayes_thresh_" + title)
# %% plot the different simulations theta distributions
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(get(np.arange(model_ndr.N)), get(np.rad2deg(np.sort(model_ndr.theta)).T), label="IDR", color="blue")
plt.show()
# %%
plot_firing_rate_for_stims(model_idr, model_ndr, savename="fr_" + title)
plot_cumulative_firing_rate_for_stims(model_idr, model_ndr, savename="cumulative_fr_h0_" + title)
plot_cumulative_firing_rate_for_stims(model_idr, model_ndr, choice_thresh=0, savename="cumulative_fr_0_" + title)
plot_cumulative_firing_rate_for_stims(model_idr, model_ndr, choice_thresh="bayesian",
                                      savename="cumulative_fr_bayes_" + title)
preferred_orientation_plot(model_idr, model_ndr, savename="preferred_orientation_" + title)

# %%
import matplotlib.pyplot as plt

plt.hist(get(stim_list), bins=90, density=True)
plt.xticks(
    get(np.linspace(0, 2 * np.pi, 5)),
    [f"{get(np.rad2deg(x)):.0f}" for x in np.linspace(0, 2 * np.pi, 5)],
    rotation=45
)
plt.show()

# %%
idr_learning_thetas = np.array(idr_learning_thetas)
idr_learning_thetas_diff = circ_distance(idr_learning_thetas[1:], idr_learning_thetas[:-1])

ndr_learning_thetas = np.array(ndr_learning_thetas)
ndr_learning_thetas_diff = circ_distance(ndr_learning_thetas[1:], ndr_learning_thetas[:-1])

idr_learning_tuning_widths = np.array(idr_learning_tuning_widths)
idr_learning_tuning_widths_diff = circ_distance(idr_learning_tuning_widths[1:], idr_learning_tuning_widths[:-1])

ndr_learning_tuning_widths = np.array(ndr_learning_tuning_widths)
ndr_learning_tuning_widths_diff = circ_distance(ndr_learning_tuning_widths[1:], ndr_learning_tuning_widths[:-1])
