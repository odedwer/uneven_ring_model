from model import train_model
from utils import get, vm_like, get_natural_stats_distribution, reload, circ_distance
import cupy as np
import matplotlib.pyplot as plt

# from viz import main_plot, plot_firing_rate_for_stims, plot_cumulative_firing_rate_for_stims, preferred_orientation_plot

params = {
    "j0": 0,
    "j1": 3,
    "h0": 0.75,
    "h1": 1,
    "lr": 0.005,
    "noise": 0.0,
    "stim_noise": 0,
    "count_thresh": 0.9,
    "width_scaling": 1,
    "n_stim": 450,
    "N": 420,
    "T": 2,
    "dt": 1e-2,
    "n_sims": 1,
    "nonlinearity": lambda x: np.maximum(x, 0),
    "recalculate_connectivity": True,
    "limit_width": False
}

np.random.seed(42)
stim_list = get_natural_stats_distribution(int(params["n_stim"]), kappa=4.5) + np.pi

model, learning_thetas, learning_tuning_widths, connectivity_list = train_model(
    stimuli=stim_list, j0=params["j0"], j1=params["j1"], h0=params["h0"], h1=params["h1"], N=params["N"],
    lr=params["lr"], T=params["T"], dt=params["dt"], noise=params["noise"], stim_noise=params["stim_noise"],
    count_thresh=params["count_thresh"], width_scaling=params["width_scaling"], n_sims=params["n_sims"],
    nonlinearity=params["nonlinearity"], tuning_widths=8,
    tuning_func=vm_like, gains=1, update=True, recalculate_connectivity=params["recalculate_connectivity"],
    normalize_fr=True, limit_width=params["limit_width"], use_tqdm=True
)


def tuning_derivative(theta, theta_i, kappa_i):
    """
    Compute the derivative of the tuning function.
    """
    return -2 * np.sin(theta_i - theta) * kappa_i * np.exp(kappa_i * (np.cos(theta_i - theta) - 1))


def get_M(J, r_star):
    g_tag = np.squeeze((r_star > 0).astype(int))
    Dg = np.diag(g_tag)
    inv = np.linalg.inv(np.eye(Dg.shape[0]) - J @ Dg)
    return inv @ Dg


fi_stim_list = np.linspace(0, 2 * np.pi, 361)  # 0 to 2*pi in 1 degree increments
fi = np.zeros((len(learning_tuning_widths), fi_stim_list.size),dtype=float)

for t in range(len(learning_tuning_widths)):
    for i in range(len(fi_stim_list)):
        f_tag = tuning_derivative(np.pi, learning_thetas[0], learning_tuning_widths[0])

        model.J = connectivity_list[t]
        model.tuning_widths = learning_tuning_widths[t]
        model.thetas = learning_thetas[t]

        r_star = model.run(fi_stim_list[i])
        M = get_M(connectivity_list[0], r_star)

        fi[t] = (f_tag @ M.T) @ (M @ f_tag)
