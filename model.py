from tqdm import tqdm

from utils import get_natural_stats_distribution, circ_distance, is_iterable, vm_like

# import numpy as np
import cupy as np
import pandas as pd



class Model:
    def __init__(self, j0, j1, h0, h1, N, gains, tuning_widths, tuning_func, lr, count_thresh=0, width_scaling=1, T=1,
                 dt=1e-2, noise=0., stim_noise=0., n_sims=1000, nonlinearity=lambda x: x):
        self.j0 = j0
        self.j1 = j1
        self.h0 = h0
        self.h1 = h1
        self.N = N
        self.T = T
        self.dt = dt
        self.time = np.arange(0, T + dt, dt)
        self.n_sims = n_sims
        if is_iterable(gains):
            self.gains = np.array(gains)
        else:
            self.gains = np.ones(N).astype(float) * gains
        self.theta = np.linspace(0, 2 * np.pi, N)
        if is_iterable(tuning_widths):
            self.tuning_widths = np.array(tuning_widths).astype(float)
        else:
            self.tuning_widths = np.ones(N).astype(float) * tuning_widths
        self.tuning_func = tuning_func
        self.J = (1 / self.N) * (self.j0 + self.j1 * np.cos(self.theta[:, None] - self.theta[None, :]))
        self.r = np.zeros((self.time.size, N, n_sims))
        self.noise = noise
        self._dW = np.random.normal(loc=0.0, scale=np.sqrt(dt)* self.noise, size=(self.time.size - 1, N, self.n_sims))
        self.lr = lr
        self.width_scaling = width_scaling
        self.count_thresh = count_thresh
        self.base_factor = self.get_near_factor()
        self.stim_noise = stim_noise
        self.stim_history = []
        self.nonlinearity = nonlinearity

    def deterministic_func(self, y, stim):
        return -y + self.nonlinearity(((self.h0 + self.h1 * self.tuning_func((self.theta - stim),
                                                                             self.tuning_widths)) * self.gains)[:,
                                      None] + self.J @ y)

    def euler_maruyama(self, y, stim, i):
        return y + self.deterministic_func(y, stim) * self.dt + self._dW[i - 1]

    def run(self, stim):
        self.stim_history.append(stim)
        # add noise to the stimulus, different noise for each neuron and time point

        noisy_stim = stim + np.random.normal(0, self.stim_noise, (self.time.size, self.N))
        noisy_stim[noisy_stim < 0] = 2 * np.pi + noisy_stim[noisy_stim < 0]
        noisy_stim %= 2 * np.pi
        for i in range(1, self.time.size):
            self.r[i] = self.euler_maruyama(self.r[i - 1], noisy_stim[i], i)
        return self.r[-1].copy()

    def update(self, normalize_fr=True, recalculate_connectivity=False):
        fr = self.r[-1, :, :]
        # Perform hebbian learning of self.theta based on the firing rates and the last stimulus
        if normalize_fr:
            fr = (fr - fr.min()) / (fr.max() - fr.min())
        dist = circ_distance(self.stim_history[-1], self.theta)
        old_theta, old_widths = self.theta.copy(), self.tuning_widths.copy()
        self.theta += self.lr * fr.mean(-1) * dist
        self.theta[self.theta < 0] = (2 * np.pi) + self.theta[self.theta < 0]
        self.theta %= 2 * np.pi
        self.tuning_widths *= np.abs((self.width_scaling * (
                (self.get_near_factor().astype(float) / self.base_factor.astype(float)) - 1)) + 1)
        if recalculate_connectivity:
            self.J = (1 / self.N) * (self.j0 + self.j1 * np.cos(self.theta[:, None] - self.theta[None, :]))
            self.J[np.arange(self.N), np.arange(self.N)] = 0
        return old_theta, old_widths

    def get_near_factor(self):
        resps = self.tuning_func(self.theta[:, None] - self.theta[None, :], self.tuning_widths[:, None])
        # for each neuron in the response, count the number of self.theta values that have a response greater than self.count_thresh
        near_count = (resps > self.count_thresh).sum(1)
        return near_count


def train_model(stimuli, j0, j1, h0, h1, N, lr, T, dt, noise, stim_noise,
                count_thresh, width_scaling, n_sims, nonlinearity,
                tuning_widths, tuning_func, gains,
                update, recalculate_connectivity, normalize_fr, use_tqdm=False
                ):
    """
    Train the model with the given parameters
    :param params: dictionary of parameters
    :return: trained model
    """
    model = Model(j0=j0, j1=j1, h0=h0, h1=h1, N=N, lr=lr, T=T, dt=dt, noise=noise, stim_noise=stim_noise,
                  count_thresh=count_thresh, width_scaling=width_scaling, n_sims=n_sims, nonlinearity=nonlinearity,
                  tuning_widths=tuning_widths, tuning_func=tuning_func, gains=gains)
    for stim in tqdm(stimuli) if use_tqdm else stimuli:
        model.run(stim)
        if update:
            model.update(recalculate_connectivity=recalculate_connectivity, normalize_fr=normalize_fr)
    return model

# %%
# param_sweep_res = pd.read_csv("model_res.csv", index_col=0).dropna()
# good_rows = (
#         (param_sweep_res["EMD_IDR_UNIFORM"] > 0.067469)
#         & (param_sweep_res["EMD_NDR_STIM"] < 0.046875)
#     # & (param_sweep_res["IDR_CORRECT_BIAS_MSE"] < 1e-4)
#     # & (param_sweep_res["IDR_CORRECT_BIAS_MSE"] < param_sweep_res["IDR_INCORRECT_BIAS_MSE"])
#     # & (param_sweep_res["NDR_CORRECT_BIAS_MSE"] < 1e-4)
#     # & (param_sweep_res["NDR_CORRECT_BIAS_MSE"] < param_sweep_res["NDR_INCORRECT_BIAS_MSE"])
# )
# good_params = param_sweep_res[good_rows]
# %%

# %%
# animate_tuning_widths("model", get(all_idr_tuning), get(all_ndr_tuning), get(all_idr_theta), get(all_ndr_theta),
#                       kappa_wide, kappa_sharp)
# %% plot the ndr model relative change in width
# kappa_sharp = 8
# kappa_wide = 2
# fig, ax = plt.subplots(figsize=(7, 5))
# ax.plot(model_ndr.theta.get(), model_ndr.tuning_widths.get() / kappa_sharp, color=NT_COLOR, label="NDR", linewidth=3)
# # set xlabels to be in radians, 0 till 2pi
# ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
#               ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
# # plot gray dotted vlines at cardiscatternal orientations, and a solid black like at 0
# ax.vlines(np.pi / 2, get(model_ndr.tuning_widths.get().min()).item() / kappa_sharp,
#           get(model_ndr.tuning_widths.get().max()).item() / kappa_sharp,
#           color='gray', linestyle='--', lw=2)
# ax.set_xlim([0, np.pi])
# ax.hlines(1, 0, 2 * np.pi, linestyles="--", colors='black')
# ax.set_xlabel("Preferred Orientation")
# ax.set_ylabel("Relative change in width")
# plt.tight_layout()
# plt.show()
#
# # %% plot the tuning curve of the NDR model at pi/2
# fig, ax = plt.subplots(figsize=(6, 4))
# # ax.axis("off")
# # plot one gaussian with high variance in ASD_COLOR and one with low variance in NT_COLOR
# x = np.linspace(-np.pi, np.pi, 1000)
# max_tuning = model_ndr.tuning_func(x=x, kappa=model_ndr.tuning_widths.max().item())
# min_tuning = model_ndr.tuning_func(x=x, kappa=model_ndr.tuning_widths.min().item())
#
# ax.plot(get(x), get(max_tuning), color=NT_COLOR, linewidth=3)
# ax.plot(get(x), get(min_tuning), color=NT_COLOR, linewidth=3, linestyle="--")
# ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
#               ["$-\pi$", r"$-\frac{\pi}{2}$", "$0$", r"$\frac{\pi}{2}$", "$\pi$"])
#
# ax.set_xlabel("Orientation")
# ax.set_ylabel("Activation")
# plt.tight_layout()
# plt.show()
# # %%
# fig, ax = plt.subplots()
# ax.axis("off")
# # plot one gaussian with high variance in ASD_COLOR and one with low variance in NT_COLOR
# x = get(np.linspace(-np.pi, np.pi, 1000))
# y1 = scipy.stats.vonmises.pdf(x, kappa=kappa_wide)
# y2 = scipy.stats.vonmises.pdf(x, kappa=kappa_sharp)
# ax.plot(x, y1, color=ASD_COLOR, linewidth=3)
# ax.plot(x, y2, color=NT_COLOR, linewidth=3)
# plt.savefig("tuning_curve_illustration.pdf")
#
# # %% plot ndr_oblique_choices and ndr_cardinal_choices histograms in two separate subplots, and add a text with the distribution width
# fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
# plot_choice_hist(axes[1], oblique_ndr_choices, oblique_ndr_percent_close, NT_COLOR, "Oblique NDR")
# axes[1].text(0.25, 0.75, f"Width: {get_choice_distribution_width(oblique_ndr_choices):.2f}", ha='center', va='center',
#              fontsize=12, fontweight='bold', transform=axes[1].transAxes)
# # add dotted gray line on the stimulus
# axes[1].vlines(oblique_stim, 0, 1, color='gray', linestyle='--', linewidths=3)
# # add dotted red line on the average choice
# # axes[1].vlines(get(oblique_ndr_choices.mean()), 0, 1, color='red', linestyle='--', linewidths=1)
# axes[0].set_ylabel("Density")
# plot_choice_hist(axes[0], cardinal_ndr_choices, cardinal_ndr_percent_close, NT_COLOR, "Cardinal NDR")
# axes[0].text(0.25, 0.75, f"Width: {get_choice_distribution_width(cardinal_ndr_choices):.2f}", ha='center', va='center',
#              fontsize=12, fontweight='bold', transform=axes[0].transAxes)
# axes[0].vlines(cardinal_stim, 0, 1, color='gray', linestyle='--', linewidths=3)
# # add dotted red line on the average choice
# # axes[0].vlines(get(cardinal_ndr_choices.mean()), 0, 1, color='red', linestyle='--', linewidths=1)
# for ax in axes:
#     ax.set_xlabel("Decoded Stimulus")
#     # set xticks with 1/4 pi intervals from 0 to 2pi
#     ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4],
#                   [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$",
#                    r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$"]
#                   )
# plt.tight_layout()
# plt.show()
# # %% plot the bias curve of the NDR model
# fig, ax = plt.subplots(figsize=(7, 5))
# ax.plot(get(stimuli), bias_ndr, color=NT_COLOR, label="NDR", linewidth=3)
# # set the xticks to 0-2pi in pi/4 increments
# ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
#               [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"],
#               fontsize=18
#               )
# # plot gray dotted vlines at cardinal orientations, and a solid black like at 0
# ax.vlines([0, np.pi / 2, np.pi], get(bias_ndr.min()).item(), get(bias_ndr.max()).item(),
#           color='gray', linestyle='--', lw=2)
# ax.hlines(0, 0, np.pi, color='black', linestyle='-', lw=1)
# ax.set_xlabel("Stimulus")
# ax.set_ylabel("Bias (rad)")
# plt.tight_layout()
# plt.show()
#
# # %%
# ndr_cr_bound = get(((1 + np.gradient(np.array(bias_ndr))) ** 2) / np.array(variance_ndr))
# idr_cr_bound = get(((1 + np.gradient(np.array(bias_idr))) ** 2) / np.array(variance_idr))
# # %%
# fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
# axes[0].plot(get(stimuli), bias_ndr, color=NT_COLOR, label="NDR", linewidth=3)
# axes[0].plot(get(stimuli), bias_idr, linestyle="--", color=ASD_COLOR, label="IDR", linewidth=3)
# axes[1].plot(get(stimuli), variance_ndr, color=NT_COLOR, label="NDR", linewidth=3)
# axes[1].plot(get(stimuli), variance_idr, color=ASD_COLOR, label="IDR", linestyle="--", linewidth=3)
#
# axes[2].plot(get(stimuli), ndr_cr_bound, color=NT_COLOR, label="NDR", linewidth=3)
# axes[2].plot(get(stimuli), idr_cr_bound, color=ASD_COLOR, label="IDR", linestyle="--", linewidth=3)
# axes[2].set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
#                    [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"], fontsize=16
#                    )
# for ax, ylabel, (stim1, stim2) in zip(axes, ["Bias (rad)", "Variance (rad)", "FI"],
#                                       [(bias_ndr, bias_idr), (variance_ndr, variance_idr),
#                                        (ndr_cr_bound, idr_cr_bound)]):
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.set_yticks(ax.get_yticks(), np.round(ax.get_yticks(), 2), fontsize=16)
#     ax.set_ylabel(ylabel, fontsize=18)
#     ax.vlines([0, np.pi / 2, np.pi], min(get(stim1.min()).item(), get(stim2.min()).item()),
#               max(get(stim1.max()).item(), get(stim2.max()).item()), color='gray', linestyle='--', lw=2)
#     ax.legend(fontsize=16, loc="upper left")
# axes[-1].set_xlabel("Stimulus", fontsize=18)
#
# plt.tight_layout()
# plt.show()
#
# # %% plot sigmoid
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def hill(x, k, n):
#     return x ** n / (x ** n + k ** n)
#
#
# plt.rcParams.update({
#     "figure.figsize": (12, 8),
#     "axes.labelsize": 16,
#     "axes.labelweight": "bold",
#     "axes.titlesize": 18,
#     "axes.titleweight": 'bold',
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
#     "axes.spines.top": False,
#     "axes.spines.right": False,
# })
#
# x = np.linspace(0, 1, 1000)
# ndr = hill(x, 0.5, 16)
# idr = hill(x, 0.5, 6)
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(x, ndr, color=NT_COLOR, linewidth=4)
# ax.set_xlabel("Input")
# ax.set_ylabel("Output")
# plt.tight_layout()
# plt.savefig("hill1.svg")
# plt.show()
#
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(x, ndr, color=NT_COLOR, linewidth=4)
# ax.plot(x, idr, color=ASD_COLOR, linewidth=4)
# ax.set_xlabel("Input")
# ax.set_ylabel("Output")
# plt.tight_layout()
# plt.savefig("hill2.svg")
# plt.show()
#
# # %%
# all_choices_ndr = []
# all_choices_idr = []
# for stim in stimuli:
#     all_choices_ndr.append(circ_distance(get_choices(model_ndr, stim, n_choices=10000), stim))
#     all_choices_idr.append(circ_distance(get_choices(model_idr, stim, n_choices=10000), stim))
# all_choices_ndr = np.array(all_choices_ndr)
# all_choices_idr = np.array(all_choices_idr)
# # %%
# from matplotlib.colors import LinearSegmentedColormap, Normalize
# import seaborn as sns
#
# fig, axes = plt.subplots(3, 1, figsize=(10, 15))
# colors = ["red", "white", "red"]
#
# # Create the colormap
#
# hist, xedges, yedges = map(get, np.histogram2d(np.repeat(stimuli, all_choices_ndr.shape[-1]), all_choices_ndr.ravel(),
#                                                bins=60, density=True))
# im = axes[0].pcolormesh((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2, hist.T, cmap='Reds')
# hist, xedges, yedges = map(get, np.histogram2d(np.repeat(stimuli, all_choices_idr.shape[-1]), all_choices_idr.ravel(),
#                                                bins=60, density=True))
# axes[1].pcolormesh((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2, hist.T, cmap='Reds')
#
# axes[0].set_title("NDR")
# axes[1].set_title("IDR")
# axes[0].set_xlabel("Stimulus")
# axes[1].set_xlabel("Stimulus")
# axes[0].set_ylabel("Distance (radians)")
#
# # set xticks to 0-pi in pi/4 increments
# axes[0].set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
#                    [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
# axes[1].set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
#                    [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
#
# # plot the mean on top
# axes[0].plot(get(stimuli), get(all_choices_ndr.mean(-1)), color=NT_COLOR, linewidth=5)
# axes[1].plot(get(stimuli), get(all_choices_idr.mean(-1)), color='blue', linewidth=5)
# # plot the mean all choices for each model on the same ax
# axes[2].plot(get(stimuli), get(all_choices_ndr.mean(-1)), color=NT_COLOR, linewidth=3, label="NDR")
# axes[2].plot(get(stimuli), get(all_choices_idr.mean(-1)), color='blue', linewidth=3, label="IDR")
# axes[2].set_xlabel("Stimulus")
# axes[2].set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
#                    [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
# plt.tight_layout()
# # add black line at 0, gray lines at cardinal orientations and dashed lines at oblique
# for ax, val in zip(axes, [1, 1, 0.05]):
#     ax.vlines([0, np.pi / 2, np.pi], -val, val,
#               color='gray', linestyle='--', lw=2)
#     ax.hlines(0, 0, np.pi, color='black', linestyle='-', lw=1)
#     ax.vlines([np.pi / 4, 3 * np.pi / 4], -val, val,
#               color='gray', linestyle='-.', lw=2)
#     ax.legend()
# plt.show()
#
# # %% plot the tuning function of each neuron in the IDR model on a polar plot
#
# fig, ax = plt.subplots(figsize=(16, 8))
# # plot the tuning function of each neuron in the IDR model
# for i in range(model_idr.N):
#     ax.plot((np.linspace(0, 2 * np.pi, 181).get() + model_idr.theta[i].item()) % (2 * np.pi),
#             get(model_idr.tuning_func(x=np.linspace(0, 2 * np.pi, 181), kappa=model_idr.tuning_widths[i])),
#             color=ASD_COLOR, alpha=0.1)
# # set xticks in radians
# ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4],
#               [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$",
#                r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$"]
#               )
# plt.show()
