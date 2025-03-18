import functools
# import cupy.random
import scipy
# import numpy as np
import cupy as np
import matplotlib.pyplot as plt
import os

from scipy.stats import circvar, circmean
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

ASD_COLOR, NT_COLOR = "#FF0000", "#00A08A"


def get(x):
    try:
        return x.get()
    except:
        return x


def vm_like(x, kappa):
    resp = (2 * np.exp(kappa * (np.cos(x) - 1))) - 1
    return resp


def vm(x, kappa):
    return scipy.stats.vonmises.pdf(x, kappa=kappa) - (
            scipy.stats.vonmises.pdf(0, kappa=kappa).max(0, keepdims=True) / 2)


def get_tuning_widths(N, kappa, precision=6, min_val=0.5):
    b = get_location_based_increase(N, precision, min_val)
    return b * kappa


def circ_distance(x, y):
    return np.atan2(np.sin(x - y), np.cos(x - y))


def animate_model_example(name, stim, y, theta, i=0):
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    line, = ax.plot(get(theta), get(y[0, :, i]), lw=2)
    ax.vlines(get(stim), get(y).min(), get(y).max(), color='r', linestyle='--')
    ax.set_xlim(0, 2 * np.pi)
    ax.set_ylim(y.min(), y.max())
    ax.set_title(f"t=0")
    ax.set_xlabel("Theta")
    ax.set_ylabel("Activation")

    def update(j):
        line.set_ydata(get(y[j, :, 0]))
        ax.set_title(f"t={j}")
        return line,

    rng: tqdm = tqdm(range(get(y).shape[0]))
    ani = FuncAnimation(fig, update, frames=rng, interval=50)
    # save the animation as an mp4.
    ani.save(f'{name}_activation_evolution.gif', writer='ffmpeg')
    rng.close()


def animate_tuning_widths(name, idr_tuning_widths, ndr_tuning_widths, idr_theta, ndr_theta, kappa_wide=2,
                          kappa_sharp=8):
    fig, ax = plt.subplots()
    idr_line, = ax.plot(idr_theta[0], idr_tuning_widths[0] / kappa_wide, color=ASD_COLOR, label="IDR",
                        linewidth=3)
    ndr_line, = ax.plot(ndr_theta[0], ndr_tuning_widths[0] / kappa_sharp, color=NT_COLOR, label="NDR",
                        linewidth=3)
    # set xlabels to be in radians, 0 till 2pi
    ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                  ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
    ax.set_ylabel("Relative change in width", fontsize=14, fontweight='bold')
    ax.hlines(0, 0, 2 * np.pi, linestyles="--", colors='gray')
    ax.legend()
    ax.set_ylim(0, max((ndr_tuning_widths / kappa_sharp).max(), (idr_tuning_widths / kappa_wide).max()))
    ax.set_title(f"t=0")

    def update(j):
        idr_line.set_ydata(idr_tuning_widths[j] / kappa_wide)
        idr_line.set_xdata(idr_theta[j])
        ndr_line.set_ydata(ndr_tuning_widths[j] / kappa_sharp)
        ndr_line.set_xdata(ndr_theta[j])
        ax.set_title(f"t={j}")
        return idr_line, ndr_line

    rng: tqdm = tqdm(range(get(idr_tuning_widths).shape[0]))
    ani = FuncAnimation(fig, update, frames=rng, interval=50)
    # save the animation as an mp4.
    ani.save(f'{name}_activation_evolution.gif', writer='ffmpeg')
    rng.close()


def get_location_based_increase(N, precision, min_val=0.5):
    a = np.array(
        scipy.stats.vonmises.pdf(get(np.arange(-np.pi / 4, np.pi / 4, np.pi / (2 * N // 4))), kappa=precision))
    a -= a.min()
    a /= a.max()
    a *= min_val
    a += 1 - min_val
    b = np.roll(np.concatenate([a, a, a, a]), (N // 4) / 2)
    return b


class Model:
    def __init__(self, j0, j1, h0, h1, N, gains, tuning_widths, tuning_func, lr, count_thresh=0, width_scaling=1, T=1,
                 dt=1e-2, noise=0., stim_noise=0., n_sims=1000):
        self.j0 = j0
        self.j1 = j1
        self.h0 = h0
        self.h1 = h1
        self.N = N
        self.T = T
        self.dt = dt
        self.time = np.arange(0, T + dt, dt)
        self.n_sims = n_sims
        self.gains = np.array(gains)
        self.theta = np.linspace(0, 2 * np.pi, N)
        self.tuning_widths = np.array(tuning_widths).astype(float)
        self.tuning_func = tuning_func
        self.J = (1 / self.N) * (self.j0 + self.j1 * np.cos(self.theta[:, None] - self.theta[None, :]))
        self.r = np.zeros((self.time.size, N, n_sims))
        self.noise = noise
        self._dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(self.time.size - 1, N, self.n_sims)) * self.noise
        self.lr = lr
        self.width_scaling = width_scaling
        self.count_thresh = count_thresh
        self.base_factor = self.get_near_factor()
        self.stim_noise = stim_noise
        self.stim_history = []

    def deterministic_func(self, y, stim):
        return -y + ((self.h0 + self.h1 * self.tuning_func((self.theta - stim),
                                                           self.tuning_widths)) * self.gains)[:,
                    None] + self.J @ y

    def euler_maruyama(self, y, stim, i):
        return y + self.deterministic_func(y, stim) * self.dt + self._dW[i - 1]

    def run(self, stim):
        self.stim_history.append(stim)
        # add noise to the stimulus, different noise for each neuron and time point

        noisy_stim = stim + np.random.normal(0, self.stim_noise, (self.time.size, self.N))
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
        self.tuning_widths *= ((self.width_scaling * (
                (self.get_near_factor().astype(float) / self.base_factor.astype(float)) - 1)) + 1)
        if recalculate_connectivity:
            self.J = (1 / self.N) * (self.j0 + self.j1 * np.cos(self.theta[:, None] - self.theta[None, :]))
            self.J[np.arange(self.N), np.arange(self.N)] = 0
        return old_theta, old_widths

    def get_near_factor(self):
        resps = self.tuning_func(self.theta[:, None] - self.theta[None, :], self.tuning_widths[:, None])
        # for each neuron in the response, count the number of self.theta values that have a response greater than self.count_thresh
        near_count = (resps > self.count_thresh * resps.max()).sum(1)
        return near_count


def circ_distance(x, y):
    try:
        return np.arctan2(np.sin(x - y), np.cos(x - y))
    except Exception as e:
        return np.atan2(np.sin(x - y), np.cos(x - y))


def get_natural_stats_distribution(n_points, peaks=None, kappa=6):
    if peaks is None:
        peaks = [-np.pi, -np.pi / 2, 0, np.pi / 2]
    out = np.concatenate([np.random.vonmises(peak, kappa, n_points) for peak in peaks])
    np.random.shuffle(out)
    out += 3 * np.pi
    out %= 2 * np.pi
    return out - np.pi


def get_bias_variance(model, sigma=0.75, seed=97):
    np.random.seed(seed)
    stimuli = np.linspace(0, np.pi, 181)
    bias = np.zeros_like(stimuli)
    bias_ci = np.zeros(stimuli.shape)
    variance = np.zeros_like(stimuli)
    for i, stim in enumerate(tqdm(stimuli)):
        fr = np.squeeze(model.run(stim))
        choices = np.random.choice(model.theta, replace=True,
                                   p=(fr - fr.min()) / (fr - fr.min()).sum(),
                                   size=10000)
        b = circ_distance(choices, stim)
        bias[i] = b.mean()
        bias_ci[i] = b.std()
        variance[i] = circvar(get(choices))
    if sigma > 0:
        smooth_bias = scipy.ndimage.gaussian_filter1d(get(bias), sigma)
        smooth_bias_ci = scipy.ndimage.gaussian_filter1d(get(bias_ci), sigma)
        smooth_variance = scipy.ndimage.gaussian_filter1d(get(variance), sigma)
    else:
        smooth_bias, smooth_variance, smooth_bias_ci = get(bias), get(variance), get(bias_ci)
    return smooth_bias, smooth_variance, stimuli, smooth_bias_ci


def get_choices(model, stim, n_choices=10000, seed=97):
    np.random.seed(seed)
    resps = model.run(stim)
    prob = np.squeeze((resps - resps.min()) / (resps - resps.min()).sum())
    choices = np.random.choice(np.squeeze(model.theta), replace=True,
                               p=prob, size=n_choices)
    return choices


def shift_function(x, y, return_dist=False):
    """
    Calculates the shift function for First vector - Second vector
    :param x: First vector
    :param y: Second vector
    :param return_dist: bool, if True returns the decile difference bootstrap distribution
    :return: mean decile difference, low ci, high ci
    """
    num_boot = 20000
    boot_x = np.random.choice(x, size=(num_boot, x.size))
    boot_y = np.random.choice(y, size=(num_boot, y.size))
    deciles = np.arange(0.1, 1, 0.1)
    x_deciles = np.quantile(boot_x, deciles, axis=1)
    y_deciles = np.quantile(boot_y, deciles, axis=1)
    decile_diff = x_deciles - y_deciles
    decile_diff_mean = decile_diff.mean(axis=1)
    diff_low_ci, diff_high_ci = np.quantile(decile_diff, [0.025, 0.975], axis=1)
    if return_dist:
        return decile_diff_mean, diff_low_ci, diff_high_ci, decile_diff
    return decile_diff_mean, diff_low_ci, diff_high_ci


def add_error_bars(ax, x, y, low_bars, high_bars, width, color='k'):
    """
    Adds error bars to an axis
    :param ax: the axis
    :param x: x values of the plotted values
    :param y: y values of the plotted values
    :param low_bars: y values for the low end of the error bars
    :param high_bars: y values for the high end of the error bars
    :param width: the width of the horizontal lines at the ends of the error bars
    :param color: the color of the error bars
    """
    x, y, low_bars, high_bars = list(map(np.array, [x, y, low_bars, high_bars]))
    ax.vlines(get(x), get(low_bars), get(high_bars), colors=color)
    ax.hlines(get(low_bars), get(x - width), get(x + width), colors=color)
    ax.hlines(get(high_bars), get(x - width), get(x + width), colors=color)
    ax.set_ylim(min(y.min(), -np.abs(y).max() / 10),
                max(y.max(), np.abs(y).max() / 10))
    ax.set_xticks(get(x))


def plot_shift_func(x, y, ax=None, marker_size=6):
    """
    Plot a shift function of x-y
    :param x: First vector
    :param y: Second vector
    :return: the figure on which the shift function is plotted
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    deciles = np.arange(0.1, 1, 0.1)
    decile_diff_mean, diff_low_ci, diff_high_ci, decile_diff = shift_function(x, y, True)
    add_error_bars(ax, get(deciles), get(decile_diff), get(diff_low_ci), get(diff_high_ci), 0.02)
    ax.scatter(get(deciles), get(decile_diff_mean), c='k', s=marker_size ** 2)
    ax.hlines(0, ax.get_xlim()[0], ax.get_xlim()[1], linestyles=":", colors="k")
    # remove spines and ticks


def get_hdi_indices(dist, confidence=0.05):
    alpha = confidence / 2
    dist -= dist.min()
    dist /= dist.sum()
    cumsum = np.cumsum(dist)
    low_idx = np.argmin(np.abs(cumsum - alpha))
    high_idx = np.argmin(np.abs(cumsum - (1 - alpha)))
    return [low_idx.item(), high_idx.item()]


# %%
N = 420
kappa_sharp = 8
kappa_wide = 2
n_stim = 500
np.random.seed(42)
normalize_fr = True
params = dict(j0=0.2, j1=0.9, h0=0.1, h1=0.5, N=N, lr=1e-2, T=1, dt=1e-2, noise=0., stim_noise=0.,
              count_thresh=0.90, width_scaling=1, n_sims=1)
widths_ndr = [kappa_sharp] * N  # get_tuning_widths(N, kappa, precision=18, min_val=0.3)
widths_idr = [kappa_wide] * N  # get_tuning_widths(N, 3, precision=18, min_val=0.66)
# widths_idr = [kappa // 2] * N
gains = [1] * N  # get_location_based_increase(N, precision=6, min_val=0.75)

model_idr = Model(gains=gains, tuning_widths=widths_idr, tuning_func=vm_like, **params)
model_ndr = Model(gains=gains, tuning_widths=widths_ndr, tuning_func=vm_like, **params)

model_idr_no_update = Model(gains=gains, tuning_widths=widths_idr, tuning_func=vm_like, **params)
model_ndr_no_update = Model(gains=gains, tuning_widths=widths_ndr, tuning_func=vm_like, **params)

all_idr_resps = []
all_ndr_resps = []
all_idr_theta = []
all_ndr_theta = []
all_idr_tuning = []
all_ndr_tuning = []
# stim_list = np.arange(0, 2 * np.pi, np.pi / 180)
stim_list = get_natural_stats_distribution(n_stim) + np.pi
for stim in tqdm(stim_list):
    # model_idr.run(stim)
    # model_ndr.run(stim)
    all_idr_resps.append(model_idr.run(stim))
    all_ndr_resps.append(model_ndr.run(stim))
    all_idr_theta.append(model_idr.theta.copy())
    all_ndr_theta.append(model_ndr.theta.copy())
    all_idr_tuning.append(model_idr.tuning_widths.copy())
    all_ndr_tuning.append(model_ndr.tuning_widths.copy())
    model_idr.update(recalculate_connectivity=False)
    model_ndr.update(recalculate_connectivity=False)

all_idr_resps = np.array(all_idr_resps)  # (stim, N, n_sims)
all_ndr_resps = np.array(all_ndr_resps)
all_idr_theta = np.array(all_idr_theta)
all_ndr_theta = np.array(all_ndr_theta)
all_idr_tuning = np.array(all_idr_tuning)
all_ndr_tuning = np.array(all_ndr_tuning)
# %% setup before plot
oblique_stim = 5 * np.pi / 4
cardinal_stim = np.pi

idr_theta = model_idr.theta.get()
ndr_theta = model_ndr.theta.get()

oblique_idr_choices = get_choices(model_idr, oblique_stim, n_choices=10000)
cardinal_idr_choices = get_choices(model_idr, cardinal_stim, n_choices=10000)
oblique_idr_no_update_choices = get_choices(model_idr_no_update, oblique_stim, n_choices=10000)
cardinal_idr_no_update_choices = get_choices(model_idr_no_update, cardinal_stim, n_choices=10000)

oblique_ndr_choices = get_choices(model_ndr, oblique_stim, n_choices=10000)
cardinal_ndr_choices = get_choices(model_ndr, cardinal_stim, n_choices=10000)
oblique_ndr_no_update_choices = get_choices(model_ndr_no_update, oblique_stim, n_choices=10000)
cardinal_ndr_no_update_choices = get_choices(model_ndr_no_update, cardinal_stim, n_choices=10000)


def get_choice_distribution_width(choices, bins=100):
    # Calculate the histogram of choices
    hist, bin_edges = np.histogram(choices, bins=bins)
    # Find the bin with the maximum probability
    max_prob_idx = np.argmax(hist)
    max_prob = hist[max_prob_idx]
    # Calculate the threshold probability
    threshold_prob = max_prob / np.e
    # Find the bin where the probability falls below the threshold
    below_threshold_idx = np.where(hist < threshold_prob)[0]
    # Find the closest bin to the max_prob_idx that is below the threshold
    closest_below_threshold_idx = below_threshold_idx[np.argmin(np.abs(below_threshold_idx - max_prob_idx))]
    # Calculate the width in radians
    width = np.abs(bin_edges[closest_below_threshold_idx] - bin_edges[max_prob_idx])
    return width


print("oblique_idr/cardinal_idr",
      get_choice_distribution_width(oblique_idr_choices) / get_choice_distribution_width(cardinal_idr_choices))

get_choice_distribution_width(oblique_ndr_choices) / get_choice_distribution_width(cardinal_ndr_choices)

get_choice_distribution_width(oblique_idr_no_update_choices)
get_choice_distribution_width(cardinal_idr_no_update_choices)

get_choice_distribution_width(oblique_ndr_no_update_choices)
get_choice_distribution_width(cardinal_ndr_no_update_choices)

threshold = 0.05 * np.pi
oblique_idr_percent_close = get(np.abs(circ_distance(oblique_idr_choices, oblique_stim)) < threshold).mean()
oblique_idr_no_update_percent_close = get(
    np.abs(circ_distance(oblique_idr_no_update_choices, oblique_stim)) < threshold).mean()
cardinal_idr_percent_close = get(np.abs(circ_distance(cardinal_idr_choices, cardinal_stim)) < threshold).mean()
cardinal_idr_no_update_percent_close = get(
    np.abs(circ_distance(cardinal_idr_no_update_choices, cardinal_stim)) < threshold).mean()

oblique_ndr_percent_close = get(np.abs(circ_distance(oblique_ndr_choices, oblique_stim)) < threshold).mean()
oblique_ndr_no_update_percent_close = get(
    np.abs(circ_distance(oblique_ndr_no_update_choices, oblique_stim)) < threshold).mean()
cardinal_ndr_percent_close = get(np.abs(circ_distance(cardinal_ndr_choices, cardinal_stim)) < threshold).mean()
cardinal_ndr_no_update_percent_close = get(
    np.abs(circ_distance(cardinal_ndr_no_update_choices, cardinal_stim)) < threshold).mean()

print(f"Oblique IDR: {100 * (oblique_idr_percent_close - oblique_idr_no_update_percent_close)}")
print(f"Cardinal IDR: {100 * (cardinal_idr_percent_close - cardinal_idr_no_update_percent_close)}")
print(f"Cardinal-Oblique IDR: {100 * (cardinal_idr_percent_close - oblique_idr_percent_close)}")

print(f"Oblique NDR: {100 * (oblique_ndr_percent_close - oblique_ndr_no_update_percent_close)}")
print(f"Cardinal NDR: {100 * (cardinal_ndr_percent_close - cardinal_ndr_no_update_percent_close)}")
print(f"Cardinal-Oblique NDR: {100 * (cardinal_ndr_percent_close - oblique_ndr_percent_close)}")

bias_idr, variance_idr, stimuli, bias_ci_idr = get_bias_variance(model_idr, sigma=1)
bias_ndr, variance_ndr, _, bias_ci_ndr = get_bias_variance(model_ndr, sigma=1)
# %% Create a figure with the following subplots:
# A) 3 polar plots for the distribution of stimuli, NDR theta and IDR theta
# B) The precision of the IDR and NDR models as a function of the stimulus, divided by the initial precision
# C) 4 polar plots for the chosen orientations based on the firing rates of the IDR and NDR models for the oblique and cardinal stimuli
# D) 2 plots, bias and variance of the IDR and NDR models as a function of the stimulus
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update(
    {
        "legend.fontsize": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "axes.titleweight": 'bold',
        "xtick.labelsize": 16,
        "ytick.labelsize": 14
    }
)
fig = plt.figure(figsize=(18, 10))
n_cols = 16
gs = GridSpec(3, n_cols, figure=fig, height_ratios=[1, 0.7, 0.7])

row1_width = n_cols // 5
ax1 = fig.add_subplot(gs[0, 0:row1_width], projection='polar')
ax2 = fig.add_subplot(gs[0, row1_width:2 * row1_width], projection='polar')
ax3 = fig.add_subplot(gs[0, 2 * row1_width:3 * row1_width], projection='polar')
ax4 = fig.add_subplot(gs[0, 3 * row1_width + 1:])

row2_width = n_cols // 4
ax5 = fig.add_subplot(gs[1, :row2_width])
ax6 = fig.add_subplot(gs[1, row2_width:2 * row2_width], sharey=ax5, sharex=ax5)
ax7 = fig.add_subplot(gs[1, 2 * row2_width:3 * row2_width], sharey=ax5, sharex=ax5)
ax8 = fig.add_subplot(gs[1, 3 * row2_width:4 * row2_width], sharey=ax5, sharex=ax5)

# row3_width = n_cols // 2
# ax_shift_idr = fig.add_subplot(gs[2, :row3_width])
# ax_shift_ndr = fig.add_subplot(gs[2, row3_width:], sharey=ax_shift_idr)

row4_width = n_cols // 2
ax9 = fig.add_subplot(gs[2, :row4_width])
ax10 = fig.add_subplot(gs[2, row4_width + 1:])

# plot the distribution of the stimuli


axes = [ax1, ax2, ax3, ax5, ax6, ax7, ax8]
axes[0].hist(stim_list.get(), bins=120, density=True, alpha=0.5, color="black")
axes[1].hist(idr_theta, bins=120, density=True, color=ASD_COLOR, alpha=0.5)
axes[2].hist(ndr_theta, bins=120, density=True, color=NT_COLOR, alpha=0.5)
axes[0].set_title("Stimulus")
axes[1].set_title("IDR")
axes[2].set_title("NDR")
for ax in axes:
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4],
                  [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$",
                   r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$"]
                  )

for ax in [ax1, ax2, ax3]:
    ax.set_yticks([])
    ax.tick_params(axis='x', which='major', pad=0.3)

ax4.plot(model_idr.theta.get(), model_idr.tuning_widths.get() / kappa_wide, color=ASD_COLOR, label="IDR", linewidth=3)
ax4.plot(model_ndr.theta.get(), model_ndr.tuning_widths.get() / kappa_sharp, color=NT_COLOR, label="NDR", linewidth=3)
# set xlabels to be in radians, 0 till 2pi
ax4.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
               ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
ax4.text(0.5, -0.2, "Preferred Orientation", ha='center', va='center', fontsize=14, transform=ax4.transAxes)
ax4.set_ylabel("Relative change in width", fontsize=14, fontweight='bold')
ax4.hlines(1, 0, 2 * np.pi, linestyles="--", colors='gray')
ax4.legend()


def plot_choice_hist(ax: plt.Axes, choices, percent_close, percent_close_no_update, color, title, ax_inset_share=None):
    # ax_inset = inset_axes(ax, width="40%", height="40%", loc='center left', borderpad=1,
    #                       axes_kwargs={"sharex": ax_inset_share})
    # # plot the idr_percent_close for altered and no update models as side by side hbars on ax_inset
    # bars = ax_inset.barh([0, 1], [percent_close, percent_close_no_update],
    #                      color=[color, "black"], alpha=0.5)
    # # add text on the center of the bars "Simple model" and "altered model"
    # for bar, text in zip(bars, [("Altered Model %d" % (percent_close * 100)) + "%",
    #                             ("Simple Model %d" % (percent_close_no_update * 100)) + "%"]):
    #     ax_inset.text(0.01 * bar.get_width(), bar.get_y() + bar.get_height() / 2, text, ha='left', va='center',
    #                   fontsize=11)
    # ax_inset.text(0.5, 1.1, "% Correct", ha='center', va='center', fontsize=12, fontweight='bold',
    #               transform=ax_inset.transAxes)
    # ax_inset.axis('off')

    ax.hist(get(choices), bins=40, density=True, alpha=0.5, color=color)
    ax.text(0.5, 0.975, title, ha='center', va='center', fontsize=16, fontweight='bold',
            transform=ax.transAxes)
    # ax.set_xlim([-np.pi / 4, 2 * np.pi])
    # return ax_inset


# add the width of the choices to the plot in text
plot_choice_hist(ax5, oblique_idr_choices, oblique_idr_percent_close, oblique_idr_no_update_percent_close,
                 ASD_COLOR, "Oblique IDR")
ax5.text(0.5, 0.75, f"Width: {get_choice_distribution_width(oblique_idr_choices):.2f}", ha='center', va='center',
         fontsize=12,
         fontweight='bold', transform=ax5.transAxes)
plot_choice_hist(ax6, cardinal_idr_choices, cardinal_idr_percent_close, cardinal_idr_no_update_percent_close,
                 ASD_COLOR, "Cardinal IDR")
ax6.text(0.5, 0.75, f"Width: {get_choice_distribution_width(cardinal_idr_choices):.2f}", ha='center', va='center',
         fontsize=12, fontweight='bold', transform=ax6.transAxes)
plot_choice_hist(ax7, oblique_ndr_choices, oblique_ndr_percent_close, oblique_ndr_no_update_percent_close,
                 NT_COLOR, "Oblique NDR")
ax7.text(0.5, 0.75, f"Width: {get_choice_distribution_width(oblique_ndr_choices):.2f}", ha='center', va='center',
         fontsize=12, fontweight='bold', transform=ax7.transAxes)
plot_choice_hist(ax8, cardinal_ndr_choices, cardinal_ndr_percent_close, cardinal_ndr_no_update_percent_close,
                 NT_COLOR, "Cardinal NDR")
ax8.text(0.5, 0.75, f"Width: {get_choice_distribution_width(cardinal_ndr_choices):.2f}", ha='center', va='center',
         fontsize=12, fontweight='bold', transform=ax8.transAxes)

# ax5_inset = plot_choice_hist(ax5, oblique_idr_choices, oblique_idr_percent_close, oblique_idr_no_update_percent_close,
#                              ASD_COLOR, "Oblique IDR")
# ax6_inset = plot_choice_hist(ax6, cardinal_idr_choices, cardinal_idr_percent_close,
#                              cardinal_idr_no_update_percent_close,
#                              ASD_COLOR, "Cardinal IDR", ax5_inset)
# ax7_inset = plot_choice_hist(ax7, oblique_ndr_choices, oblique_ndr_percent_close, oblique_ndr_no_update_percent_close,
#                              NT_COLOR, "Oblique NDR", ax5_inset)
# ax8_inset = plot_choice_hist(ax8, cardinal_ndr_choices, cardinal_ndr_percent_close,
#                              cardinal_ndr_no_update_percent_close,
#                              NT_COLOR, "Cardinal NDR", ax5_inset)

# plot_shift_func(oblique_idr_choices - oblique_stim, cardinal_idr_choices - cardinal_stim, ax_shift_idr)
# ax_shift_idr.spines['top'].set_visible(False)
# ax_shift_idr.spines['right'].set_visible(False)
# ax_shift_idr.text(0.5, 0.975, "IDR Oblique-Cardinal Shift Function", ha='center', va='center', fontsize=16,
#                   fontweight='bold', transform=ax_shift_idr.transAxes)
# ax_shift_idr.set_xlabel("Decile")
# ax_shift_idr.set_ylabel("Shift (radians)")

# plot_shift_func(oblique_ndr_choices - oblique_stim, cardinal_ndr_choices - cardinal_stim, ax_shift_ndr)
# ax_shift_ndr.spines['top'].set_visible(False)
# ax_shift_ndr.spines['right'].set_visible(False)
# ax_shift_ndr.text(0.5, 0.975, "NDR Oblique-Cardinal Shift Function", ha='center', va='center', fontsize=16,
#                   fontweight='bold', transform=ax_shift_ndr.transAxes)
# ax_shift_ndr.set_xlabel("Decile")
# ax_shift_idr.set_ylim(ax_shift_idr.get_ylim()[0], ax_shift_idr.get_ylim()[1] * 1.2)

ax6.yaxis.set_tick_params(labelleft=False)
ax7.yaxis.set_tick_params(labelleft=False)
ax8.yaxis.set_tick_params(labelleft=False)
# ax_shift_ndr.yaxis.set_tick_params(labelleft=False)
# add the oblique\cardinal stimulus in a gray dotted line
ax5.vlines(oblique_stim, 0, 1, color='gray', linestyle='--')
ax6.vlines(cardinal_stim, 0, 1, color='gray', linestyle='--')
ax7.vlines(oblique_stim, 0, 1, color='gray', linestyle='--')
ax8.vlines(cardinal_stim, 0, 1, color='gray', linestyle='--')

ax9.plot(get(stimuli), bias_idr, color=ASD_COLOR, label="IDR", linewidth=3)
# plot shaded ci area
# ax9.fill_between(get(stimuli), bias_idr+bias_ci_idr, bias_idr-bias_ci_idr, color=ASD_COLOR, alpha=0.3)
ax9.plot(get(stimuli), bias_ndr, color=NT_COLOR, label="NDR", linewidth=3)
# ax9.fill_between(get(stimuli), bias_ndr+bias_ci_ndr, bias_ndr-bias_ci_ndr, color=NT_COLOR, alpha=0.3)
# set the xticks to 0-2pi in pi/4 increments
ax9.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
               [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"]
               )
# plot gray dotted vlines at cardinal orientations, and a solid black like at 0
ax9.vlines([0, np.pi / 2, np.pi], get(bias_idr.min()).item(), get(bias_ndr.max()).item(),
           color='gray', linestyle='--', lw=2)
ax9.hlines(0, 0, np.pi, color='black', linestyle='-', lw=1)

ax9.set_xlabel("Stimulus")
ax9.set_ylabel("Bias")
# set yticklabels to a larger font size
ax9.set_yticklabels(np.round(ax9.get_yticks(), 2))
ax9.legend()

ax10.plot(get(stimuli), get(variance_idr), color=ASD_COLOR, label="IDR", linewidth=3)
ax10.plot(get(stimuli), get(variance_ndr), color=NT_COLOR, label="NDR", linewidth=3)
# set the xticks to 0-2pi in pi/4 increments
ax10.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"])
# plot gray dotted vlines at cardinal orientations, and a solid black like at 0
ax10.vlines([0, np.pi / 2, np.pi], get(variance_ndr.min()).item(),
            get(variance_idr.max()).item(), color='gray', linestyle='--', lw=2)
# ax.hlines(0, 0, 2*np.pi, color='black', linestyle='-', lw=1)

ax10.set_xlabel("Stimulus")
ax10.set_ylabel("Variance")
# set yticklabels to a larger font size
ax10.set_yticklabels(np.round(ax10.get_yticks(), 2))
ax10.legend()

# remove the spines of ax4
for ax in [ax4, ax5, ax6, ax7, ax8, ax9, ax10]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
fig.subplots_adjust(hspace=0.4, wspace=1, top=0.925, bottom=0.05, left=0.075, right=0.95)
fig.text(0.025, 0.965, "A", fontsize=24, fontweight="bold")
fig.text(0.575, 0.965, "B", fontsize=24, fontweight="bold")
fig.text(0.025, 0.55, "C", fontsize=24, fontweight="bold")
# fig.text(0.025, 0.45, "D", fontsize=24, fontweight="bold")
fig.text(0.025, 0.27, "D", fontsize=24, fontweight="bold")
fig.text(0.51, 0.27, "E", fontsize=24, fontweight="bold")
plt.savefig("bias_ring_model.pdf")
plt.show()

# %%
# animate_tuning_widths("model", get(all_idr_tuning), get(all_ndr_tuning), get(all_idr_theta), get(all_ndr_theta),
#                       kappa_wide, kappa_sharp)
#%% plot the ndr model relative change in width
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(model_ndr.theta.get(), model_ndr.tuning_widths.get() / kappa_sharp, color=NT_COLOR, label="NDR", linewidth=3)
# set xlabels to be in radians, 0 till 2pi
ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
# plot gray dotted vlines at cardinal orientations, and a solid black like at 0
ax.vlines(np.pi / 2, get(model_ndr.tuning_widths.get().min()).item()/ kappa_sharp,
            get(model_ndr.tuning_widths.get().max()).item()/ kappa_sharp,
            color='gray', linestyle='--', lw=2)
ax.set_xlim([0, np.pi])
ax.hlines(1, 0, 2 * np.pi, linestyles="--", colors='black')
ax.set_xlabel("Preferred Orientation")
ax.set_ylabel("Relative change in width")
plt.tight_layout()
plt.show()

#%% plot the tuning curve of the NDR model at pi/2
fig, ax = plt.subplots(figsize=(6,4))
# ax.axis("off")
# plot one gaussian with high variance in ASD_COLOR and one with low variance in NT_COLOR
x = np.linspace(-np.pi, np.pi, 1000)
max_tuning = model_ndr.tuning_func(x=x,kappa=model_ndr.tuning_widths.max().item())
min_tuning = model_ndr.tuning_func(x=x,kappa=model_ndr.tuning_widths.min().item())

ax.plot(get(x), get(max_tuning), color=NT_COLOR, linewidth=3)
ax.plot(get(x), get(min_tuning), color=NT_COLOR, linewidth=3, linestyle="--")
ax.set_xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                ["$-\pi$", r"$-\frac{\pi}{2}$", "$0$", r"$\frac{\pi}{2}$", "$\pi$"])

ax.set_xlabel("Orientation")
ax.set_ylabel("Activation")
plt.tight_layout()
plt.show()
# %%
fig, ax = plt.subplots()
ax.axis("off")
# plot one gaussian with high variance in ASD_COLOR and one with low variance in NT_COLOR
x = get(np.linspace(-np.pi, np.pi, 1000))
y1 = scipy.stats.vonmises.pdf(x, kappa=kappa_wide)
y2 = scipy.stats.vonmises.pdf(x, kappa=kappa_sharp)
ax.plot(x, y1, color=ASD_COLOR, linewidth=3)
ax.plot(x, y2, color=NT_COLOR, linewidth=3)
plt.savefig("tuning_curve_illustration.pdf")

#%% plot ndr_oblique_choices and ndr_cardinal_choices histograms in two separate subplots, and add a text with the distribution width
fig, axes = plt.subplots(1, 2, figsize=(8, 5),sharey=True)
plot_choice_hist(axes[1], oblique_ndr_choices, oblique_ndr_percent_close, oblique_ndr_no_update_percent_close,
                 NT_COLOR, "Oblique NDR")
axes[1].text(0.25, 0.75, f"Width: {get_choice_distribution_width(oblique_ndr_choices):.2f}", ha='center', va='center',
                fontsize=12, fontweight='bold', transform=axes[1].transAxes)
# add dotted gray line on the stimulus
axes[1].vlines(oblique_stim, 0, 1, color='gray', linestyle='--', linewidths=3)
# add dotted red line on the average choice
# axes[1].vlines(get(oblique_ndr_choices.mean()), 0, 1, color='red', linestyle='--', linewidths=1)
axes[0].set_ylabel("Density")
plot_choice_hist(axes[0], cardinal_ndr_choices, cardinal_ndr_percent_close, cardinal_ndr_no_update_percent_close,
                 NT_COLOR, "Cardinal NDR")
axes[0].text(0.25, 0.75, f"Width: {get_choice_distribution_width(cardinal_ndr_choices):.2f}", ha='center', va='center',
                fontsize=12, fontweight='bold', transform=axes[0].transAxes)
axes[0].vlines(cardinal_stim, 0, 1, color='gray', linestyle='--', linewidths=3)
# add dotted red line on the average choice
# axes[0].vlines(get(cardinal_ndr_choices.mean()), 0, 1, color='red', linestyle='--', linewidths=1)
for ax in axes:
    ax.set_xlabel("Decoded Stimulus")
    # set xticks with 1/4 pi intervals from 0 to 2pi
    ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4],
                    [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$",
                     r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$"]
                    )
plt.tight_layout()
plt.show()
#%% plot the bias curve of the NDR model
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(get(stimuli), bias_ndr, color=NT_COLOR, label="NDR", linewidth=3)
# set the xticks to 0-2pi in pi/4 increments
ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"],
              fontsize=18
                )
# plot gray dotted vlines at cardinal orientations, and a solid black like at 0
ax.vlines([0, np.pi / 2, np.pi], get(bias_ndr.min()).item(), get(bias_ndr.max()).item(),
            color='gray', linestyle='--', lw=2)
ax.hlines(0, 0, np.pi, color='black', linestyle='-', lw=1)
ax.set_xlabel("Stimulus")
ax.set_ylabel("Bias (rad)")
plt.tight_layout()
plt.show()

# %%
ndr_cr_bound = get(((1 + np.gradient(np.array(bias_ndr))) ** 2) / np.array(variance_ndr))
idr_cr_bound = get(((1 + np.gradient(np.array(bias_idr))) ** 2) / np.array(variance_idr))
# %%
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
axes[0].plot(get(stimuli), bias_ndr, color=NT_COLOR, label="NDR", linewidth=3)
axes[0].plot(get(stimuli), bias_idr, linestyle="--", color=ASD_COLOR, label="IDR", linewidth=3)
axes[1].plot(get(stimuli), variance_ndr, color=NT_COLOR, label="NDR", linewidth=3)
axes[1].plot(get(stimuli), variance_idr, color=ASD_COLOR, label="IDR", linestyle="--", linewidth=3)

axes[2].plot(get(stimuli), ndr_cr_bound, color=NT_COLOR, label="NDR", linewidth=3)
axes[2].plot(get(stimuli), idr_cr_bound, color=ASD_COLOR, label="IDR", linestyle="--", linewidth=3)
axes[2].set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi],
                   [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$"], fontsize=16
                   )
for ax, ylabel, (stim1, stim2) in zip(axes, ["Bias (rad)", "Variance (rad)", "FI"],
                                      [(bias_ndr, bias_idr), (variance_ndr, variance_idr),
                                       (ndr_cr_bound, idr_cr_bound)]):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_yticks(ax.get_yticks(), np.round(ax.get_yticks(), 2), fontsize=16)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.vlines([0, np.pi / 2, np.pi], min(get(stim1.min()).item(), get(stim2.min()).item()),
              max(get(stim1.max()).item(), get(stim2.max()).item()), color='gray', linestyle='--', lw=2)
    ax.legend(fontsize=16, loc="upper left")
axes[-1].set_xlabel("Stimulus", fontsize=18)

plt.tight_layout()
plt.show()


# %% plot sigmoid
import numpy as np
import matplotlib.pyplot as plt
def hill(x, k, n):
    return x ** n / (x ** n + k ** n)

plt.rcParams.update({
    "figure.figsize": (12, 8),
    "axes.labelsize": 16,
    "axes.labelweight": "bold",
    "axes.titlesize": 18,
    "axes.titleweight": 'bold',
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

x = np.linspace(0, 1, 1000)
ndr = hill(x, 0.5, 16)
idr = hill(x, 0.5, 6)
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, ndr, color=NT_COLOR, linewidth=4)
ax.set_xlabel("Input")
ax.set_ylabel("Output")
plt.tight_layout()
plt.savefig("hill1.svg")
plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(x, ndr, color=NT_COLOR, linewidth=4)
ax.plot(x, idr, color=ASD_COLOR, linewidth=4)
ax.set_xlabel("Input")
ax.set_ylabel("Output")
plt.tight_layout()
plt.savefig("hill2.svg")
plt.show()
