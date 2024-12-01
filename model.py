import functools

# import cupy.random
import scipy
# import numpy as np
import cupy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import circvar
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


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
                 dt=1e-2, noise=0.,
                 n_sims=1000):
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
        self.stim_history = []

    def deterministic_func(self, y, stim):
        return -y + ((self.h0 + self.h1 * self.tuning_func((self.theta - stim),
                                                           self.tuning_widths)) * self.gains)[:,
                    None] + self.J @ y

    def euler_maruyama(self, y, stim, i):
        return y + self.deterministic_func(y, stim) * self.dt + self._dW[i - 1]

    def run(self, stim):
        self.stim_history.append(stim)
        for i in range(1, self.time.size):
            self.r[i] = self.euler_maruyama(self.r[i - 1], stim, i)
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
        near_count = (resps > self.count_thresh).sum(1)
        return near_count


def circ_distance(x, y):
    try:
        return np.arctan2(np.sin(x - y), np.cos(x - y))
    except Exception as e:
        return np.atan2(np.sin(x - y), np.cos(x - y))


ASD_COLOR, NT_COLOR = "#FF0000", "#00A08A"


def get_natural_stats_distribution(n_points, peaks=None, kappa=6):
    if peaks is None:
        peaks = [-np.pi, -np.pi / 2, 0, np.pi / 2]
    out = np.concatenate([np.random.vonmises(peak, kappa, n_points) for peak in peaks])
    np.random.shuffle(out)
    out += 3 * np.pi
    out %= 2 * np.pi
    return out - np.pi


# %%
N = 360
kappa_sharp = 8
kappa_wide = 2
n_stim = 500
np.random.seed(0)
normalize_fr = True
recalculate_connectivity = True
params = dict(j0=0.5, j1=0.5, h0=0.5, h1=0.5, N=360, lr=1e-2, T=1, dt=1e-2, noise=0, count_thresh=0.5,
              width_scaling=1.5, n_sims=1)
widths_ndr = [kappa_sharp] * N  # get_tuning_widths(N, kappa, precision=18, min_val=0.3)
widths_idr = [kappa_wide] * N  # get_tuning_widths(N, 3, precision=18, min_val=0.66)
# widths_idr = [kappa // 2] * N
gains = [1] * N  # get_location_based_increase(N, precision=6, min_val=0.75)

model_idr = Model(gains=gains, tuning_widths=widths_idr, tuning_func=vm_like, **params)
model_ndr = Model(gains=gains, tuning_widths=widths_ndr, tuning_func=vm_like, **params)

all_idr_resps = []
all_ndr_resps = []
all_idr_theta = []
all_ndr_theta = []
all_idr_tuning = []
all_ndr_tuning = []
# stim_list = np.arange(0, 2 * np.pi, np.pi / 180)
stim_list = get_natural_stats_distribution(n_stim) + np.pi
for stim in tqdm(stim_list):
    all_idr_resps.append(model_idr.run(stim))
    all_ndr_resps.append(model_ndr.run(stim))
    all_idr_theta.append(model_idr.theta.copy())
    all_ndr_theta.append(model_ndr.theta.copy())
    all_idr_tuning.append(model_idr.tuning_widths.copy())
    all_ndr_tuning.append(model_ndr.tuning_widths.copy())
    model_idr.update()
    model_ndr.update()

all_idr_resps = np.array(all_idr_resps)  # (stim, N, n_sims)
all_ndr_resps = np.array(all_ndr_resps)
all_idr_theta = np.array(all_idr_theta)
all_ndr_theta = np.array(all_ndr_theta)
all_idr_tuning = np.array(all_idr_tuning)
all_ndr_tuning = np.array(all_ndr_tuning)
# %% plot the initial value of the tuning widths and the final value of the tuning widths
plt.plot(model_idr.theta.get(), model_idr.tuning_widths.get(), color=ASD_COLOR, label="IDR")
plt.plot(model_ndr.theta.get(), model_ndr.tuning_widths.get(), color=NT_COLOR, label="NDR")
# set xlabels to be in radians, 0 till 2pi
plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
           ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
plt.xlabel("Preferred Orientation")
plt.ylabel(r"Tuning precision ($\kappa$)")
plt.hlines([kappa_sharp, kappa_wide], 0, 2 * np.pi, linestyles="--", colors='gray')
plt.legend()
plt.show()
# %%
fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(131, projection='polar')
ax2 = fig.add_subplot(132, projection='polar', sharey=ax1)
ax3 = fig.add_subplot(133, projection='polar', sharey=ax1)
wide_theta = model_idr.theta.get()
sharp_theta = model_ndr.theta.get()

axes = [ax1, ax2, ax3]
axes[0].hist(stim_list.get(), bins=120, density=True, alpha=0.5)
axes[1].hist(wide_theta, bins=120, density=True, color=ASD_COLOR, alpha=0.5)
axes[2].hist(sharp_theta, bins=120, density=True, color=NT_COLOR, alpha=0.5)
axes[0].set_title("Stimulus", fontsize=24)
axes[1].set_title("Wide", fontsize=24)
axes[2].set_title("Sharp", fontsize=24)
for ax in axes:
    # ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$",
                        r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$"], fontsize=24)
    ax.set_yticklabels([])

# add padding to the
# ax1.set_ylabel("Density",fontsize=24)
plt.show()

# %% plot the first theta and the final theta
fig, ax = plt.subplots()
ax.plot(np.linspace(0, 2 * np.pi, N).get(), model_idr.theta.get(), color=ASD_COLOR, label="IDR")
ax.plot(np.linspace(0, 2 * np.pi, N).get(), model_ndr.theta.get(), color=NT_COLOR, label="NDR")
ax.plot([0, 2 * np.pi], [0, 2 * np.pi], color="gray", linestyle='--')
ax.set_xlabel("Initial Theta")
ax.set_ylabel("Final Theta")
ax.set_title("Initial Theta VS Final Theta")
ax.legend()
# set xlabels to be in radians, 0 till 2pi
plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
           ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
plt.show()

# %%
a = all_idr_resps[0].var(0)
idr_var = get(all_idr_resps.mean(-1).var(-1))
ndr_var = get(all_ndr_resps.mean(-1).var(-1))
plt.scatter(get(stim_list), idr_var, label="IDR", s=2, alpha=0.5)
plt.scatter(get(stim_list), ndr_var, label="NDR", s=2, alpha=0.5)
cardinal_orientations = get(np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]))
plt.vlines(cardinal_orientations, min(idr_var.min(), ndr_var.min()),
           max(idr_var.max(), ndr_var.max()), linestyles="--",
           colors='gray', label="Cardinal")
plt.vlines(cardinal_orientations + (np.pi / 4), min(idr_var.min(), ndr_var.min()),
           max(idr_var.max(), ndr_var.max()), linestyles="--",
           colors='black', label="Oblique")

plt.legend()
plt.show()
# %% Calculate binned variance and bias for IDR and NDR in 1 degree bins
bin_size = 1
stim = np.arange(0, 360, bin_size) * np.pi / 180
idr_binned_var = np.zeros_like(stim)
ndr_binned_var = np.zeros_like(stim)
ndr_binned_bias = np.zeros_like(stim)
idr_binned_bias = np.zeros_like(stim)
for i, s in enumerate(tqdm(stim)):
    # run idr and ndr models on the stimulus
    idr_fr = model_idr.run(s)
    ndr_fr = model_ndr.run(s)

    # normalize the firing rates to a probability distribution and choose orientations based on the firing rates
    idr_fr -= idr_fr.min()
    idr_fr /= idr_fr.sum()
    ndr_fr -= ndr_fr.min()
    ndr_fr /= ndr_fr.sum()

    idr_choices = np.random.choice(np.squeeze(model_idr.theta), replace=True, p=np.squeeze(idr_fr), size=1000)
    ndr_choices = np.random.choice(np.squeeze(model_ndr.theta), replace=True, p=np.squeeze(ndr_fr), size=1000)

    # calculate the variance and bias of the choices
    idr_binned_var[i] = circvar(get(idr_choices))
    ndr_binned_var[i] = circvar(get(ndr_choices))
    idr_binned_bias[i] = circ_distance(get(idr_choices).mean(), s)
    ndr_binned_bias[i] = circ_distance(get(ndr_choices).mean(), s)

# calculate the derivative of the bias
idr_binned_bias = np.gradient(idr_binned_bias, bin_size)
ndr_binned_bias = np.gradient(ndr_binned_bias, bin_size)

# calculate the cramer-rao lower bound
idr_crlb = (1 + idr_binned_bias) / np.sqrt(idr_binned_var)
ndr_crlb = (1 + ndr_binned_bias) / np.sqrt(ndr_binned_var)

bound = 15

# plot the cramers-rao lower bound
plt.plot(get(stim)[15:-15], get(idr_crlb)[15:-15], label="IDR", color=ASD_COLOR)
plt.plot(get(stim)[15:-15], get(ndr_crlb)[15:-15], label="NDR", color=NT_COLOR)
plt.legend()
plt.show()

# %%
plt.plot(get(model_ndr.theta), get(model_ndr.tuning_widths), label="NDR")
plt.plot(get(model_idr.theta), get(model_idr.tuning_widths), label="IDR")
plt.vlines(cardinal_orientations, 2, 6, linestyles="--",
           colors='gray', label="Cardinal")
plt.vlines(cardinal_orientations + (np.pi / 4), 2, 6, linestyles="--",
           colors='black', label="Oblique")
plt.legend()
plt.show()
# %% get the idx of a cardinal orientation stimulus and an oblique orientation stimulus
cardinal_idx = get(np.argmin(np.abs(stim_list - np.pi))).item()
oblique_idx = get(np.argmin(np.abs(stim_list - np.pi / 4))).item()

# Plot one example of stimulus response
plt.plot(get(model_ndr.theta), get(all_ndr_resps[oblique_idx, :, 0]), label="Oblique")
plt.plot(get(model_ndr.theta), get(all_ndr_resps[cardinal_idx, :, 0]), label="Cardinal")
plt.gca().set_prop_cycle(None)
plt.plot(get(model_ndr.theta), get(all_idr_resps[oblique_idx, :, 0]), label="Oblique IDR", linestyle=":")
plt.plot(get(model_ndr.theta), get(all_idr_resps[cardinal_idx, :, 0]), label="Cardinal IDR", linestyle=":")
plt.legend()
plt.show()

# %%
model_idr.run(stim_list[cardinal_idx])
animate_model_example("idr_180", stim_list[cardinal_idx], model_idr.r, model_idr.theta, 0)

# %% calculate bias and variance
from scipy.stats import circvar

stimuli = np.linspace(0, 2 * np.pi, 101)
wide_bias = np.zeros_like(stimuli)
sharp_bias = np.zeros_like(stimuli)
wide_variance = np.zeros_like(stimuli)
sharp_variance = np.zeros_like(stimuli)
for i, stim in enumerate(tqdm(stimuli)):
    wide_fr = np.squeeze(model_idr.run(stim))
    sharp_fr = np.squeeze(model_ndr.run(stim))
    wide_choices = np.random.choice(model_idr.theta, replace=True,
                                    p=(wide_fr - wide_fr.min()) / (wide_fr - wide_fr.min()).sum(),
                                    size=1000)
    sharp_choices = np.random.choice(model_ndr.theta, replace=True,
                                     p=(sharp_fr - sharp_fr.min()) / (sharp_fr - sharp_fr.min()).sum(),
                                     size=1000)

    wide_bias[i] = circ_distance(wide_choices, stim).mean()
    sharp_bias[i] = circ_distance(sharp_choices, stim).mean()
    wide_variance[i] = circvar(get(wide_choices))
    sharp_variance[i] = circvar(get(sharp_choices))
smooth_wide_bias = scipy.ndimage.gaussian_filter1d(get(wide_bias), 0.75)
smooth_sharp_bias = scipy.ndimage.gaussian_filter1d(get(sharp_bias), 0.75)
smooth_wide_variance = scipy.ndimage.gaussian_filter1d(get(wide_variance), 0.75)
smooth_sharp_variance = scipy.ndimage.gaussian_filter1d(get(sharp_variance), 0.75)
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(get(stimuli), smooth_wide_bias, color=ASD_COLOR, label="Wide", linewidth=3)
ax.plot(get(stimuli), smooth_sharp_bias, color=NT_COLOR, label="Sharp", linewidth=3)
# set the xticks to 0-2pi in pi/4 increments
ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4, 2 * np.pi])
ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$",
                    r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$", r"$2\pi$"])
# plot gray dotted vlines at cardinal orientations, and a solid black like at 0
ax.vlines([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], get(wide_bias.min()).item(), get(wide_bias.max()).item(),
          color='gray', linestyle='--',
          lw=2)
ax.hlines(0, 0, 2 * np.pi, color='black', linestyle='-', lw=1)

ax.set_xlabel("Stimulus", fontsize=24)
ax.set_ylabel("Bias", fontsize=24)
# set yticklabels to a larger font size
ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
ax.legend(fontsize=20)

# remove the spines of ax4
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.savefig("bias_ring_model.pdf")
plt.show()
# %%
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(get(stimuli), get(wide_variance), color=ASD_COLOR, label="Wide", linewidth=3)
ax.plot(get(stimuli), get(sharp_variance), color=NT_COLOR, label="Sharp", linewidth=3)
# set the xticks to 0-2pi in pi/4 increments
ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4, 2 * np.pi])
ax.set_xticklabels([r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$",
                    r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$", r"$2\pi$"])
# plot gray dotted vlines at cardinal orientations, and a solid black like at 0
ax.vlines([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], get(sharp_variance.min()).item(),
          get(wide_variance.max()).item(), color='gray', linestyle='--',
          lw=2)
# ax.hlines(0, 0, 2*np.pi, color='black', linestyle='-', lw=1)

ax.set_xlabel("Stimulus", fontsize=24)
ax.set_ylabel("Variance", fontsize=24)
# set yticklabels to a larger font size
ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
ax.legend(fontsize=20)

# remove the spines of ax4
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
fig.tight_layout()
plt.savefig("variance_ring_model.pdf")
plt.show()
