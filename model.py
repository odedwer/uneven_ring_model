import functools

import cupy.random
import scipy
# import numpy as np
import matplotlib.pyplot as plt
import os

from matplotlib.animation import FuncAnimation
from tqdm import tqdm

import cupy as np


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

    rng:tqdm = tqdm(range(get(y).shape[0]))
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
    def __init__(self, j0, j1, h0, h1, N, tuning_locs, tuning_widths, tuning_func, T=1, dt=1e-2, noise=0., n_sims=1000):
        self.j0 = j0
        self.j1 = j1
        self.h0 = h0
        self.h1 = h1
        self.N = N
        self.T = T
        self.dt = dt
        self.time = np.arange(0, T + dt, dt)
        self.n_sims = n_sims
        self.tuning_locs = np.array(tuning_locs)
        self.theta = np.linspace(0, 2 * np.pi, N)
        self.tuning_widths = np.array(tuning_widths)
        self.tuning_func = tuning_func
        self.J = (1 / self.N) * (self.j0 + self.j1 * np.cos(self.theta[:, None] - self.theta[None, :]))
        self.r = np.zeros((self.time.size, N, n_sims))
        self.noise = noise
        self._dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(self.time.size - 1, N, self.n_sims)) * self.noise

    def deterministic_func(self, y, stim):
        return -y + ((self.h0 + self.h1 * self.tuning_func((self.theta - stim),
                                                           self.tuning_widths)) * self.tuning_locs)[:,
                    None] + self.J @ y

    def euler_maruyama(self, y, stim, i):
        return y + self.deterministic_func(y, stim) * self.dt + self._dW[i - 1]

    def run(self, stim):
        for i in range(1, self.time.size):
            self.r[i] = self.euler_maruyama(self.r[i - 1], stim, i)
        return self.r[-1].copy()


# %%
N = 360
kappa = 6
params = dict(j0=0.5, j1=0.5, h0=0.5, h1=0.5, N=360, T=1, dt=1e-2, noise=0.1)
widths_ndr = get_tuning_widths(N, kappa, precision=18, min_val=0.3)
widths_idr = get_tuning_widths(N, 3, precision=18, min_val=0.66)
# widths_idr = [kappa // 2] * N
locs = [1] * N  # get_location_based_increase(N, precision=6, min_val=0.75)

model_idr = Model(tuning_locs=locs, tuning_widths=widths_idr, tuning_func=vm_like, **params)
model_ndr = Model(tuning_locs=locs, tuning_widths=widths_ndr, tuning_func=vm_like, **params)

all_idr_resps = []
all_ndr_resps = []
stim_list = np.arange(0, 2 * np.pi, np.pi / 180)

for stim in tqdm(stim_list):
    all_idr_resps.append(model_idr.run(stim))
    all_ndr_resps.append(model_ndr.run(stim))

all_idr_resps = np.array(all_idr_resps)  # (stim, N, n_sims)
all_ndr_resps = np.array(all_ndr_resps)

# %%
a = all_idr_resps[0].var(0)
idr_var = get(all_idr_resps.mean(-1).var(-1))
ndr_var = get(all_ndr_resps.mean(-1).var(-1))
plt.plot(get(stim_list), idr_var, label="IDR")
plt.plot(get(stim_list), ndr_var, label="NDR")
cardinal_orientations = get(np.array([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]))
plt.vlines(cardinal_orientations, min(idr_var.min(), ndr_var.min()),
           max(idr_var.max(), ndr_var.max()), linestyles="--",
           colors='gray', label="Cardinal")
plt.vlines(cardinal_orientations + (np.pi / 4), min(idr_var.min(), ndr_var.min()),
           max(idr_var.max(), ndr_var.max()), linestyles="--",
           colors='black', label="Oblique")

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
# %% Plot one example of stimulus response
plt.plot(get(model_ndr.theta), get(all_ndr_resps[135, :, 0]), label="Oblique")
# plt.plot(get(model_ndr.theta), get(np.roll(all_ndr_resps[90, :, 0], 135 - 90)), label="Cardinal")
plt.plot(get(model_ndr.theta), get(all_ndr_resps[90, :, 0]), label="Cardinal")
plt.plot(get(model_ndr.theta), get(all_idr_resps[135, :, 0]), label="Oblique IDR")
# plt.plot(get(model_ndr.theta), get(np.roll(all_idr_resps[90, :, 0], 135 - 90)), label="Cardinal IDR")
plt.plot(get(model_ndr.theta), get(all_idr_resps[90, :, 0]), label="Cardinal IDR")
plt.legend()
plt.show()

# %%
stim_list[135] * 180 / np.pi

a = all_ndr_resps[135, :, :].var(0).mean()
aa = all_ndr_resps[90, :, :].var(0).mean()
# %%
stim1 = np.pi / 6
stim2 = np.pi / 4
idr_resps_pi2 = model_idr.run(stim1)
idr_resps_pi = model_idr.run(stim2)

plt.plot(get(model_idr.theta), get(idr_resps_pi[:, 0]), label="stim 1")
plt.plot(get(model_idr.theta), get(idr_resps_pi2[:, 0]), label="stim 2")
plt.vlines([stim1, stim2], 0, 1, linestyles="--")
plt.legend()
plt.show()
# %%
model_ndr.run(stim_list[90])
animate_model_example("ndr_90", stim_list[90], model_ndr.r, model_ndr.theta, 0)
