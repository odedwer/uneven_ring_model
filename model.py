from tqdm import tqdm

from utils import get_natural_stats_distribution, circ_distance, is_iterable, vm_like

# import numpy as np
import cupy as np

import pandas as pd


class Model:
    def __init__(self, j0, j1, h0, h1, N, gains, tuning_widths, tuning_func, lr, count_thresh=0, width_scaling=1, T=1,
                 dt=1e-2, noise=0., stim_noise=0., n_sims=1000, nonlinearity=lambda x: x, limit_width=False):
        self.j0 = j0
        self.j1 = j1
        self.h0 = h0
        self.h1 = h1
        self.N = N
        self.T = T
        self.dt = dt
        self.time = np.arange(0, T + dt, dt)
        self.n_sims = n_sims
        self.limit_width = limit_width
        if is_iterable(gains):
            self.gains = np.array(gains)
        else:
            self.gains = np.ones((self.n_sims, N)).astype(float) * gains
        self.theta = np.repeat(np.linspace(0, 2 * np.pi, N), self.n_sims).reshape((self.n_sims, N), order='F')
        if is_iterable(tuning_widths):
            self.tuning_widths = np.array(tuning_widths).astype(float)
        else:
            self.tuning_widths = np.ones((self.n_sims, N)).astype(float) * tuning_widths
        self.tuning_func = tuning_func

        self.J = (1 / self.N) * (self.j0 + self.j1 * np.cos(self.theta[:, None, :] - self.theta[..., None]))

        self.r = np.zeros((self.time.size, n_sims, N))
        self.noise = noise
        self._dW = np.random.normal(loc=0.0, scale=np.sqrt(dt) * self.noise, size=(self.time.size - 1, self.n_sims, N))
        self.lr = lr
        self.width_scaling = width_scaling
        self.count_thresh = count_thresh
        self.base_factor = self.get_near_factor()
        self.stim_noise = stim_noise
        self.stim_history = []
        self.nonlinearity = nonlinearity

    def deterministic_func(self, y, stim):
        return -y + self.nonlinearity(
            ((self.h0 + self.h1 * self.tuning_func((self.theta - stim), self.tuning_widths)) * self.gains) +
            np.squeeze(self.J @ y[..., None], -1)
        )

    def euler_maruyama(self, y, stim, i):
        return y + self.deterministic_func(y, stim) * self.dt + self._dW[i - 1]

    def run(self, stim, save_stim=True):
        if not is_iterable(stim):
            stim = np.repeat(np.array([stim]), self.n_sims).reshape((1, self.n_sims, 1))
        else:
            stim = np.array(stim).reshape((1, self.n_sims, 1))
        if save_stim:
            self.stim_history.append(stim[0])
        # add noise to the stimulus, different noise for each neuron and time point
        noisy_stim = stim + np.random.normal(0, self.stim_noise, (self.time.size, self.n_sims, self.N))
        noisy_stim[noisy_stim < 0] = 2 * np.pi + noisy_stim[noisy_stim < 0]
        noisy_stim %= 2 * np.pi
        for i in range(1, self.time.size):
            self.r[i] = self.euler_maruyama(self.r[i - 1], noisy_stim[i], i)
        return self.r[-1].copy()

    def update(self, normalize_fr=True, recalculate_connectivity=False):
        fr = self.r[-1, :, :]
        # Perform hebbian learning of self.theta based on the firing rates and the last stimulus
        if normalize_fr:
            fr = (fr - fr.min(-1, keepdims=True)) / (fr.max(-1, keepdims=True) - fr.min(-1, keepdims=True))
        dist = circ_distance(self.stim_history[-1], self.theta)
        old_theta, old_widths = self.theta.copy(), self.tuning_widths.copy()
        self.theta += self.lr * fr * np.squeeze(dist)
        self.theta[self.theta < 0] = (2 * np.pi) + self.theta[self.theta < 0]
        self.theta %= 2 * np.pi
        self.tuning_widths *= np.abs((self.width_scaling * (
                (self.get_near_factor().astype(float) / self.base_factor.astype(float)) - 1)) + 1)
        if self.limit_width:
            self.tuning_widths = np.maximum(self.tuning_widths, 1)

        if recalculate_connectivity:
            self.J = (1 / self.N) * (self.j0 + self.j1 * np.cos(self.theta[:, None, :] - self.theta[..., None]))
        return old_theta, old_widths

    def get_near_factor(self):
        resps = self.tuning_func(self.theta[:, None, :] - self.theta[..., None], self.tuning_widths[..., None])
        # for each neuron in the response, count the number of self.theta values that have a response greater than self.count_thresh
        near_count = (resps > self.count_thresh).sum(-1)
        return near_count



def train_model(stimuli, j0, j1, h0, h1, N, lr, T, dt, noise, stim_noise,
                count_thresh, width_scaling, n_sims, nonlinearity,
                tuning_widths, tuning_func, gains,
                update, recalculate_connectivity, normalize_fr, limit_width, use_tqdm=False, save_process=False
                ):
    """
    Train the model with the given parameters
    :param params: dictionary of parameters
    :return: trained model
    """
    model = Model(j0=j0, j1=j1, h0=h0, h1=h1, N=N, lr=lr, T=T, dt=dt, noise=noise, stim_noise=stim_noise,
                  count_thresh=count_thresh, width_scaling=width_scaling, n_sims=n_sims, nonlinearity=nonlinearity,
                  tuning_widths=tuning_widths, tuning_func=tuning_func, gains=gains, limit_width=limit_width)
    if save_process:
        learning_thetas = [model.theta.copy()]
        learning_tuning_widths = [model.tuning_widths.copy()]
        learning_connectivity = [model.J.copy()]
    for stim in tqdm(stimuli) if use_tqdm else stimuli:
        model.run(np.squeeze(stim))
        if update:
            model.update(recalculate_connectivity=recalculate_connectivity, normalize_fr=normalize_fr)
            if save_process:
                learning_thetas.append(model.theta.copy())
                learning_tuning_widths.append(model.tuning_widths.copy())
                learning_connectivity.append(model.J.copy())
    if save_process:
        return model, learning_thetas, learning_tuning_widths, learning_connectivity
    return model, None, None, None
