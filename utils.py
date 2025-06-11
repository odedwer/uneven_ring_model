import functools
import sys

# import cupy.random
import scipy
# import numpy as np
import cupy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy.stats as stats
from scipy.stats import circvar, circmean
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
from itertools import product

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


# def circ_distance(x, y):
#     return np.atan2(np.sin(x - y), np.cos(x - y))


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


def get_bias_variance(model, sigma=0.75, seed=97, choice_thresh=None):
    np.random.seed(seed)
    stimuli = np.linspace(0, np.pi, 91)
    bias = np.zeros_like(stimuli)
    bias_ci = np.zeros(stimuli.shape)
    variance = np.zeros_like(stimuli)
    for i, stim in enumerate(stimuli):
        choices = get_choices(model, stim, n_choices=10000, seed=seed, choice_thresh=choice_thresh)
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


def is_iterable(var):
    """
    Check if a variable is iterable (not a string)
    :param var: variable to check
    :return: True if iterable, False otherwise
    """
    try:
        iter(var)
    except Exception:
        return False
    return not isinstance(var, str)


def get_choices(model, stim, n_choices=10000, seed=97, choice_thresh=None):
    np.random.seed(seed)
    resps = np.squeeze(model.run(stim))
    if choice_thresh is not None:
        if isinstance(choice_thresh,int) or isinstance(choice_thresh, float):
            resps[resps < choice_thresh] = 0
        elif choice_thresh == "h0":
            resps[resps < model.h0] = 0
        elif choice_thresh == "bayesian":
            resps*=model.tuning_widths
    prob = np.squeeze((resps - resps.min()) / (resps - resps.min()).sum())
    try:
        choices = np.random.choice(np.squeeze(model.theta), replace=True,
                               p=prob, size=n_choices)
    except ValueError:
        choices = np.full(n_choices, np.nan)
    return choices

def reload(func):
    del sys.modules[func.__module__]
    exec(f"from {func.__module__} import {func.__name__}")

def get_skew(x):
    """
    Calculate the skewness of a distribution
    :param x: distribution
    :return: skewness
    """
    return stats.skew(get(x), axis=0)
