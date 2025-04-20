from utils import get, vm_like, get_natural_stats_distribution, get_bias_variance, circ_distance, get_choices, \
    get_choice_distribution_width
import cupy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ASD_COLOR, NT_COLOR = "#FF0000", "#00A08A"


def _prep_model_for_main_plot(model, oblique_stim, cardinal_stim,choice_thresh):
    theta = model.theta.get()
    oblique_choices = get_choices(model, oblique_stim, n_choices=10000,choice_thresh=choice_thresh)
    cardinal_choices = get_choices(model, cardinal_stim, n_choices=10000,choice_thresh=choice_thresh)
    oblique_choices_width = get_choice_distribution_width(oblique_choices)
    cardinal_choices_width = get_choice_distribution_width(cardinal_choices)
    return theta, oblique_choices, cardinal_choices, oblique_choices_width, cardinal_choices_width


def main_plot(stim_list, model_idr, model_ndr, choice_thresh=None):
    oblique_stim = 5 * np.pi / 4
    cardinal_stim = np.pi

    idr_theta, oblique_idr_choices, cardinal_idr_choices, oblique_idr_choices_width, cardinal_idr_choices_width = _prep_model_for_main_plot(
        model_idr, oblique_stim, cardinal_stim, choice_thresh)
    ndr_theta, oblique_ndr_choices, cardinal_ndr_choices, oblique_ndr_choices_width, cardinal_ndr_choices_width = _prep_model_for_main_plot(
        model_ndr, oblique_stim, cardinal_stim, choice_thresh)

    bias_idr, variance_idr, stimuli, bias_ci_idr = get_bias_variance(model_idr, sigma=1, choice_thresh=choice_thresh)
    bias_ndr, variance_ndr, _, bias_ci_ndr = get_bias_variance(model_ndr, sigma=1, choice_thresh=choice_thresh)

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

    idr_sort_idx = np.argsort(model_idr.theta).get()
    ndr_sort_idx = np.argsort(model_ndr.theta).get()
    ax4.plot(model_idr.theta[idr_sort_idx].get(), model_idr.tuning_widths[idr_sort_idx].get(), color=ASD_COLOR,
             label="IDR", linewidth=3)
    ax4.plot(model_ndr.theta[ndr_sort_idx].get(), model_ndr.tuning_widths[ndr_sort_idx].get(), color=NT_COLOR,
             label="NDR", linewidth=3)
    # set xlabels to be in radians, 0 till 2pi
    ax4.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                   ["0", r"$\frac{\pi}{2}$", r"$\pi$", r"$\frac{3\pi}{2}$", r"$2\pi$"])
    ax4.text(0.5, -0.2, "Preferred Orientation", ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    ax4.set_ylabel("Relative change in width", fontsize=14, fontweight='bold')
    # ax4.hlines(1, 0, 2 * np.pi, linestyles="--", colors='gray')
    ax4.legend()

    def plot_choice_hist(ax: plt.Axes, choices, percent_close, color, title, ax_inset_share=None):
        ax.hist(get(choices), bins=40, density=True, alpha=0.5, color=color)
        ax.text(0.5, 0.975, title, ha='center', va='center', fontsize=16, fontweight='bold',
                transform=ax.transAxes)

    # add the width of the choices to the plot in text
    plot_choice_hist(ax5, oblique_idr_choices, oblique_idr_choices_width, ASD_COLOR, "Oblique IDR")
    ax5.text(0.5, 0.75, f"Width: {oblique_idr_choices.var().item():.2g}", ha='center', va='center',
             fontsize=12,
             fontweight='bold', transform=ax5.transAxes)
    plot_choice_hist(ax6, cardinal_idr_choices, cardinal_idr_choices_width, ASD_COLOR, "Cardinal IDR")
    ax6.text(0.5, 0.75, f"Width: {cardinal_idr_choices.var().item():.2g}", ha='center', va='center',
             fontsize=12, fontweight='bold', transform=ax6.transAxes)
    plot_choice_hist(ax7, oblique_ndr_choices, oblique_ndr_choices_width, NT_COLOR, "Oblique NDR")
    ax7.text(0.5, 0.75, f"Width: {oblique_ndr_choices.var().item():.2g}", ha='center', va='center',
             fontsize=12, fontweight='bold', transform=ax7.transAxes)
    plot_choice_hist(ax8, cardinal_ndr_choices, cardinal_ndr_choices_width, NT_COLOR, "Cardinal NDR")
    ax8.text(0.5, 0.75, f"Width: {cardinal_ndr_choices.var().item():.2g}", ha='center', va='center',
             fontsize=12, fontweight='bold', transform=ax8.transAxes)

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


def plot_firing_rate_for_stims(model_idr, model_ndr):
    oblique_stim = 3 * np.pi / 4
    cardinal_stim = np.pi
    near_oblique_stim = oblique_stim + np.pi / 12
    near_cardinal_stim = cardinal_stim + np.pi / 12
    center_stim = 3 * np.pi / 8
    near_center_stim = center_stim + np.pi / 12

    oblique_idr_resps = np.squeeze(model_idr.run(oblique_stim))
    cardinal_idr_resps = np.squeeze(model_idr.run(cardinal_stim))
    near_oblique_idr_resps = np.squeeze(model_idr.run(near_oblique_stim))
    near_cardinal_idr_resps = np.squeeze(model_idr.run(near_cardinal_stim))
    center_idr_resps = np.squeeze(model_idr.run(center_stim))
    near_center_idr_resps = np.squeeze(model_idr.run(near_center_stim))
    oblique_ndr_resps = np.squeeze(model_ndr.run(oblique_stim))
    cardinal_ndr_resps = np.squeeze(model_ndr.run(cardinal_stim))
    near_oblique_ndr_resps = np.squeeze(model_ndr.run(near_oblique_stim))
    near_cardinal_ndr_resps = np.squeeze(model_ndr.run(near_cardinal_stim))
    center_ndr_resps = np.squeeze(model_ndr.run(center_stim))
    near_center_ndr_resps = np.squeeze(model_ndr.run(near_center_stim))

    plt.rcParams.update(
        {
            "legend.fontsize": 14,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "axes.titleweight": 'bold',
            "axes.labelweight": 'bold',
            "xtick.labelsize": 16,
            "ytick.labelsize": 14,
            'axes.spines.right': False,
            'axes.spines.top': False,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    for ax, resp, title, theta, (stim, near_stim) in zip(
            axes.flatten(),
            [(oblique_idr_resps, near_oblique_idr_resps), (cardinal_idr_resps, near_cardinal_idr_resps),
             (center_idr_resps, near_center_idr_resps), (oblique_ndr_resps, near_oblique_ndr_resps),
             (cardinal_ndr_resps, near_cardinal_ndr_resps), (center_ndr_resps, near_center_ndr_resps)],
            ["Oblique IDR", "Cardinal IDR", "Center IDR", "Oblique NDR", "Cardinal NDR", "Center NDR"],
            [model_idr.theta, model_idr.theta, model_idr.theta, model_ndr.theta, model_ndr.theta, model_ndr.theta],
            [(oblique_stim, near_oblique_stim), (cardinal_stim, near_cardinal_stim), (center_stim, near_center_stim),
             (oblique_stim, near_oblique_stim), (cardinal_stim, near_cardinal_stim), (center_stim, near_center_stim)]
    ):
        ax: plt.Axes
        sort_idx = np.argsort(theta)
        ax.plot(get(theta[sort_idx]), get(resp[0][sort_idx]), label="Exact")
        ax.plot(get(theta[sort_idx]), get(resp[1][sort_idx]), label="Near")
        # set xticks to theta
        ax.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4],
                      [r"$0$", r"$\frac{\pi}{4}$", r"$\frac{\pi}{2}$", r"$\frac{3\pi}{4}$", r"$\pi$",
                       r"$\frac{5\pi}{4}$", r"$\frac{3\pi}{2}$", r"$\frac{7\pi}{4}$"]
                      )
        ax.hlines(model_idr.h0, 0, 2 * np.pi, colors='gray', linestyles='--')
        ax.axvline(stim, color='b', linestyle='--', label="Exact Stimulus")
        ax.axvline(near_stim, color='r', linestyle='--', label="Near Stimulus")
        resp[1][resp[1] < model_idr.h0] = 0
        normed_resp = (resp[1] - resp[1].min()) / (resp[1] - resp[1].min()).sum()
        ax.axvline(get(normed_resp * theta).sum(), 0, 2 * np.pi, color='green', linestyle='--', label="Mean resp")
        ax.legend()
    axes[0, 0].set_title("Oblique")
    axes[0, 1].set_title("Cardinal")
    axes[0, 2].set_title("Center")
    axes[0, 0].set_ylabel("IDR\n\nFiring Rate")
    axes[1, 0].set_ylabel("NDR\n\nFiring Rate")
    axes[1, 0].set_xlabel("Orientation ($\\theta$)")
    axes[1, 1].set_xlabel("Orientation ($\\theta$)")
    axes[1, 2].set_xlabel("Orientation ($\\theta$)")
    fig.tight_layout()
    plt.show()
