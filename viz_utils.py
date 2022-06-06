import os
from pathlib import Path
from turtle import color
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def values2color_dict(
    value_list, cmap="hot", range=(0.2, 0.7), reverse=True, given_values=None
):
    value_unique = np.unique(value_list)
    value_len = len(value_unique)
    cmap = matplotlib.cm.get_cmap(cmap)
    if given_values is not None:
        value_normalized = given_values
    else:
        value_normalized = np.linspace(range[0], range[1], num=value_len)
    if reverse:
        value_normalized = np.flip(value_normalized)
    val2color = {}
    for value in value_unique:
        index = np.where(value_unique == value)[0][0]
        # color_unique.append(cmap(value_normalized[index]))
        color = cmap(value_normalized[index])
        val2color.update({value: color})

    return val2color


def window_avg_plot(
    axes: matplotlib.axes.Axes,
    var: Union[np.array, List[float]],
    window_sz: int = 250,
    zoom_factor: float = 1.0,
    var_name: Optional[str] = None,
    set_xlabel: bool = False,
) -> None:
    """Computes the average over successive windows of an array and plot correspoding plot

    Args:
        axes (matplotlib.axes.Axes): _description_
        var (Union[np.array, List[float]]): _description_
        window_sz (int, optional): _description_. Defaults to 250.
        zoom_factor (float, optional): _description_. Defaults to 1.0.
        var_name (Optional[str], optional): _description_. Defaults to None.
        set_xlabel (bool, optional): _description_. Defaults to False.
    """

    var_np = np.array(var)
    avg_var = np.empty((int(var_np.shape[0] / window_sz),))
    for i in range(0, int(var_np.shape[0] / window_sz)):
        avg_var[i] = var_np[window_sz * i : window_sz * (i + 1) - 1].sum() / window_sz
    episodes = (
        np.linspace(1, avg_var.shape[0], avg_var.shape[0]) * window_sz * zoom_factor
    )
    axes.plot(episodes, avg_var)
    if var_name is not None:
        axes.set_ylabel(var_name)
    if set_xlabel:
        axes.set_xlabel("Episode")


def test_window_avg_plot():
    _, ax = plt.subplots(figsize=(10, 4))
    rewards_example = np.linspace(-1, 1, 20000) + np.random.rand(20000)
    window_avg_plot(axes=ax, var=rewards_example, window_sz=20)


def metrics_plots(
    m_opts: List[float],
    m_rands: List[float],
    figsize: Tuple[float] = (10, 6),
    val_interval: int = 250,
    figtitle: Optional[str] = None,
    save_dir: Union[Path, str] = "plot",
    save_fn: Optional[str] = None,
    suffix: str = "_metrics",
    viz_fig: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    base = [val_interval * (i + 1) for i in range(len(m_opts))]
    axes[0].plot(base, m_opts)
    axes[0].set_ylabel("$m_{opt}$")
    axes[1].plot(base, m_rands)
    axes[1].set_ylabel("$m_{rand}$")
    axes[1].set_xlabel("Episode")
    if figtitle is not None:
        fig.suptitle(figtitle)

    if save_fn is not None:
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, save_fn + suffix + ".pdf"), dpi=300)
        if viz_fig:
            plt.show()


def mul_metrics_plots(
    metrics_dict: Dict,
    val_list: List[float] = None,  # n_star_list | eps_list
    val4label: Optional[str] = None,  # n^{*} | {\epsilon}_{opt}
    label_latex: bool = True,
    figsize: Tuple[float] = (10, 6),
    val_interval: int = 250,
    figtitle: Optional[str] = None,
    save_dir: Union[Path, str] = "plot",
    save_fn: Optional[str] = None,
    suffix: str = "_metrics_mul",
    xlim_max: float = 26000.0,
    viz_fig: bool = False,
    cmap: Optional[str] = "hot",
    include_max: bool = False,
    include_last: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    game_idx = [
        val_interval * (i + 1) for i in range(len(metrics_dict["M_rand"][val_list[0]]))
    ]
    label_prefix = ""
    if val4label is not None:
        label_prefix += f"{val4label} ="
    val2color = values2color_dict(val_list, cmap=cmap)
    ls_set = ["-", "--", "-."]
    # m_opts
    for idx, val in enumerate(val_list):
        if label_latex:
            label_name = f"${label_prefix} {val}$"
        else:
            label_name = f"{label_prefix} {val}"
        # m_opts
        label_name_opt = label_name
        M_opt = metrics_dict["M_opt"][val]
        if include_max:
            label_name_opt += " (max: {:.3f})".format(max(M_opt))
        if include_last:
            label_name_opt += " (last: {:.3f})".format(M_opt[-1])
        axes[0].plot(
            game_idx,
            M_opt,
            ls=ls_set[idx % len(ls_set)],
            color=val2color[val],
            label=label_name_opt,
            lw=2,
        )

        # m_rands
        label_name_rand = label_name
        M_rand = metrics_dict["M_rand"][val]
        if include_max:
            label_name_rand += " (max: {:.3f})".format(max(M_rand))
        if include_last:
            label_name_rand += " (last: {:.3f})".format(M_rand[-1])
        axes[1].plot(
            game_idx,
            M_rand,
            ls=ls_set[idx % len(ls_set)],
            color=val2color[val],
            label=label_name_rand,
            lw=2,
        )

    axes[0].set_ylabel("$m_{opt}$")
    axes[0].legend(title="$M_{opt}$")
    axes[0].set_xlim([0, xlim_max])

    axes[1].set_ylabel("$m_{rand}$")
    axes[1].set_xlabel("Episode")
    axes[1].legend(title="$M_{rand}$")
    axes[1].set_xlim([0, xlim_max])

    if figtitle is not None:
        fig.suptitle(figtitle)

    if save_fn is not None:
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, save_fn + suffix + ".pdf"), dpi=300)
        if viz_fig:
            plt.show()


def reward_loss_plots(
    rewards: List[float],
    losses: List[float],
    figsize: Tuple[float] = (10, 6),
    figtitle: Optional[str] = None,
    save_dir: Union[Path, str] = "plot",
    save_fn: Optional[str] = None,
    suffix: str = "_reward_loss",
    viz_fig: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes[0].plot([250 * (i + 1) for i in range(len(losses))], losses)
    axes[0].set_ylabel("Avg. Loss")
    window_avg_plot(axes=axes[1], var=rewards, var_name="Avg. Reward", set_xlabel=True)
    if figtitle is not None:
        fig.suptitle(figtitle)

    if save_fn is not None:
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, save_fn + suffix + ".pdf"), dpi=300)
        if viz_fig:
            plt.show()


def mul_reward_loss_plots(
    reward_dict: Dict,
    loss_dict: Dict,
    val_list: List[float] = None,  # n_star_list | eps_list
    val4label: Optional[str] = None,  # {\epsilon} | n^{*} | {\epsilon}_{opt}
    label_latex: bool = True,
    figsize: Tuple[float] = (10, 6),
    val_interval: int = 250,
    figtitle: Optional[str] = None,
    save_dir: Union[Path, str] = "plot",
    save_fn: Optional[str] = None,
    suffix: str = "_reward_loss_mul",
    xlim_max: float = 26000.0,
    viz_fig: bool = False,
    cmap: Optional[str] = "hot",
    include_last: bool = False,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    game_idx = [val_interval * (i + 1) for i in range(len(loss_dict[val_list[0]]))]

    label_prefix = ""
    if val4label is not None:
        label_prefix += f"{val4label} ="
    val2color = values2color_dict(val_list, cmap=cmap)
    ls_set = ["-", "--", "-."]
    for idx, val in enumerate(val_list):
        if label_latex:
            label_name = f"${label_prefix} {val}$"
        else:
            label_name = f"{label_prefix} {val}"
        # loss
        label_name_loss = label_name
        loss = loss_dict[val]
        if include_last:
            label_name_loss += " (last: {:.3f})".format(loss[-1])
        axes[0].plot(
            game_idx,
            loss,
            ls=ls_set[idx % len(ls_set)],
            color=val2color[val],
            label=label_name_loss,
            lw=2,
        )

        # reward
        label_name_reward = label_name
        reward = reward_dict[val]
        reward = np.array(reward)
        avg_reward = np.empty((int(reward.shape[0] / val_interval),))
        for i in range(0, int(reward.shape[0] / val_interval)):
            avg_reward[i] = (
                reward[val_interval * i : val_interval * (i + 1) - 1].sum()
                / val_interval
            )
        episodes = [(i + 1) * val_interval for i in range(avg_reward.shape[0])]

        if include_last:
            label_name_reward += " (last: {:.3f})".format(avg_reward[-1])
        axes[1].plot(
            episodes,
            avg_reward,
            ls=ls_set[idx % len(ls_set)],
            color=val2color[val],
            label=label_name_reward,
            lw=2,
        )

    axes[0].set_ylabel("Avg. Loss")
    axes[0].legend(title="$Loss_{avg}$")
    axes[0].set_xlim([0, xlim_max])

    axes[1].set_ylabel("Avg. Reward")
    axes[1].set_xlabel("Episode")
    axes[1].legend(title="$Reward_{avg}$")
    axes[1].set_xlim([0, xlim_max])

    if figtitle is not None:
        fig.suptitle(figtitle)

    if save_fn is not None:
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, save_fn + suffix + ".pdf"), dpi=300)
        if viz_fig:
            plt.show()
