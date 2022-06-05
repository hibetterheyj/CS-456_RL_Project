import os
from typing import Optional, Union, List, Tuple
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


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
        np.linspace(0, avg_var.shape[0], avg_var.shape[0]) * window_sz * zoom_factor
    )
    axes.plot(episodes, avg_var)
    if var_name is not None:
        axes.set_ylabel(var_name)
    if set_xlabel:
        axes.set_xlabel('Episode')


def reward_loss_plots(
    rewards: List[float],
    losses: List[float],
    figsize: Tuple[float] = (10, 6),
    figtitle: Optional[str] = None,
    save_dir: Union[Path, str] = 'plot',
    save_fn: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    axes[0].plot([250 * i for i in range(len(losses))], losses)
    axes[0].set_ylabel('Avg. Loss')
    window_avg_plot(axes=axes[1], var=rewards, var_name='Avg. Reward', set_xlabel=True)
    if figtitle is not None:
        fig.suptitle(figtitle)

    if save_fn is not None:
        plt.savefig(os.path.join(save_dir, save_fn), dpi=300)
        plt.show()


def metrics_plots(
    m_opts: List[float],
    m_rands: List[float],
    figsize: Tuple[float] = (10, 6),
    val_interval: int = 250,
    figtitle: Optional[str] = None,
    save_dir: Union[Path, str] = 'plot',
    save_fn: Optional[str] = None,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    base = [val_interval * i for i in range(len(m_opts))]
    axes[0].plot(base, m_opts)
    axes[0].set_ylabel('$m_{opt}$')
    axes[1].plot(base, m_rands)
    axes[1].set_ylabel('$m_{rand}$')
    axes[1].set_xlabel('Episode')
    if figtitle is not None:
        fig.suptitle(figtitle)

    if save_fn is not None:
        plt.savefig(os.path.join(save_dir, save_fn), dpi=300)
        plt.show()


def test_window_avg_plot():
    _, ax = plt.subplots(figsize=(10, 4))
    rewards_example = np.linspace(-1, 1, 20000) + np.random.rand(20000)
    window_avg_plot(axes=ax, var=rewards_example, window_sz=20)
