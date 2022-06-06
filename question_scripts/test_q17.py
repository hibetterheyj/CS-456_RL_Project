"""
_summary_: In self-practice, training with decreasing eps given different values of n*
"""

# std
import os
from typing import List, Dict, Tuple

# imported
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# customized
from tic_env import TictactoeEnv, OptimalPlayer
from dqn_utils import *
from dqn_player import DQNPlayer, setup_seed
from viz_utils import reward_loss_plots, metrics_plots, mul_metrics_plots


os.makedirs('plot', exist_ok=True)
os.makedirs('res', exist_ok=True)

# setup seed for random, numpy, and torch
setup_seed(2022)

save_prefix = 'question17'
save_single = True
save_all = True

val_interval = 250
n_star_list = [1, 100, 1000, 10000, 20000, 40000]

rewards_list = []
losses_list = []
metrics_dict = {"M_opt": {}, "M_rand": {}}

# [1, 100, 1000, 10000, 20000, 40000]
# [1, 20000]


def get_res_from_tests(include_max, include_last):
    dqn_players = [
        DQNPlayer(
            epsilon=0.2,  # according to q16
            buffer_sz=10000,
            batch_sz=64,
            explore=True,
            n_star=n_star,
            verbose=False,
            eps_min=0.1,
            eps_max=0.8,
        )
        for n_star in n_star_list
    ]

    for idx, n_star in enumerate(n_star_list):
        # print(f'{idx+1}/{len(n_star_list)}: n_star = {n_star}')
        dqn_player = dqn_players[idx]
        rewards, losses = dqn_player.self_train(
            nr_episodes=20000, val_interval=val_interval
        )

        ## data collection
        metrics_dict["M_opt"].update({n_star: dqn_player.m_opts})
        metrics_dict["M_rand"].update({n_star: dqn_player.m_rands})

        ## viz
        if save_single:
            inside_prefix = save_prefix + f'_n_star{n_star}'
            reward_loss_plots(rewards, losses, save_dir='plot', save_fn=inside_prefix)
            metrics_plots(
                dqn_player.m_opts,
                dqn_player.m_rands,
                val_interval=val_interval,
                save_dir='plot',
                save_fn=inside_prefix,
            )

            np.savez_compressed(
                os.path.join('res', inside_prefix),
                rewards=rewards,
                losses=losses,
                m_opts=dqn_player.m_opts,
                m_rands=dqn_player.m_rands,
            )
            # res = np.load(os.path.join('res', inside_prefix + '.npz'))

    if save_all:
        mul_metrics_plots(
            metrics_dict=metrics_dict,
            val_list=n_star_list,
            val4label='n^{*}',
            label_latex=True,
            figsize=(10, 6),
            save_dir='plot',
            save_fn=save_prefix,
            include_max=include_max,
        )


def get_res_from_saves(include_max):
    max_m_opts, max_m_rands = [], []
    for n_star in n_star_list:
        # print(f'{idx+1}/{sample_number}: n_star = {n_star}')
        inside_prefix = save_prefix + f'_n_star{n_star}'
        res = np.load(os.path.join('res', inside_prefix + '.npz'))
        m_opts = res['m_opts']
        m_rands = res['m_rands']
        max_m_opts.append(max(m_opts))
        max_m_rands.append(max(m_rands))
        # max_m_opts.append(m_opts[-1])
        # max_m_rands.append(m_rands[-1])

        ## data collection
        metrics_dict["M_opt"].update({n_star: m_opts})
        metrics_dict["M_rand"].update({n_star: m_rands})

    max_m_opts_indices = [i for i, x in enumerate(max_m_opts) if x == max(max_m_opts)]
    max_m_rands_indices = [
        i for i, x in enumerate(max_m_rands) if x == max(max_m_rands)
    ]

    print(
        f'Max m_opts {max(max_m_opts)} achieves when n_star = ',
        [n_star_list[idx] for idx in max_m_opts_indices],
    )
    print(
        f'Max m_rands {max(max_m_rands)} achieves when n_star = ',
        [n_star_list[idx] for idx in max_m_rands_indices],
    )

    if save_all:
        mul_metrics_plots(
            metrics_dict=metrics_dict,
            val_list=n_star_list,
            val4label='n^{*}',
            label_latex=True,
            figsize=(10, 6),
            save_dir='plot',
            save_fn=save_prefix,
            viz_fig=True,
            include_max=include_max,
        )


if __name__ == "__main__":
    # get_res_from_tests(include_max=False, include_last=True)
    get_res_from_saves(include_max=False)
