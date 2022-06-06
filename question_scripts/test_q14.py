"""
_summary_: Training with different eps_opt given best value of n* and get the results for q15
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

save_prefix = 'question14'
val_interval = 250
save_single = True
save_all = True

losses_dict = {}
rewards_dict = {}
metrics_dict = {"M_opt": {}, "M_rand": {}}

# val_list: epsilon_list
sample_number = 6
epsilon_list = []
for i in range(sample_number):
    epsilon_list.append(i / (sample_number - 1))
chosen_n_star = 10000
# for fast comparison
# epsilon_list = [epsilon_list[0], epsilon_list[-1]]
# n_star -> 10000: 18h23


def get_res_from_tests():
    dqn_players = [
        DQNPlayer(
            epsilon=0.01,
            buffer_sz=10000,
            batch_sz=64,
            explore=True,
            n_star=chosen_n_star,
            verbose=False,
        )
        for _ in range(sample_number)
    ]
    for idx, eps_opt in enumerate(epsilon_list):
        dqn_player = dqn_players[idx]
        print(f'{idx+1}/{sample_number}: eps_opt = {eps_opt}')
        expert = OptimalPlayer(epsilon=eps_opt)
        rewards, losses = dqn_player.train(
            expert, nr_episodes=20000, val_interval=val_interval
        )

        ## data collection
        metrics_dict["M_opt"].update({eps_opt: dqn_player.m_opts})
        metrics_dict["M_rand"].update({eps_opt: dqn_player.m_rands})
        losses_dict.update({eps_opt: losses})
        rewards_dict.update({eps_opt: rewards})

        ## viz
        if save_single:
            inside_prefix = save_prefix + f'_eps_opt{eps_opt}'
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
            val_list=epsilon_list,
            val4label='{\epsilon}_{opt}',
            label_latex=True,
            figsize=(10, 6),
            save_dir='plot',
            save_fn=save_prefix,
        )


def get_res_from_saves():
    max_m_opts, max_m_rands = [], []
    for idx, eps_opt in enumerate(epsilon_list):
        # print(f'{idx+1}/{sample_number}: eps_opt = {eps_opt}')
        inside_prefix = save_prefix + f'_eps_opt{eps_opt}'
        res = np.load(os.path.join('res', inside_prefix + '.npz'))
        m_opts = res['m_opts']
        m_rands = res['m_rands']
        # max_m_opts.append(max(m_opts))
        # max_m_rands.append(max(m_rands))
        max_m_opts.append(m_opts[-1])
        max_m_rands.append(m_rands[-1])
        # print(max(m_opts), max(m_rands))

        ## data collection
        metrics_dict["M_opt"].update({eps_opt: m_opts})
        metrics_dict["M_rand"].update({eps_opt: m_rands})

    max_m_opts_indices = [i for i, x in enumerate(max_m_opts) if x == max(max_m_opts)]
    max_m_rands_indices = [
        i for i, x in enumerate(max_m_rands) if x == max(max_m_rands)
    ]

    print(
        f'Max m_opts {max(max_m_opts)} achieves when epsilon = ',
        [epsilon_list[idx] for idx in max_m_opts_indices],
    )
    print(
        f'Max m_rands {max(max_m_rands)} achieves when epsilon = ',
        [epsilon_list[idx] for idx in max_m_rands_indices],
    )

    if save_all:
        mul_metrics_plots(
            metrics_dict=metrics_dict,
            val_list=epsilon_list,
            val4label='{\epsilon}_{opt}',
            label_latex=True,
            figsize=(10, 6),
            save_dir='plot',
            save_fn=save_prefix,
        )


if __name__ == "__main__":
    # get_res_from_tests()
    get_res_from_saves()
