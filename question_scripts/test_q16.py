"""
_summary_: In self-practice, training with different fixed eps
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
save_prefix = 'question16'
val_interval = 250
save_single = True
save_all = True

losses_dict = {}
rewards_dict = {}
metrics_dict = {"M_opt": {}, "M_rand": {}}
epsilon_list = [i / 5 for i in range(6)]


def single_run():
    agent = DQNPlayer(epsilon=0.5, verbose=True)

    rewards, losses = agent.self_train(nr_episodes=20000, val_interval=val_interval)

    ## viz
    reward_loss_plots(rewards, losses, save_dir='plot', save_fn=save_prefix)
    metrics_plots(
        agent.m_opts,
        agent.m_rands,
        val_interval=val_interval,
        save_dir='plot',
        save_fn=save_prefix,
    )

    np.savez_compressed(
        os.path.join('res', save_prefix),
        rewards=rewards,
        losses=losses,
        m_opts=agent.m_opts,
        m_rands=agent.m_rands,
    )
    # res = np.load(os.path.join('res', save_prefix + '.npz'))


def multi_runs(include_max, include_last):
    for idx, eps in enumerate(epsilon_list):
        print(f'{idx+1}/{len(epsilon_list)}: eps = {eps}')

        agent = DQNPlayer(epsilon=eps, verbose=False)

        rewards, losses = agent.self_train(nr_episodes=20000, val_interval=val_interval)

        ## data collection
        metrics_dict["M_opt"].update({eps: agent.m_opts})
        metrics_dict["M_rand"].update({eps: agent.m_rands})
        losses_dict.update({eps: losses})
        rewards_dict.update({eps: rewards})

        ## viz
        if save_single:
            inside_prefix = save_prefix + f'_eps{eps}'
            reward_loss_plots(rewards, losses, save_dir='plot', save_fn=inside_prefix)
            metrics_plots(
                agent.m_opts,
                agent.m_rands,
                val_interval=val_interval,
                save_dir='plot',
                save_fn=inside_prefix,
            )

            np.savez_compressed(
                os.path.join('res', inside_prefix),
                rewards=rewards,
                losses=losses,
                m_opts=agent.m_opts,
                m_rands=agent.m_rands,
            )

    if save_all:
        mul_metrics_plots(
            metrics_dict=metrics_dict,
            val_list=epsilon_list,
            val4label='{\epsilon}',
            label_latex=True,
            figsize=(10, 6),
            save_dir='plot',
            save_fn=save_prefix,
            include_max=include_max,
            include_last=include_last,
        )


def get_res_from_saves(include_max, include_last):
    for idx, eps in enumerate(epsilon_list):
        inside_prefix = save_prefix + f'_eps{eps}'
        res = np.load(os.path.join('res', inside_prefix + '.npz'))
        m_opts = res['m_opts']
        m_rands = res['m_rands']
        ## data collection
        metrics_dict["M_opt"].update({eps: m_opts})
        metrics_dict["M_rand"].update({eps: m_rands})

    if save_all:
        mul_metrics_plots(
            metrics_dict=metrics_dict,
            val_list=epsilon_list,
            val4label='{\epsilon}',
            label_latex=True,
            figsize=(10, 6),
            save_dir='plot',
            save_fn=save_prefix,
            include_max=include_max,
            include_last=include_last,
        )


if __name__ == "__main__":
    # single_run()
    # multi_runs(include_max=False, include_last=True)
    get_res_from_saves(include_max=False, include_last=True)
