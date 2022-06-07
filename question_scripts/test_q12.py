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
from viz_utils import (
    reward_loss_plots,
    metrics_plots,
    mul_metrics_plots,
    mul_reward_loss_plots,
)

os.makedirs('plot', exist_ok=True)
os.makedirs('res', exist_ok=True)

# setup seed for random, numpy, and torch
setup_seed(2022)

save_prefix = 'question12'
val_interval = 250
save_single = True
save_all = True

losses_dict = {}
rewards_dict = {}
metrics_dict = {"M_opt": {}, "M_rand": {}}
epsilon_list = [i / 5.0 for i in range(5)]


def single_run():
    dqn_player = DQNPlayer(epsilon=0.2, buffer_sz=1, batch_sz=1)
    expert = OptimalPlayer(0.5)
    rewards, losses = dqn_player.train(
        expert, nr_episodes=20000, val_interval=val_interval
    )

    ## viz
    reward_loss_plots(rewards, losses, save_dir='plot', save_fn=save_prefix)
    metrics_plots(
        dqn_player.m_opts,
        dqn_player.m_rands,
        val_interval=val_interval,
        save_dir='plot',
        save_fn=save_prefix,
    )

    np.savez_compressed(
        os.path.join('res', save_prefix),
        rewards=rewards,
        losses=losses,
        m_opts=dqn_player.m_opts,
        m_rands=dqn_player.m_rands,
    )
    # res = np.load(os.path.join('res', save_prefix + '.npz'))


def multi_runs(include_last):
    for idx, eps in enumerate(epsilon_list):
        print(f'{idx+1}/{len(epsilon_list)}: eps = {eps}')

        dqn_player = DQNPlayer(epsilon=eps, batch_sz=64, buffer_sz=10000, verbose=False)
        expert = OptimalPlayer(0.5)
        rewards, losses = dqn_player.train(
            expert, nr_episodes=20000, val_interval=val_interval
        )

        ## data collection
        metrics_dict["M_opt"].update({eps: dqn_player.m_opts})
        metrics_dict["M_rand"].update({eps: dqn_player.m_rands})
        losses_dict.update({eps: losses})
        rewards_dict.update({eps: rewards})

        ## viz
        if save_single:
            reward_loss_plots(rewards, losses, save_dir='plot', save_fn=save_prefix)
            metrics_plots(
                dqn_player.m_opts,
                dqn_player.m_rands,
                val_interval=val_interval,
                save_dir='plot',
                save_fn=save_prefix,
            )
            inside_prefix = save_prefix + f'_eps{eps}'
            np.savez_compressed(
                os.path.join('res', inside_prefix),
                rewards=rewards,
                losses=losses,
                m_opts=dqn_player.m_opts,
                m_rands=dqn_player.m_rands,
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
        )
        mul_reward_loss_plots(
            reward_dict=rewards_dict,
            loss_dict=losses_dict,
            val_list=epsilon_list,
            val4label='{\epsilon}',
            label_latex=True,
            figsize=(10, 6),
            save_dir='plot',
            save_fn=save_prefix,
            include_last=include_last,
        )


if __name__ == "__main__":
    # single_run()
    multi_runs(include_last=False)
