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
from dqn_player import DQNPlayer
from dqn_utils import *
from viz_utils import reward_loss_plots, metrics_plots, mul_metrics_plots

os.makedirs('plot', exist_ok=True)
os.makedirs('res', exist_ok=True)
save_prefix = 'question13'
save_single = True
save_all = True

rewards_list = []
losses_list = []
metrics_dict = {"M_opt": {}, "M_rand": {}}

expert = OptimalPlayer(0.5)
val_interval = 250
# [1, 100, 1000, 10000, 20000, 40000]
# [1, 20000]
n_star_list = [1, 100, 1000, 10000, 20000, 40000]


def get_res_from_tests():
    for idx, n_star in enumerate(n_star_list):
        print(f'{idx+1}/{len(n_star_list)}: n_star = {n_star}')
        agent = DQNPlayer(
            epsilon=0.01,
            buffer_sz=10000,
            batch_sz=64,
            explore=True,
            n_star=n_star,
            verbose=False,
        )
        rewards, losses = agent.train(
            expert, nr_episodes=20000, val_interval=val_interval(10, 6)
        )

        ## data collection
        metrics_dict["M_opt"].update({n_star: agent.m_opts})
        metrics_dict["M_rand"].update({n_star: agent.m_rands})

        ## viz
        if save_single:
            inside_prefix = save_prefix + f'_n_star{n_star}'
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
        )


def get_res_from_saves():
    for n_star in n_star_list:
        inside_prefix = save_prefix + f'_n_star{n_star}'
        res = np.load(os.path.join('res', inside_prefix + '.npz'))
        m_opts = res['m_opts']
        m_rands = res['m_rands']
        ## data collection
        metrics_dict["M_opt"].update({n_star: m_opts})
        metrics_dict["M_rand"].update({n_star: m_rands})

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
        )


if __name__ == "__main__":
    # get_res_from_tests()
    get_res_from_saves()
