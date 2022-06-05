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
save_prefix = 'question14'
val_interval = 250
save_single = True
save_all = True

# TODO: record other params!!!
rewards_list = []
losses_list = []
metrics_dict = {"M_opt": {}, "M_rand": {}}


# val_list: epsilon_list
sample_number = 6
epsilon_list = []
for i in range(sample_number):
    epsilon_list.append(i / (sample_number - 1))
# for fast comparison
# epsilon_list = [epsilon_list[0], epsilon_list[-1]]
# n_star -> 10000: 18h23
chosen_n_star = 1000


def get_res_from_tests():
    for idx, eps_opt in enumerate(epsilon_list):
        print(f'{idx+1}/{sample_number}: eps_opt = {eps_opt}')
        agent = DQNPlayer(
            epsilon=0.01,
            buffer_sz=10000,
            batch_sz=64,
            explore=True,
            n_star=chosen_n_star,  # TODO: res from q13
            verbose=False,
        )
        expert = OptimalPlayer(epsilon=eps_opt)
        rewards, losses = agent.train(
            expert, nr_episodes=20000, val_interval=val_interval, self_practice=False
        )

        ## data collection
        metrics_dict["M_opt"].update({eps_opt: agent.m_opts})
        metrics_dict["M_rand"].update({eps_opt: agent.m_rands})

        ## viz
        if save_single:
            inside_prefix = save_prefix + f'_eps_opt{eps_opt}'
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
            val_list=epsilon_list,
            val4label='{\epsilon}_{opt}',
            label_latex=True,
            figsize=(10, 6),
            save_dir='plot',
            save_fn=save_prefix,
        )


def get_res_from_saves():

    for eps_opt in epsilon_list:
        inside_prefix = save_prefix + f'_eps_opt{eps_opt}'
        res = np.load(os.path.join('res', inside_prefix + '.npz'))
        m_opts = res['m_opts']
        m_rands = res['m_rands']
        ## data collection
        metrics_dict["M_opt"].update({eps_opt: m_opts})
        metrics_dict["M_rand"].update({eps_opt: m_rands})

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
    get_res_from_tests()
    # get_res_from_saves()
