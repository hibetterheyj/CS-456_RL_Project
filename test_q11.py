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
from viz_utils import reward_loss_plots

os.makedirs('plot', exist_ok=True)
os.makedirs('res', exist_ok=True)

agent = DQNPlayer(epsilon=0.01)
expert = OptimalPlayer(0.5)
rewards, losses = agent.train(
    expert, nr_episodes=20000, val_interval=250, self_practice=False
)

reward_loss_plots(rewards, losses, save_dir='plot', save_fn='question11.pdf')
