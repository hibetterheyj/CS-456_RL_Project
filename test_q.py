# std
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

q11_exp_cfg = Path('./cfg/exp/q11.yml')
exp = load_yaml(q11_exp_cfg)

player_dict = {**exp['agent'], **exp['trainer']}
agent = DQNPlayer(**player_dict)

agent = DQNPlayer(epsilon=0.01)
expert = OptimalPlayer(0.5)
history = agent.learn(expert, nr_episodes=20000, val_interval=1000, self_practice=False)

# from agent import DeepAgent

# agent = DeepAgent(epsilon=0.1)
# opponent = OptimalPlayer(0.5)
# history = agent.learn(
#     opponent, N=20000, test_phase=2000, self_practice=False, save_avg_losses=True
# )
