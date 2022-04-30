# CS-456_RL_Project

## Introduction

In this mini-project, our goal is to use Q-Learning and Deep Q-Learning to train artificial agents that can play the famous game of [Tic Tac Toe](https://en.wikipedia.org/wiki/Tic-tac-toe).

## Environment

```python
conda install -y numpy
conda install -y matplotlib

# notebook and extension
python -m pip install -y notebook
python -m pip install -y jupyter_contrib_nbextensions
jupyter contrib nbextension install --user

# CUDA 11.3
conda install -y pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge

# misc
python -m pip install tqdm

# TODO: logging, tensorboard
```

- use openai gym

```shell
python -m pip install gym
python -m pip install pygame
# Fatal Python error: (pygame parachute) Segmentation Fault
# try with import pygame
```

## TODO

- [ ] Implement Q-Learning
- [ ] Implement Deep Q-Learning
- [ ] Compare Q-Learning and Deep Q-Learning
- [ ] Miscellaneous
  - [ ] Integrate with tensorboard or other tunable visualization tools with logging module
  - [ ] Implement measurement of performance
  - [ ] Implement functions to save and load policy
  - [ ] Plot performance
  - [ ] Init with different random seeds
