# CS-456_RL_Project

## Introduction

In this mini-project, our goal is to use Q-Learning and Deep Q-Learning to train artificial agents that can play the famous game of [Tic Tac Toe](https://en.wikipedia.org/wiki/Tic-tac-toe).

## Deadline

**Submit by June 6 (before 23:30) and give the fraud detection interview on June 9 or June 10.**

## Environment

```python
conda create -n ann python=3.7
conda activate ann
conda install -y numpy
conda install -y matplotlib

# notebook and extension
python -m pip install notebook
conda install -c conda-forge ipywidgets -y
python -m pip install jupyter_contrib_nbextensions
pip install nbconvert==6.4.3
# jupyter contrib nbextension install --user

# CUDA 11.3 with torch
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
  - [x] Implement Basic pipeline
  - [ ] Modify different arch with dict
  - [ ] Implement functions to save and load policy
- [ ] Compare Q-Learning and Deep Q-Learning
- [ ] Miscellaneous
  - [ ] Integrate with tensorboard or other tunable visualization tools with logging module
  - [ ] Implement measurement of performance
  - [ ] Init with different random seeds
