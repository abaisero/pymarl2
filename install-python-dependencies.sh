#!/bin/bash

reinstall_packages=(six)

install_packages=(
  gym==0.10.8
  imageio
  matplotlib
  numpy
  probscale
  "protobuf<3.21"
  pydantic
  pygame
  pytest
  pyyaml
  sacred
  scipy
  seaborn
  snakeviz
  tensorboard-logger
  wandb
)

upgrade_packages=(
  git+https://github.com/oxwhirl/smacv2.git
)

dev_packages=(
  ipdb
  ipython
)

python -m pip install --ignore-installed "${reinstall_packages[@]}"
python -m pip install "${install_packages[@]}"
python -m pip install --upgrade "${upgrade_packages[@]}"
python -m pip install "${dev_packages[@]}"
