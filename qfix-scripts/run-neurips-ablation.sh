#!/bin/bash

source ./source-this.sh

NUM_SEEDS=5
SEEDS=($(seq "$NUM_SEEDS"))

for _ in "${SEEDS[@]}"; do
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-protoss.5v5.toml    qfix-scripts/configs/run-q+fix-mono-detach-smaller.toml   qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-terran.5v5.toml     qfix-scripts/configs/run-q+fix-mono-detach-smaller.toml   qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-zerg.5v5.toml       qfix-scripts/configs/run-q+fix-mono-detach-smaller.toml   qfix-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-protoss.10v10.toml  qfix-scripts/configs/run-q+fix-mono-detach-smaller.toml   qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-terran.10v10.toml   qfix-scripts/configs/run-q+fix-mono-detach-smaller.toml   qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-zerg.10v10.toml     qfix-scripts/configs/run-q+fix-mono-detach-smaller.toml   qfix-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-protoss.20v20.toml  qfix-scripts/configs/run-q+fix-mono-detach-smaller.toml   qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-terran.20v20.toml   qfix-scripts/configs/run-q+fix-mono-detach-smaller.toml   qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-zerg.20v20.toml     qfix-scripts/configs/run-q+fix-mono-detach-smaller.toml   qfix-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-protoss.5v5.toml    qfix-scripts/configs/run-qmix-bigger.toml  qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-terran.5v5.toml     qfix-scripts/configs/run-qmix-bigger.toml  qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-zerg.5v5.toml       qfix-scripts/configs/run-qmix-bigger.toml  qfix-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-protoss.10v10.toml  qfix-scripts/configs/run-qmix-bigger.toml  qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-terran.10v10.toml   qfix-scripts/configs/run-qmix-bigger.toml  qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-zerg.10v10.toml     qfix-scripts/configs/run-qmix-bigger.toml  qfix-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-protoss.20v20.toml  qfix-scripts/configs/run-qmix-bigger.toml  qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-terran.20v20.toml   qfix-scripts/configs/run-qmix-bigger.toml  qfix-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh qfix-scripts/configs/setup-zerg.20v20.toml     qfix-scripts/configs/run-qmix-bigger.toml  qfix-scripts/configs/setup.use-wandb.toml
done
