#!/bin/bash

source ./source-this.sh

NUM_SEEDS=5
SEEDS=($(seq "$NUM_SEEDS"))

for _ in "${SEEDS[@]}"; do
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.5v5.toml    run-scripts/configs/run-vdn.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.5v5.toml     run-scripts/configs/run-vdn.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.5v5.toml       run-scripts/configs/run-vdn.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.10v10.toml  run-scripts/configs/run-vdn.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.10v10.toml   run-scripts/configs/run-vdn.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.10v10.toml     run-scripts/configs/run-vdn.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.20v20.toml  run-scripts/configs/run-vdn.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.20v20.toml   run-scripts/configs/run-vdn.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.20v20.toml     run-scripts/configs/run-vdn.toml run-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.5v5.toml    run-scripts/configs/run-qmix.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.5v5.toml     run-scripts/configs/run-qmix.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.5v5.toml       run-scripts/configs/run-qmix.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.10v10.toml  run-scripts/configs/run-qmix.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.10v10.toml   run-scripts/configs/run-qmix.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.10v10.toml     run-scripts/configs/run-qmix.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.20v20.toml  run-scripts/configs/run-qmix.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.20v20.toml   run-scripts/configs/run-qmix.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.20v20.toml     run-scripts/configs/run-qmix.toml run-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.5v5.toml    run-scripts/configs/run-qplex.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.5v5.toml     run-scripts/configs/run-qplex.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.5v5.toml       run-scripts/configs/run-qplex.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.10v10.toml  run-scripts/configs/run-qplex.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.10v10.toml   run-scripts/configs/run-qplex.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.10v10.toml     run-scripts/configs/run-qplex.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.20v20.toml  run-scripts/configs/run-qplex.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.20v20.toml   run-scripts/configs/run-qplex.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.20v20.toml     run-scripts/configs/run-qplex.toml run-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.5v5.toml    run-scripts/configs/run-q+fix-sum-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.5v5.toml     run-scripts/configs/run-q+fix-sum-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.5v5.toml       run-scripts/configs/run-q+fix-sum-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.10v10.toml  run-scripts/configs/run-q+fix-sum-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.10v10.toml   run-scripts/configs/run-q+fix-sum-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.10v10.toml     run-scripts/configs/run-q+fix-sum-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.20v20.toml  run-scripts/configs/run-q+fix-sum-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.20v20.toml   run-scripts/configs/run-q+fix-sum-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.20v20.toml     run-scripts/configs/run-q+fix-sum-detach.toml run-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.5v5.toml    run-scripts/configs/run-q+fix-mono-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.5v5.toml     run-scripts/configs/run-q+fix-mono-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.5v5.toml       run-scripts/configs/run-q+fix-mono-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.10v10.toml  run-scripts/configs/run-q+fix-mono-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.10v10.toml   run-scripts/configs/run-q+fix-mono-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.10v10.toml     run-scripts/configs/run-q+fix-mono-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.20v20.toml  run-scripts/configs/run-q+fix-mono-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.20v20.toml   run-scripts/configs/run-q+fix-mono-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.20v20.toml     run-scripts/configs/run-q+fix-mono-detach.toml run-scripts/configs/setup.use-wandb.toml

  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.5v5.toml    run-scripts/configs/run-q+fix-lin-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.5v5.toml     run-scripts/configs/run-q+fix-lin-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.5v5.toml       run-scripts/configs/run-q+fix-lin-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.10v10.toml  run-scripts/configs/run-q+fix-lin-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.10v10.toml   run-scripts/configs/run-q+fix-lin-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.10v10.toml     run-scripts/configs/run-q+fix-lin-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-protoss.20v20.toml  run-scripts/configs/run-q+fix-lin-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-terran.20v20.toml   run-scripts/configs/run-q+fix-lin-detach.toml run-scripts/configs/setup.use-wandb.toml
  qfix-scripts/run-commands.sh run-scripts/configs/setup-zerg.20v20.toml     run-scripts/configs/run-q+fix-lin-detach.toml run-scripts/configs/setup.use-wandb.toml
done
