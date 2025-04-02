#!/bin/bash

function repeat() {
  n=$1
  shift
  command=$1
  shift

  for _ in $(seq "$n"); do
    "$command" "$@"
  done
}

# protoss 166k
# zerg 161k
# repeat 1 ./run-commands.sh run-configs/setup-zerg.5v5.toml    run-configs/run-qmix-bigger.toml   run-configs/setup.no-wandb.toml

# protoss 341k
# zerg 331k
# repeat 1 ./run-commands.sh run-configs/setup-zerg.10v10.toml    run-configs/run-qmix-bigger.toml   run-configs/setup.no-wandb.toml

# protoss 767k
# zerg 747k
# repeat 1 ./run-commands.sh run-configs/setup-zerg.20v20.toml    run-configs/run-qmix-bigger.toml   run-configs/setup.no-wandb.toml

for _ in seq 2; do
  ./run-commands.sh run-configs/setup-zerg.5v5.toml    run-configs/run-qmix-bigger.toml   run-configs/setup.use-wandb.toml
  ./run-commands.sh run-configs/setup-zerg.10v10.toml    run-configs/run-qmix-bigger.toml   run-configs/setup.use-wandb.toml
  ./run-commands.sh run-configs/setup-zerg.20v20.toml    run-configs/run-qmix-bigger.toml   run-configs/setup.use-wandb.toml
done
