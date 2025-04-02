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

repeat 1 ./run-commands.sh run-configs/setup-protoss.5v5.toml    run-configs/run-vdn.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.5v5.toml     run-configs/run-vdn.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.5v5.toml       run-configs/run-vdn.toml run-configs/setup.use-wandb.toml
repeat 0 ./run-commands.sh run-configs/setup-protoss.10v10.toml  run-configs/run-vdn.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.10v10.toml   run-configs/run-vdn.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.10v10.toml     run-configs/run-vdn.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-protoss.20v20.toml  run-configs/run-vdn.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.20v20.toml   run-configs/run-vdn.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.20v20.toml     run-configs/run-vdn.toml run-configs/setup.use-wandb.toml

repeat 1 ./run-commands.sh run-configs/setup-protoss.5v5.toml    run-configs/run-qmix.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.5v5.toml     run-configs/run-qmix.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.5v5.toml       run-configs/run-qmix.toml run-configs/setup.use-wandb.toml
repeat 0 ./run-commands.sh run-configs/setup-protoss.10v10.toml  run-configs/run-qmix.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.10v10.toml   run-configs/run-qmix.toml run-configs/setup.use-wandb.toml
repeat 0 ./run-commands.sh run-configs/setup-zerg.10v10.toml     run-configs/run-qmix.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-protoss.20v20.toml  run-configs/run-qmix.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.20v20.toml   run-configs/run-qmix.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.20v20.toml     run-configs/run-qmix.toml run-configs/setup.use-wandb.toml

repeat 1 ./run-commands.sh run-configs/setup-protoss.5v5.toml    run-configs/run-qplex.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.5v5.toml     run-configs/run-qplex.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.5v5.toml       run-configs/run-qplex.toml run-configs/setup.use-wandb.toml
repeat 0 ./run-commands.sh run-configs/setup-protoss.10v10.toml  run-configs/run-qplex.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.10v10.toml   run-configs/run-qplex.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.10v10.toml     run-configs/run-qplex.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-protoss.20v20.toml  run-configs/run-qplex.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.20v20.toml   run-configs/run-qplex.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.20v20.toml     run-configs/run-qplex.toml run-configs/setup.use-wandb.toml

repeat 1 ./run-commands.sh run-configs/setup-protoss.5v5.toml    run-configs/run-q+fix-sum-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.5v5.toml     run-configs/run-q+fix-sum-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.5v5.toml       run-configs/run-q+fix-sum-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-protoss.10v10.toml  run-configs/run-q+fix-sum-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.10v10.toml   run-configs/run-q+fix-sum-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.10v10.toml     run-configs/run-q+fix-sum-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-protoss.20v20.toml  run-configs/run-q+fix-sum-detach.toml run-configs/setup.use-wandb.toml
repeat 0 ./run-commands.sh run-configs/setup-terran.20v20.toml   run-configs/run-q+fix-sum-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.20v20.toml     run-configs/run-q+fix-sum-detach.toml run-configs/setup.use-wandb.toml

repeat 1 ./run-commands.sh run-configs/setup-protoss.5v5.toml    run-configs/run-q+fix-mono-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.5v5.toml     run-configs/run-q+fix-mono-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.5v5.toml       run-configs/run-q+fix-mono-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-protoss.10v10.toml  run-configs/run-q+fix-mono-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.10v10.toml   run-configs/run-q+fix-mono-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.10v10.toml     run-configs/run-q+fix-mono-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-protoss.20v20.toml  run-configs/run-q+fix-mono-detach.toml run-configs/setup.use-wandb.toml
repeat 0 ./run-commands.sh run-configs/setup-terran.20v20.toml   run-configs/run-q+fix-mono-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.20v20.toml     run-configs/run-q+fix-mono-detach.toml run-configs/setup.use-wandb.toml

repeat 1 ./run-commands.sh run-configs/setup-protoss.5v5.toml    run-configs/run-q+fix-lin-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.5v5.toml     run-configs/run-q+fix-lin-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.5v5.toml       run-configs/run-q+fix-lin-detach.toml run-configs/setup.use-wandb.toml
repeat -2 ./run-commands.sh run-configs/setup-protoss.10v10.toml  run-configs/run-q+fix-lin-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.10v10.toml   run-configs/run-q+fix-lin-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.10v10.toml     run-configs/run-q+fix-lin-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-protoss.20v20.toml  run-configs/run-q+fix-lin-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-terran.20v20.toml   run-configs/run-q+fix-lin-detach.toml run-configs/setup.use-wandb.toml
repeat 1 ./run-commands.sh run-configs/setup-zerg.20v20.toml     run-configs/run-q+fix-lin-detach.toml run-configs/setup.use-wandb.toml

