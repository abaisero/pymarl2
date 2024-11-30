#!/usr/bin/env python
import functools
import itertools as itt
import operator
import os
import subprocess
import tomllib
from argparse import ArgumentParser, Namespace
from datetime import datetime
from typing import Iterable

from pydantic import BaseModel, Field


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--logfile", default="run.log")
    parser.add_argument("configs", nargs="+")

    return parser.parse_args()


class RunConfig(BaseModel):
    config: str
    arguments: list[str] = Field(default_factory=list)


class Config(BaseModel):
    env_config: str = "sc2"
    n_runs: int = 3
    arguments: list[str] = Field(default_factory=list)
    runs: list[RunConfig] = Field(default_factory=list)


def load_tomllib(filename: str) -> dict:
    with open(filename, "rb") as f:
        return tomllib.load(f)


def load_config(filenames: list[str]) -> Config:
    configs = [load_tomllib(filename) for filename in filenames]
    return Config(**combine_configs(configs))


def combine_configs(configs: list[dict]) -> dict:
    config = functools.reduce(operator.or_, configs, {})

    arguments = (r for c in configs if (r := c.get("arguments")) is not None)
    runs = (r for c in configs if (r := c.get("runs")) is not None)

    config["arguments"] = sum(arguments, [])
    config["runs"] = sum(runs, [])
    return config


def log(filename: str, text: str):
    with open(filename, "a") as f:
        timestamp = datetime.now().strftime("[%F %T]")
        print(f"{timestamp} {text}", file=f)


def make_command(env_config: str, config: str, arguments: list[str]) -> str:
    with_arguments = "with {}".format(" ".join(arguments)) if arguments else ""
    return f"python src/main.py --env-config={env_config} --config={config} {with_arguments}"


def make_commands(config: Config) -> Iterable[str]:
    run_ids = range(config.n_runs)

    for _, run in itt.product(run_ids, config.runs):
        arguments = config.arguments + run.arguments
        yield make_command(config.env_config, run.config, arguments)


def run_command(command: str, **kwargs):
    subprocess.run(command.split(), **kwargs)


def make_env(args: Namespace) -> dict:
    env = {}

    if "SC2PATH" not in env:
        home = os.environ["HOME"]
        env["SC2PATH"] = f"{home}/programs/StarCraftII"

    if args.debug:
        print("Settig enviroment:")
        for k, v in env.items():
            print(f"- {k} = {v}")
        print()

    env = dict(os.environ, **env)

    return env


def main(args: Namespace):
    env = make_env(args)
    config = load_config(args.configs)

    for command in make_commands(config):
        if args.debug:
            command = f"echo {command}"

        log(args.logfile, command)
        run_command(command, env=env)


if __name__ == "__main__":
    args = parse_args()
    main(args)
