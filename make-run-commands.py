#!/usr/bin/env python
import functools
import operator
import tomllib
from argparse import ArgumentParser, Namespace

from pydantic import BaseModel, Field


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("configs", nargs="+")

    return parser.parse_args()


class RunConfig(BaseModel):
    config: str
    arguments: list[str] = Field(default_factory=list)


class Config(BaseModel):
    env_config: str
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


def make_run_command(env_config: str, config: str, arguments: list[str]) -> str:
    run_command = f"src/main.py --env-config={env_config} --config={config}"

    if arguments:
        run_command = f"{run_command} with {" ".join(arguments)}"

    return run_command


def main(args: Namespace):
    config = load_config(args.configs)

    for run in config.runs:
        arguments = config.arguments + run.arguments
        run_command = make_run_command(config.env_config, run.config, arguments)
        print(run_command)


if __name__ == "__main__":
    args = parse_args()
    main(args)
