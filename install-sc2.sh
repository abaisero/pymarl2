#!/bin/bash

source ./source-this.sh

if [ ! -d "$SC2PATH" ]; then
  parent_path=$(dirname "$SC2PATH")
  mkdir -p "$parent_path"

  echo "installing StarCraftII 2.4.10"
  (
    cd "$parent_path" || exit
    wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
    unzip -P iagreetotheeula SC2.4.10.zip
    rm -rf SC2.4.10.zip
  )
fi

./install-sc2-maps.sh
