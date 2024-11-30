#!/bin/bash

source ./source-this.sh

maps_path="$SC2PATH/Maps/SMAC_Maps"
mkdir -p "$maps_path"

echo "installing SMAC maps"
(
  cd "$maps_path" || exit
  wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
  unzip SMAC_Maps.zip
  rm -rf SMAC_Maps.zip
)
