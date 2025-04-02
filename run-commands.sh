#!/bin/bash

logfile="run.log"

function log() {
  timestamp=$(date '+%F %T')
  echo "[$timestamp] $*" >> "$logfile"
}

source ./source-this.sh

python make-run-commands.py "$@" | while read -r command; do
  log "python $command"
  # NOTE: do not double quote
  python $command
done
