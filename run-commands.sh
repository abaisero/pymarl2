#!/bin/bash

logfile="run.log"

function log() {
  timestamp=$(date '+%F %T')
  echo "[$timestamp] $*" >> "$logfile"
}

python make-run-commands.py "$@" | while read -r command; do
  log "python $command"
  python $command
done
