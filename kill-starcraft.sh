#!/bin/bash

pgrep -f StarCraft | xargs --no-run-if-empty kill -9
