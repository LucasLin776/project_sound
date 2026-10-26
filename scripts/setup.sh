#!/usr/bin/env sh
# Conda CPU installation; requires conda to be on PATH.
set -eu
cd "$(dirname "$0")/.."
conda env create --file environment.yml
conda run --no-capture-output -n wind-bolt-audio python scripts/download_checkpoints.py
conda run --no-capture-output -n wind-bolt-audio python -m pip check
