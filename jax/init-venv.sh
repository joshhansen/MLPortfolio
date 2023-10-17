#!/bin/sh

VENV="$PWD/venv"

python -m venv $VENV

PIP="$VENV/bin/pip"

$PIP install --upgrade pip

$PIP install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

$PIP install -r other-packages.txt
