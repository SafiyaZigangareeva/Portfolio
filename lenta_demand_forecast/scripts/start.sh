#!/bin/sh

set -e

mkdir -p data
mkdir -p data/optuna
mkdir -p data/preprocessed

#python -m spacy download en_core_web_sm

exec python main.py