#!/bin/bash

config_file=./config/steel/gcountergan-tc28/fold_0.json

qsub launch.sh hyperparams_opt_main.py $config_file 0