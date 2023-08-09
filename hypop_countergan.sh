#!/bin/bash

config_file=./config/steel/countergan-tc28/config_tree-cycles-500-28_tc-custom-oracle_countergan_fold-0.json

qsub launch.sh hypop_countergan.py $config_file 0