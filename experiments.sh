#!/bin/bash
search_dir=/NFSHOME/mprado/CODE/Themis/config/set_one
for entry in "$search_dir"/*
do
  qsub launch.sh main.py $entry
done