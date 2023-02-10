#!/bin/bash
search_dir=./config/
for entry in "$search_dir"/*
do
  qsub launch.sh main.py $entry
done