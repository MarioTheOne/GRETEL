#!/bin/bash
search_dir=./config/$1
for entry in "$search_dir"/*
do
  echo $entry
  qsub launch.sh main.py $entry
done