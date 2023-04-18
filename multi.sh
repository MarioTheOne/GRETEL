#!/bin/bash

search_dir=./config/steel/countergan-tc28

for i in {1..1}
do
    for entry in "$search_dir"/*
    do
        qsub launch.sh main.py $entry $i
    done
done