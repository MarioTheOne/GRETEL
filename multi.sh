#!/bin/bash

search_dir=./config/ablation

for i in {1..1}
do
    for entry in "$search_dir"/*
    do
        qsub launch.sh main.py $entry $i
    done
done
