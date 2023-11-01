#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 1
#$ -cwd
#$ -o ./output/std/std_$JOB_ID.out
#$ -e ./output/err/err_$JOB_ID.out
#$ -q gpu4.q


export CUDA_VISIBLE_DEVICES=$1
echo $CUDA_VISIBLE_DEVICES

python $2 $3 $4
