#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 4
#$ -cwd
#$ -o ./output/std/std_$JOB_ID.out
#$ -e ./output/err/err_$JOB_ID.out
#$ -q parallel.q


python $1 $2 $3
