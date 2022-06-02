#!/bin/bash
#$ -S /bin/bash
#$ -pe mpi 1
#$ -cwd
#$ -o /NFSHOME/mprado/CODE/Themis/out/std_$JOB_ID.out
#$ -e /NFSHOME/mprado/CODE/Themis/out/err_$JOB_ID.out
#$ -q parallel.q


python $1 $2 $3
