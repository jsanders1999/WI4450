#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
## SBATCH --reservation=subnumatest
#SBATCH --time=02:00:00
#SBATCH --job-name="likwidtopt.%j"
##SBATCH --output=600Grid12threads_Flops&Bandwidth.out
#SBATCH --output=likwidtop.%j.out

module load 2022r2 likwid
likwid-topology
#likwid-bench -w M0:2GB:12 -t sum_avx512

# run CG for a n^3 Poisson problem
#export OMP_NUM_THREADS=12
#srun ./main_cg_poisson.x 600


