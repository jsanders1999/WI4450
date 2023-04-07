#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
## SBATCH --reservation=subnumatest
#SBATCH --time=02:00:00
#SBATCH --job-name="likwidbenc_sum_avx512.%j"
##SBATCH --output=600Grid12threads_Flops&Bandwidth.out
#SBATCH --output=likwidbenc_sum_avx512.%j.out

module load 2022r2 likwid
likwid-bench -w M0:2GB:1 -t sum_avx512
likwid-bench -w M0:2GB:2 -t sum_avx512
likwid-bench -w M0:2GB:4 -t sum_avx512
likwid-bench -w M0:2GB:6 -t sum_avx512
likwid-bench -w M0:2GB:8 -t sum_avx512
likwid-bench -w M0:2GB:10 -t sum_avx512
likwid-bench -w M0:2GB:12 -t sum_avx512
#likwid-bench -w M0:2GB:12 -t sum_avx512

# run CG for a n^3 Poisson problem
#export OMP_NUM_THREADS=12
#srun ./main_cg_poisson.x 600


