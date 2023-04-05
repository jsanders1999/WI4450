#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
## SBATCH --reservation=subnumatest
#SBATCH --time=02:00:00
#SBATCH --job-name="likwidbenc_daxpy.%j"
##SBATCH --output=600Grid12threads_Flops&Bandwidth.out
#SBATCH --output=likwidbenc_daxpy.%j.out

module load 2022r2 likwid
likwid-bench -w M0:2GB:1 -t daxpy_mem_avx512_fma
likwid-bench -w M0:2GB:2 -t daxpy_mem_avx512_fma
likwid-bench -w M0:2GB:4 -t daxpy_mem_avx512_fma
likwid-bench -w M0:2GB:6 -t daxpy_mem_avx512_fma
likwid-bench -w M0:2GB:8 -t daxpy_mem_avx512_fma
likwid-bench -w M0:2GB:10 -t daxpy_mem_avx512_fma
likwid-bench -w M0:2GB:12 -t daxpy_mem_avx512_fma
#likwid-bench -w M0:2GB:12 -t daxpy_mem_avx512_fma

# run CG for a n^3 Poisson problem
#export OMP_NUM_THREADS=12
#srun ./main_cg_poisson.x 600


