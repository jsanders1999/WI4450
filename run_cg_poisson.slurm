#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
## SBATCH --reservation=subnumatest
#SBATCH --time=02:00:00
#SBATCH --job-name="cgpoisson_600_.%j"
##SBATCH --output=600Grid12threads_Flops&Bandwidth.out
#SBATCH --output=cgpoisson_600_.%j.out



# run CG for a n^3 Poisson problem
export OMP_NUM_THREADS=12
srun ./main_cg_poisson.x 600

export OMP_NUM_THREADS=10
srun ./main_cg_poisson.x 600

export OMP_NUM_THREADS=8
srun ./main_cg_poisson.x 600

export OMP_NUM_THREADS=6
srun ./main_cg_poisson.x 600

export OMP_NUM_THREADS=4
srun ./main_cg_poisson.x 600

export OMP_NUM_THREADS=2
srun ./main_cg_poisson.x 600

export OMP_NUM_THREADS=1
srun ./main_cg_poisson.x 600


