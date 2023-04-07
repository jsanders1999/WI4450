#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
## SBATCH --reservation=subnumatest
#SBATCH --time=24:00:00
#SBATCH --job-name="Blocking_290_.%j"
#SBATCH --output=Blocking_290_.%j.out


export OMP_PROC_BIND=close
export OMP_PLACES=cores

# run CG for a n^3 Poisson problem
export OMP_NUM_THREADS=1
srun ./main_cg_poisson.x 290

export OMP_NUM_THREADS=1
srun ./main_cg_block.x 290 290

export OMP_NUM_THREADS=1
srun ./main_cg_block.x 290 144 

export OMP_NUM_THREADS=1
srun ./main_cg_block.x 290 72 

export OMP_NUM_THREADS=1
srun ./main_cg_block.x 290 36

export OMP_NUM_THREADS=1
srun ./main_cg_block.x 290 5

export OMP_NUM_THREADS=1
srun ./main_cg_block.x 290 4



