#!/bin/bash

#SBATCH --exclusive
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=0
#SBATCH --account=Education-EEMCS-Courses-WI4450
## SBATCH --reservation=subnumatest
#SBATCH --time=24:00:00
#SBATCH --job-name="Correct_Blocking_600_16th.%j"
#SBATCH --output=Correct_Blocking_600_16th.%j.out


export OMP_PROC_BIND=close
export OMP_PLACES=cores

# run CG for a n^3 Poisson problem
export OMP_NUM_THREADS=16
srun ./main_cg_poisson.x 600

export OMP_NUM_THREADS=16
srun ./main_cg_block.x 600 600

export OMP_NUM_THREADS=16
srun ./main_cg_block.x 600 150 

export OMP_NUM_THREADS=16
srun ./main_cg_block.x 600 144 

export OMP_NUM_THREADS=16
srun ./main_cg_block.x 600 72

export OMP_NUM_THREADS=16
srun ./main_cg_block.x 600 7

export OMP_NUM_THREADS=16
srun ./main_cg_block.x 600 6

export OMP_NUM_THREADS=16
srun ./main_cg_block.x 600 5

export OMP_NUM_THREADS=16
srun ./main_cg_block.x 600 4



