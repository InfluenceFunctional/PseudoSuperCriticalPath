#!/bin/bash

#SBATCH --job-name=ntm_battery
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=2-00:00:00
#SBATCH --output=run_MD.out
#SBATCH --error=run_MD.err

module purge
module load lammps/openmpi/intel/20231214

srun lmp -in run_MD.lmp -screen screen.log
