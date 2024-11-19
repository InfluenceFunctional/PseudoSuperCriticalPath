#!/bin/bash
#SBATCH --job-name=ntm_battery
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=7-00:00:00
#SBATCH --array=0

module purge
module load lammps/openmpi/intel/20231214

cd ../
cd $SLURM_ARRAY_TASK_ID

srun lmp -in run_MD.lmp -screen screen.log
