#!/bin/bash

#SBATCH --job-name=ntm_battery
#SBATCH --nodes=1
#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=10GB
#SBATCH --time=2-00:00:00
#SBATCH --output=run_MD.out.%j
#SBATCH --error=run_MD.err.%j

module purge

source /scratch/work/public/singularity/greene-ib-slurm-bind.sh

/opt/slurm/bin/srun \
  singularity exec \
  --bind /share/apps \
  --overlay /scratch/ph2484/tuckerman-conda/tuckerman-conda.ext3:ro \
  /scratch/work/public/singularity/ubuntu-22.04.3.sif \
  /bin/bash -c "
  source /opt/apps/lmod/lmod/init/bash;
  export MODULEPATH=/share/apps/modulefiles;
  module load fftw/openmpi/intel/3.3.9;
  source /ext3/env.sh;
  lmp -in run_MD.lmp -screen screen.log
"
