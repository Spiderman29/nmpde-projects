#!/bin/bash

#SBATCH --job-name=brain-simulation
#SBATCH --ntasks=16
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log


module purge
module load singularity
module load gcc-glibc dealii


# Check if the singularity image exists
if [ ! -f ~/singularity/singularity-image.sif ]; then
  echo "ERROR: Singularity image not found at ~/singularity/singularity-image.sif" >&2
  exit 1
fi

mkdir -p ~/output_pvtu

# Execute simulation
singularity exec \
  --bind ~/output_pvtu:/container-app/brain-nD/build/output \
  ~/singularity/singularity-image.sif \
  bash -c "cd /container-app/brain-nD/build && mpirun -n \$SLURM_NTASKS ./main"

# Cleanup non-essential files
cd ~/output_pvtu
find . -type f \( ! -name "*.vtu" -a ! -name "*.pvtu" \) -delete