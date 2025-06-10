#!/bin/bash

#SBATCH --job-name=run-container 
#SBATCH --time=00:30
#SBATCH --ntasks=2
#SBATCH --nodes=1
#SBATCH -o slurm.out # File to which STDOUT will be written
#SBATCH -e slurm.err # File to which STDERR will be written

module load singularity
module load gcc-glibc dealii


# Check if the singularity image exists
if [ ! -f ~/singularity/singularity-image.sif ]; then
  echo "ERROR: Singularity image not found!" >&2
  exit 1
fi

# Create a temporary directory for any temporary files

singularity exec \
  --bind ~/output_pvtu:/container-app/build ~/singularity/singularity-image.sif \
  bash -c "cd /container-app/brain-nD/build; mpirun -n 16 ./main"

cd ~/output_pvtu
find . -type f ! -name "*.vtu" ! -name "*.pvtu" -delete