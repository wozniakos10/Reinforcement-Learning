#!/bin/bash
#SBATCH --job-name=sarsa_benchmark_dw
#SBATCH --time=72:00:00
#SBATCH --account=plglscclass24-cpu
#SBATCH --partition=plgrid
#SBATCH --output=slurm-sarsa-benchmark_%a_.out
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --array=1-12


module load python/3.9
source .venv/bin/activate
srun python3.9  ./benchmark.py