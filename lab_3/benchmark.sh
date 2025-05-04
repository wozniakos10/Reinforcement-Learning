#!/bin/bash
#SBATCH --job-name=mctest
#SBATCH --time=12:00:00
#SBATCH --account=plglscclass24-cpu
#SBATCH --partition=plgrid
#SBATCH -N 1


module load python
source venv/bin/activate
srun python  ./isloation_benchmark.py