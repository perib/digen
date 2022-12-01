#!/bin/bash -l

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t UNLIMITED
#SBATCH --mem=0
#SBATCH --job-name=autosk_on_digen
#SBATCH --mail-type=FAIL,BEGIN,END

source ~/minconda3/etc/profile.d/conda.sh

conda activate autosklearn_digen_env_final

python run_tests_automl.py \
--njobs 48 \
--autosklearn \
--savepath 'AutoML_Results_HPC' \
--localcachedir 'Datasets' \
--num_runs 10 \