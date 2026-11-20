#!/bin/bash
#SBATCH --job-name=DrBenchmark
#SBATCH --ntasks=1             # Total MPI process per job
#SBATCH --ntasks-per-node=1    # Total MPI process per node per job
#SBATCH --hint=nomultithread   # 1 MPI process per phisical core (no hyperthreading)
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --array=0-863%100      # 27tasks*8models*4runs = 864 jobs overall but 100 jobs max in the queue
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=<ACCOUNT>@v100

# 1. Edit NBR_RUNS and --array to match: one job per run of each task for each model of models.txt
#    (if --array is too low, not all experiments will be run: the log of each job says so)
# 2. Edit source ~/.profile for your environment file
# 3. Edit --account for your JZ account

# run.py, models.txt, recipes/ and logs/ are looked for in the directory the job is submitted from
if [ ! -f run.py ] || [ ! -f models.txt ] || [ ! -d recipes ]; then
    echo "This script has to be submitted from the root of the DrBenchmark repository, not from $PWD" >&2
    exit 1
fi

module purge
module load pytorch-gpu/py3/1.12.1

source ~/.profile  # or your local configuration file that activates conda for example
conda activate DrBenchmark

NBR_RUNS=4

# Each job of the array does one run. run.py numbers the runs of every task for every model of
# models.txt, and only does the one that has the number of the job: neither the tasks nor the
# models are listed here. A run whose results exist is not computed again: submitting this
# script again only computes the runs that are missing.
srun python run.py --tasks all --nb-run "$NBR_RUNS" --job "$SLURM_ARRAY_TASK_ID"
