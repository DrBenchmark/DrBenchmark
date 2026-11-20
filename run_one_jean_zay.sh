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
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --account=<ACCOUNT>@v100

# 1. Edit source ~/.profile for your environment file
# 2. Edit --account for your JZ account
# 3. Edit MODEL_NAME, NBR_RUNS and TASK (the task names are the ones of `python run.py --help`)
#    run.py does not compute again a run whose results exist: to add runs to a task, increase NBR_RUNS

# run.py and logs/ are looked for in the directory the job is submitted from
if [ ! -f run.py ]; then
    echo "This script has to be submitted from the root of the DrBenchmark repository, not from $PWD" >&2
    exit 1
fi

module purge
module load pytorch-gpu/py3/1.12.1

source ~/.profile  # or your local configuration file that activates conda for example
conda activate DrBenchmark

MODEL_NAME="Dr-BERT/DrBERT-7GB"
NBR_RUNS=1

# One of the names of tasks that `python run.py --help` lists
TASK='frenchmedmcqa-mcqa'

python run.py --tasks "$TASK" --models "$MODEL_NAME" --nb-run "$NBR_RUNS"
