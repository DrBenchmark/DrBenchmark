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
#SBATCH --array=0-863%100      # 864 jobs overall but 100 jobs max in the queue
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH -A <ACCOUNT>@v100

module purge
module load pytorch-gpu/py3/1.12.1

source ~/.bashrc  # or your local configuration file that activates conda for example
conda activate DrBenchmark

MODEL_NAME="Dr-BERT/DrBERT-7GB"

COMMAND="cd ./recipes/cas/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/cas/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/cas/scripts/ && srun bash run_task_3.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/cas/scripts/ && srun bash run_task_4.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/clister/scripts/ && srun bash run.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/deft2019/scripts/ && srun bash run.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/deft2020/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/deft2020/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/deft2021/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/deft2021/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/diamed/scripts/ && srun bash run.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/e3c/scripts/ && srun bash run.sh '$MODEL_NAME' 'French_clinical'"
# COMMAND="cd ./recipes/e3c/scripts/ && srun bash run.sh '$MODEL_NAME' 'French_temporal'"

# COMMAND="cd ./recipes/essai/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/essai/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/essai/scripts/ && srun bash run_task_3.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/essai/scripts/ && srun bash run_task_4.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/frenchmedmcqa/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/frenchmedmcqa/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/mantragsc/scripts/ && srun bash run.sh '$MODEL_NAME' 'fr_emea'"
# COMMAND="cd ./recipes/mantragsc/scripts/ && srun bash run.sh '$MODEL_NAME' 'fr_medline'"
# COMMAND="cd ./recipes/mantragsc/scripts/ && srun bash run.sh '$MODEL_NAME' 'fr_patents'"

# COMMAND="cd ./recipes/morfitt/scripts/ && srun bash run.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/pxcorpus/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
# COMMAND="cd ./recipes/pxcorpus/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"

# COMMAND="cd ./recipes/quaero/scripts/ && srun bash run.sh '$MODEL_NAME' 'emea'"
# COMMAND="cd ./recipes/quaero/scripts/ && srun bash run.sh '$MODEL_NAME' 'medline'"

nvidia-smi

eval $COMMAND
