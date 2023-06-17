#!/bin/bash
#SBATCH --job-name=DrBenchmark
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --cpus-per-task=6
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu_p2
#SBATCH --qos=qos_gpu-t3
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH -A rtl@v100

module purge
module load pytorch-gpu/py3/1.12.1

nvidia-smi

# COMMAND='cd ./recipes/quaero/scripts/ && srun bash run.sh Dr-BERT/DrBERT-7GB medline 1.0'
# COMMAND='cd ./recipes/quaero/scripts/ && srun bash run.sh Dr-BERT/DrBERT-7GB emea 1.0'

# COMMAND='cd ./recipes/mantragsc/scripts/ && srun bash run.sh Dr-BERT/DrBERT-7GB fr_emea 1.0'
# COMMAND='cd ./recipes/mantragsc/scripts/ && srun bash run.sh Dr-BERT/DrBERT-7GB fr_medline 1.0'
# COMMAND='cd ./recipes/mantragsc/scripts/ && srun bash run.sh Dr-BERT/DrBERT-7GB fr_patents 1.0'

# COMMAND='cd ./recipes/e3c/scripts/ && srun bash run.sh Dr-BERT/DrBERT-7GB French_clinical 1.0'
# COMMAND='cd ./recipes/e3c/scripts/ && srun bash run.sh Dr-BERT/DrBERT-7GB French_temporal 1.0'

# COMMAND='cd ./recipes/morfitt/scripts/ && srun bash run.sh Dr-BERT/DrBERT-7GB 1.0'

# COMMAND='cd ./recipes/clister/scripts/ && srun bash run.sh Dr-BERT/DrBERT-7GB 1.0'

# COMMAND='cd ./recipes/deft2020/scripts/ && srun bash run_task_1.sh Dr-BERT/DrBERT-7GB 1.0'
# COMMAND='cd ./recipes/deft2020/scripts/ && srun bash run_task_2.sh Dr-BERT/DrBERT-7GB 1.0'

# COMMAND='cd ./recipes/frenchmedmcqa/scripts/ && srun bash run_task_1.sh Dr-BERT/DrBERT-7GB 1.0'
# COMMAND='cd ./recipes/frenchmedmcqa/scripts/ && srun bash run_task_2.sh Dr-BERT/DrBERT-7GB 1.0'

# COMMAND='cd ./recipes/cas/scripts/ && srun bash run_task_1.sh Dr-BERT/DrBERT-7GB 1.0'
# COMMAND='cd ./recipes/cas/scripts/ && srun bash run_task_2.sh Dr-BERT/DrBERT-7GB 1.0'

COMMAND='cd ./recipes/essai/scripts/ && srun bash run_task_1.sh Dr-BERT/DrBERT-7GB 1.0'
# COMMAND='cd ./recipes/essai/scripts/ && srun bash run_task_2.sh Dr-BERT/DrBERT-7GB 1.0'



eval $COMMAND
