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
#SBATCH --partition=gpu
#SBATCH --constraint='GPURAM_Min_12GB'

# 1. Edit NBR_RUNS and --array to match (if --array is too low, not all experiments will be run)
# 2. Edit source ~/.profile for your environment file

source ~/.profile  # or your local configuration file that activates conda for example
conda activate DrBenchmark

NBR_RUNS=4

# Read models.txt into a bash array
# http://mywiki.wooledge.org/BashFAQ/005
declare -a MODELS
while read m ; do
    MODELS+=("$m")
done < models.txt
# If there is no newline at the end of the file $m contains the last model
test "$m" && MODELS+=("$m")

declare -a JOBS

for MODEL_NAME in "${MODELS[@]}"; do

    for ((i=0; i < $NBR_RUNS; i++)); do

        JOBS+=(
            "cd ./recipes/cas/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
            "cd ./recipes/cas/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"
            "cd ./recipes/cas/scripts/ && srun bash run_task_3.sh '$MODEL_NAME'"
            "cd ./recipes/cas/scripts/ && srun bash run_task_4.sh '$MODEL_NAME'"
            
            "cd ./recipes/clister/scripts/ && srun bash run.sh '$MODEL_NAME'"
            
            "cd ./recipes/deft2019/scripts/ && srun bash run.sh '$MODEL_NAME'"
            
            "cd ./recipes/deft2020/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
            "cd ./recipes/deft2020/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"
            
            "cd ./recipes/deft2021/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
            "cd ./recipes/deft2021/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"
            
            "cd ./recipes/diamed/scripts/ && srun bash run.sh '$MODEL_NAME'"
            
            "cd ./recipes/e3c/scripts/ && srun bash run.sh '$MODEL_NAME' 'French_clinical'"
            "cd ./recipes/e3c/scripts/ && srun bash run.sh '$MODEL_NAME' 'French_temporal'"
            
            "cd ./recipes/essai/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
            "cd ./recipes/essai/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"
            "cd ./recipes/essai/scripts/ && srun bash run_task_3.sh '$MODEL_NAME'"
            "cd ./recipes/essai/scripts/ && srun bash run_task_4.sh '$MODEL_NAME'"
            
            "cd ./recipes/frenchmedmcqa/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
            "cd ./recipes/frenchmedmcqa/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"
            
            "cd ./recipes/mantragsc/scripts/ && srun bash run.sh '$MODEL_NAME' 'fr_emea'"
            "cd ./recipes/mantragsc/scripts/ && srun bash run.sh '$MODEL_NAME' 'fr_medline'"
            "cd ./recipes/mantragsc/scripts/ && srun bash run.sh '$MODEL_NAME' 'fr_patents'"
            
            "cd ./recipes/morfitt/scripts/ && srun bash run.sh '$MODEL_NAME'"
            
            "cd ./recipes/pxcorpus/scripts/ && srun bash run_task_1.sh '$MODEL_NAME'"
            "cd ./recipes/pxcorpus/scripts/ && srun bash run_task_2.sh '$MODEL_NAME'"
            
            "cd ./recipes/quaero/scripts/ && srun bash run.sh '$MODEL_NAME' 'emea'"
            "cd ./recipes/quaero/scripts/ && srun bash run.sh '$MODEL_NAME' 'medline'"
        )
    done
done

# To list every job's command
# for com in "${JOBS[@]}"; do echo $com; done
# To list number of jobs
# echo ${#JOBS[@]};

# Select which job to run
CURRENT=$SLURM_ARRAY_TASK_ID
COMMAND=${JOBS[$CURRENT]}
echo "index: $CURRENT, value: $COMMAND"
# Run the job
eval "$COMMAND"
