#!/bin/bash
#SBATCH --job-name=DrBenchmark
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1    # Nombre de processus MPI par noeud
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --constraint='GPURAM_Min_12GB'
#SBATCH --time=01:00:00
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --array=0-2575%100         # 2576 travaux en tout mais 100 travaux max dans la file

# module purge
# module load pytorch-gpu/py3/1.12.1

source ~/.bashrc
source activate huggingface_39

# set -x

JOBS=()

NBR_RUNS=4

declare -a MODELS=("Dr-BERT/DrBERT-7GB" "Dr-BERT/DrBERT-4GB-CP-PubMedBERT" "camembert-base" "almanach/camemberta-base" "almanach/camembert-bio-base" "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext" "flaubert/flaubert_base_uncased")

declare -a PERCENTAGES=("1.0" "0.75" "0.50" "0.25")

for fewshot in ${PERCENTAGES[@]}; do

    for MODEL_NAME in ${MODELS[@]}; do

        for ((i=0; i < $NBR_RUNS; i++)); do

            for f in ./recipes/*; do

                if [ -d "$f" ]; then

                    for filename in "$f"/scripts/*; do
                        
                        dirPath=${filename%/*}/
                        filename=${filename##*/}

                        if [[ "$filename" == *".sh"* ]]; then
                            
                            if [[ "$dirPath" == *"quaero"* ]]; then

                                for subset in 'emea' 'medline'; do
                                    JOBS+=("cd $dirPath && srun bash $filename '$MODEL_NAME' '$subset' '$fewshot'")
                                done 

                            elif [[ "$dirPath" == *"mantragsc"* ]]; then
                            
                                for subset in 'fr_emea' 'fr_medline' 'fr_patents'; do
                                    JOBS+=("cd $dirPath && srun bash $filename '$MODEL_NAME' '$subset' '$fewshot'")
                                done

                            elif [[ "$dirPath" == *"e3c"* ]]; then
                            
                                for subset in 'French_clinical' 'French_temporal'; do
                                    JOBS+=("cd $dirPath && srun bash $filename '$MODEL_NAME' '$subset' '$fewshot'")
                                done

                            else
                                JOBS+=("cd $dirPath && srun bash $filename '$MODEL_NAME' '$fewshot'")
                            fi

                        fi
                    done
                fi
            done
        done
    done
done

# echo ${#JOBS[@]};

nvidia-smi

CURRENT=${SLURM_ARRAY_TASK_ID}
COMMAND=${JOBS[$CURRENT]}
echo "index: $CURRENT, value: ${COMMAND}"
eval "${COMMAND}"
