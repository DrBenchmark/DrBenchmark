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

module purge
module load pytorch-gpu/py3/1.12.1

source ~/.profile  # or your local configuration file that activates conda for example
conda activate DrBenchmark

MODEL_NAME="Dr-BERT/DrBERT-7GB"
NBR_RUNS=1

# TASK='cas-pos'
# TASK='cas-cls'
# TASK='cas-ner_neg'
# TASK='cas-ner_spec'
# TASK='clister-regr'
# TASK='diamed-cls'
# TASK='e3c-ner_clinical'
# TASK='e3c-ner_temporal'
# TASK='essai-pos'
# TASK='essai-cls'
# TASK='essai-ner_neg'
# TASK='essai-ner_spec'
TASK='frenchmedmcqa-mcqa'
# TASK='frenchmedmcqa-cls'
# TASK='mantragsc-ner_emea'
# TASK='mantragsc-ner_medline'
# TASK='mantragsc-ner_patents'
# TASK='morfitt-cls'
# TASK='deft2019-ner'
# TASK='deft2020-regr'
# TASK='deft2020-cls'
# TASK='deft2021-ner'
# TASK='deft2021-cls'
# TASK='pxcorpus-ner'
# TASK='pxcorpus-cls'
# TASK='quaero-ner_emea'
# TASK='quaero-ner_medline'

python run.py --tasks "$TASK" --models "$MODEL_NAME" --nb-run "$NBR_RUNS"
