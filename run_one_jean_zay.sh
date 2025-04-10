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
# 3. Edit MODEL_NAME and 

module purge
module load pytorch-gpu/py3/1.12.1

source ~/.profile  # or your local configuration file that activates conda for example
conda activate DrBenchmark

MODEL_NAME="Dr-BERT/DrBERT-7GB"
NBR_RUNS=1

# TASK='CAS_POS'
# TASK='CAS_CLS'
# TASK='CAS_NER NEG'
# TASK='CAS_NER SPEC'
# TASK='CLISTER'
# TASK='Diamed'
# TASK='E3C_French_clinical'
# TASK='E3C_French_temporal'
# TASK='ESSAI_POS'
# TASK='ESSAI_CLS'
# TASK='ESSAI_NER NEG'
# TASK='ESSAI_NER SPEC'
TASK='FrenchMedMCQA_MCQA'
# TASK='FrenchMedMCQA_CLS'
# TASK='MantraGSC_fr_emea'
# TASK='MantraGSC_fr_medline'
# TASK='MantraGSC_fr_patents'
# TASK='Morfitt'
# TASK='DEFT2019'
# TASK='DEFT2020_REG'
# TASK='DEFT2020_CLS'
# TASK='DEFT2021_NER'
# TASK='DEFT2021_CLS'
# TASK='PXCorpus_NER'
# TASK='PXCorpus_CLS'
# TASK='QUAERO_EMEA'
# TASK='QUAERO_MEDLINE'

python run.py --tasks "$TASK" --models "$MODEL_NAME" --nb-run "$NBR_RUNS"
