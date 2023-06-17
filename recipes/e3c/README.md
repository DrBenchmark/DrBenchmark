# E3C

E3C is a freely available multilingual corpus (English, French, Italian, Spanish, and Basque) of semantically annotated clinical narratives to allow for the linguistic analysis, benchmarking, and training of information extraction systems. It consists of two types of annotations: (i) clinical entities (e.g., pathologies), (ii) temporal information and factuality (e.g., events). Researchers can use the benchmark training and test splits of our corpus to develop and test their own models.

## Run Training

```bash
bash ./scripts/run_task_1.sh "Dr-BERT/DrBERT-7GB" "emea"
bash ./scripts/run_task_2.sh "Dr-BERT/DrBERT-7GB" "emea"
```
