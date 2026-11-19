"""Code shared by the recipes.

Each recipe has a symbolic link `scripts/common` to this package: its training scripts
only state what is specific to their task, and call the module of their kind of task.

- `utils`: command line and names of the output files
- `training`: steps shared by every kind of task
- `token_classification`: part-of-speech tagging and named-entity recognition
- `sequence_classification`: multi-class classification
- `multi_label_classification`: multi-label classification
- `regression`: semantic textual similarity
"""
