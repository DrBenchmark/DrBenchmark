"""Code shared by the recipes.

The training scripts of the recipes only state what is specific to their task, and call
the module of their kind of task. They add `recipes/` to the search path of Python to
import this package. There is no symbolic link to it in the recipes: a link may be lost
on Windows, or when the repository is extracted from a zip archive.

- `utils`: command line and names of the output files
- `training`: steps shared by every kind of task
- `token_classification`: part-of-speech tagging and named-entity recognition
- `sequence_classification`: multi-class classification
- `multi_label_classification`: multi-label classification
- `regression`: semantic textual similarity
"""
