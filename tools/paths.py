"""Files and directories of the repository that the tools read and write.

They are found from the location of this file: the tools can be run from any directory.
"""

import os

# This file is in `tools/`, at the root of the repository
REPO_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# The results of the runs are in `recipes/<recipe>/runs/`
RECIPES_DIR = os.path.join(REPO_DIR, "recipes")
# The tools write their outputs in `stats/`
STATS_DIR = os.path.join(REPO_DIR, "stats")
# The models to evaluate, one per line
MODELS_FILE = os.path.join(REPO_DIR, "models.txt")
