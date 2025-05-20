import copy
import os
from typing import Dict

import numpy as np

from src.utils.Filesys import search_file_list
from src.EA.NSGA_sol import NSGAII_sol

NSGA_opts = {
    "min": -4,
    "max": 4,
    "num_parents": 16,
    "num_generations": 100,
    "mutation_prob": 0.3,
    "crossover_prob": 0.1,
}

# Create an alias for NSGAII class so it can be imported as NSGAII
class NSGAII(NSGAII_sol):
    """Alias for NSGAII_sol class for backward compatibility"""
    pass


