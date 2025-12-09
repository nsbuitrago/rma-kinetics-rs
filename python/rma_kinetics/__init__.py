"""
RMA Kinetics Python Library

Kinetic models for the released markers of activity (RMAs)
for constitutive or drug-induced reporter expression.
"""

import sys

from . import solvers
from ._rma_kinetics import Solution, models

__all__ = ["models", "Solution", "solvers"]

sys.modules[__name__ + ".models"] = models
sys.modules[__name__ + ".models.constitutive"] = models.constitutive
sys.modules[__name__ + ".models.tetoff"] = models.tetoff
sys.modules[__name__ + ".models.dox"] = models.dox
sys.modules[__name__ + ".models.cno"] = models.cno
sys.modules[__name__ + ".models.chemogenetic"] = models.chemogenetic
sys.modules[__name__ + ".models.oscillation"] = models.oscillation
