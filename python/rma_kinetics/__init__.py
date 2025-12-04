"""
RMA Kinetics Python Library

Kinetic models for the released markers of activity (RMAs)
for constitutive or drug-induced reporter expression.
"""

from ._rma_kinetics import models
import sys

__all__ = ["models", "solvers"]

sys.modules[__name__ + ".models"] = models
sys.modules[__name__ + ".models.constitutive"] = models.constitutive
sys.modules[__name__ + ".models.tetoff"] = models.tetoff
sys.modules[__name__ + ".models.dox"] = models.dox
sys.modules[__name__ + ".models.cno"] = models.cno
