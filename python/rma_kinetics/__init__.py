from ._rma_kinetics import models

__all__ = ["models", "solvers"]

import sys
sys.modules[__name__ + ".models"] = models
