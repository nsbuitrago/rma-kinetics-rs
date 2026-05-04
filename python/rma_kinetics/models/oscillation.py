from typing import Any, cast

from .._rma_kinetics import models as _models

_models = cast(Any, _models)

Model = _models.oscillation.Model
State = _models.oscillation.State

__all__ = ["Model", "State"]
