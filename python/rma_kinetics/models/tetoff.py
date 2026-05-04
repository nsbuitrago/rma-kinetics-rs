from typing import Any, cast

from .._rma_kinetics import models as _models

_models = cast(Any, _models)

Model = _models.tetoff.Model
State = _models.tetoff.State

__all__ = ["Model", "State"]
