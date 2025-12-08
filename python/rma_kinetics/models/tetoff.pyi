"""
Tet-Off RMA expression model.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Solution
    from ..solvers import Solver
    from . import dox

class Model:
    """Tet-Off RMA expression model."""

    def __init__(
        self,
        rma_prod: float = 0.2,
        leaky_rma_prod: float = 0.002,
        rma_bbb_transport: float = 0.6,
        rma_deg: float = 0.007,
        tta_prod: float = 10.0,
        tta_deg: float = 1.0,
        tta_kd: float = 10.0,
        tta_cooperativity: float = 2.0,
        dox_pk_model: dox.Model = ...,
        dox_tta_kd: float = 10.0,
    ) -> None: ...
    def solve(
        self, t0: float, tf: float, dt: float, init_state: State, solver: Solver
    ) -> Solution: ...

class State:
    """
    Tet-Off model state.
    """

    def __init__(
        self,
        brain_rma: float = 0.0,
        plasma_rma: float = 0.0,
        tta: float = 0.0,
        brain_dox: float = 0.0,
        plasma_dox: float = 0.0,
    ) -> None: ...
    @property
    def brain_rma(self) -> float: ...
    @brain_rma.setter
    def brain_rma(self, value: float) -> None: ...
    @property
    def plasma_rma(self) -> float: ...
    @plasma_rma.setter
    def plasma_rma(self, value: float) -> None: ...
    @property
    def tta(self) -> float: ...
    @tta.setter
    def tta(self, value: float) -> None: ...
    @property
    def brain_dox(self) -> float: ...
    @brain_dox.setter
    def brain_dox(self, value: float) -> None: ...
    @property
    def plasma_dox(self) -> float: ...
    @plasma_dox.setter
    def plasma_dox(self, value: float) -> None: ...
