"""
Constitutive RMA expression model

Contains `Model` class for the constitutive RMA expression model
and `State` class for the model state.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Solution
    from ..solvers import Solver

class Model:
    """Constitutive RMA expression model"""
    def __init__(
        self, prod: float = 0.2, bbb_transport: float = 0.6, deg: float = 0.007
    ) -> None: ...
    def solve(
        self, t0: float, tf: float, dt: float, init_state: State, solver: Solver
    ) -> Solution: ...
    """
    Solve model from an initial state over the given time interval.
    """

class State:
    def __init__(self, brain_rma: float = 0.0, plasma_rma: float = 0.0) -> None: ...
    @property
    def brain_rma(self) -> float: ...
    @brain_rma.setter
    def brain_rma(self, value: float) -> None: ...
    @property
    def plasma_rma(self) -> float: ...
    @plasma_rma.setter
    def plasma_rma(self, value: float) -> None: ...
