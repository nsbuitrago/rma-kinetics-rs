"""
Constitutive RMA expression model

Contains `Model` class for the constitutive RMA expression model
and `State` class for the model state.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..solvers import Solver
    from .. import Solution

class Model:
    """
    Constitutive RMA expression model.

    Attributes
    ----------
    prod : float, optional
        RMA production rate (concentration/time), by default 0.2 nM/hr
    bbb_transport : float, optional
        RMA blood-brain barrier transport rate (1/time), by default 0.6 1/hr
    deg : float, optional
        RMA degradation rate (1/time), by default 0.007 1/hr

    Methods
    -------
    solve(t0: float, tf: float, dt: float, init_state: State, solver: Solver) -> Solution:
        Solve the model over the time interval [t0, tf] with step size dt.
        Returns a `Solution` object.

    Examples
    --------
    model = Model()
    init_state = State()
    solver = solvers.Dopri5()
    solution = model.solve(0, 504, 1, init_state, solver)
    """
    
    def __init__(
        self,
        prod: float = 0.2,
        bbb_transport: float = 0.6,
        deg: float = 0.007
    ) -> None: ...
    
    def solve(
        self,
        t0: float,
        tf: float,
        dt: float,
        init_state: State,
        solver: Solver
    ) -> Solution: ...


class State:
    """
    Constitutive model state.

    Attributes
    ----------
    brain_rma : float, optional
        Brain RMA concentration (concentration), by default 0.0 nM
    plasma_rma : float, optional
        Plasma RMA concentration (concentration), by default 0.0 nM
    """
    
    def __init__(
        self,
        brain_rma: float = 0.0,
        plasma_rma: float = 0.0
    ) -> None: ...
    
    @property
    def brain_rma(self) -> float: ...
    
    @brain_rma.setter
    def brain_rma(self, value: float) -> None: ...
    
    @property
    def plasma_rma(self) -> float: ...
    
    @plasma_rma.setter
    def plasma_rma(self, value: float) -> None: ...

