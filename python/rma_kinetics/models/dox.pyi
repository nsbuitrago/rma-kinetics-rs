"""
Doxycycline pharmacokinetic model.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .. import Solution
    from ..solvers import Solver

class Model:
    """
    Dox PK model
    """

    def __init__(
        self,
        vehicle_intake: float = 1.875e-4,
        bioavailability: float = 0.9,
        absorption: float = 0.8,
        elimination: float = 0.2,
        brain_transport: float = 0.2,
        plasma_transport: float = 1.0,
        plasma_vd: float = 0.21,
        schedule: List[AccessPeriod] = ...,
    ) -> None: ...
    def solve(
        self, t0: float, tf: float, dt: float, init_state: State, solver: Solver
    ) -> Solution: ...

class State:
    """
    Dox PK model state.
    """

    def __init__(self, plasma_dox: float = 0.0, brain_dox: float = 0.0) -> None: ...
    @property
    def plasma_dox(self) -> float: ...
    @plasma_dox.setter
    def plasma_dox(self, value: float) -> None: ...
    @property
    def brain_dox(self) -> float: ...
    @brain_dox.setter
    def brain_dox(self, value: float) -> None: ...

class AccessPeriod:
    """
    Defines the concentration and period of access of dox food or water.
    """

    def __init__(self, dose: float, start_time: float, stop_time: float) -> None: ...
    @property
    def dose(self) -> float: ...
    @property
    def start_time(self) -> float: ...
    @property
    def stop_time(self) -> float: ...

def create_dox_schedule(
    dose: float,
    start_time: float,
    duration: float,
    repeat: Optional[int] = None,
    interval: Optional[float] = None,
) -> List[AccessPeriod]: ...
