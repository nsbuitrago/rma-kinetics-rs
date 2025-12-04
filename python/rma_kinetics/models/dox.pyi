"""
Doxycycline pharmacokinetic model.
"""

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..solvers import Solver
    from .. import Solution

class Model:
    """
    Dox PK model

    Arguments
    ---------
    vehicle_intake: float (Optional, Default: 1.875e-4 mg/hr)
        Intake rate of the dox vehicle (food or water) in mass/time units.
    bioavailability: float (Optional, Default: 0.9)
        Bioavailability of the dox vehicle (food or water) in the range [0, 1].
    absorption: float (Optional, Default: 0.8 nM/hr)
        Absorption rate of dox in 1/time units.
    elimination: float (Optional, Default: 0.2 nM/hr)
        Elimination rate of dox in 1/time units.
    brain_transport: float (Optional, Default: 0.2 nM/hr)
        Brain transport rate of dox in 1/time units.
    plasma_transport: float (Optional, Default: 1.0 nM/hr)
        Plasma transport rate of dox in 1/time units.
    plasma_vd: float (Optional, Default: 0.21 L/kg)
        Volume of distribution of dox in volume/mass units.
    schedule: List[AccessPeriod] (Optional, Default: [])
        List of access periods for dox administration.
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
        schedule: List[AccessPeriod] = ...
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
    Dox PK model state.

    Arguments
    ---------
    plasma_dox: float (Optional, Default: 0.0 nM)
        Initial plasma dox concentration in concentration units.
    brain_dox: float (Optional, Default: 0.0 nM)
        Initial brain dox concentration in concentration units.
    """
    
    def __init__(
        self,
        plasma_dox: float = 0.0,
        brain_dox: float = 0.0
    ) -> None: ...
    
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

    Arguments
    ---------
    dose: float
        Dose of dox in mg.
    start_time: float
        Start time of the access period in time units.
    stop_time: float
        Stop time of the access period in time units.
    """
    
    def __init__(
        self,
        dose: float,
        start_time: float,
        stop_time: float
    ) -> None: ...
    
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
    interval: Optional[float] = None
) -> List[AccessPeriod]: ...

