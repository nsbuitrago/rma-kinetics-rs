"""
Clozapine-N-oxide/clozapine pharmacokinetic model.
"""

from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..solvers import Solver
    from .. import Solution

class Model:
    """
    CNO PK model.

    Arguments
    ---------
    doses: List[Dose] (Optional, Default: [Dose(mg=0.03, time=0.0)])
    cno_absorption: float (Optional, Default: 23.94 nM/hr)
    cno_elimination: float (Optional, Default: 0.0 nM/hr)
    cno_reverse_metabolism: float (Optional, Default: 0.0 nM/hr)
    clz_metabolism: float (Optional, Default: 0.0 nM/hr)
    clz_elimination: float (Optional, Default: 0.0 nM/hr)
    cno_brain_transport: float (Optional, Default: 0.0 nM/hr)
    cno_plasma_transport: float (Optional, Default: 0.0 nM/hr)
    clz_brain_transport: float (Optional, Default: 0.0 nM/hr)
    clz_plasma_transport: float (Optional, Default: 0.0 nM/hr)
    cno_plasma_vd: float (Optional, Default: 0.0 L/kg)
    cno_brain_vd: float (Optional, Default: 0.0 L/kg)
    clz_plasma_vd: float (Optional, Default: 0.0 L/kg)
    clz_brain_vd: float (Optional, Default: 0.0 L/kg)
    """
    def __init__(
        self,
        doses: List["Dose"] = ...,
        cno_absorption: float = 23.94,
        cno_elimination: float = 0.0,
        cno_reverse_metabolism: float = 0.0,
        clz_metabolism: float = 0.0,
        clz_elimination: float = 0.0,
        cno_brain_transport: float = 0.0,
        cno_plasma_transport: float = 0.0,
        clz_brain_transport: float = 0.0,
        clz_plasma_transport: float = 0.0,
        cno_plasma_vd: float = 0.0,
        cno_brain_vd: float = 0.0,
        clz_plasma_vd: float = 0.0,
        clz_brain_vd: float = 0.0
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
    CNO model state.

    Arguments
    ---------
    peritoneal_cno: float (Optional, Default: 0.0 nmol)
    plasma_cno: float (Optional, Default: 0.0 nM)
    brain_cno: float (Optional, Default: 0.0 nM)
    plasma_clz: float (Optional, Default: 0.0 nM)
    brain_clz: float (Optional, Default: 0.0 nM)
    """
    
    def __init__(
        self,
        peritoneal_cno: float = 0.0,
        plasma_cno: float = 0.0,
        brain_cno: float = 0.0,
        plasma_clz: float = 0.0,
        brain_clz: float = 0.0
    ) -> None: ...
    
    @property
    def peritoneal_cno(self) -> float: ...
    
    @peritoneal_cno.setter
    def peritoneal_cno(self, value: float) -> None: ...
    
    @property
    def plasma_cno(self) -> float: ...
    
    @plasma_cno.setter
    def plasma_cno(self, value: float) -> None: ...
    
    @property
    def brain_cno(self) -> float: ...
    
    @brain_cno.setter
    def brain_cno(self, value: float) -> None: ...
    
    @property
    def plasma_clz(self) -> float: ...
    
    @plasma_clz.setter
    def plasma_clz(self, value: float) -> None: ...
    
    @property
    def brain_clz(self) -> float: ...
    
    @brain_clz.setter
    def brain_clz(self, value: float) -> None: ...

class Dose:
    """Defines a CNO dose given an amount in mg and administration time."""
    
    def __init__(self, mg: float, time: float) -> None: ...
    
    @property
    def mg(self) -> float: ...
    
    @mg.setter
    def mg(self, value: float) -> None: ...
    
    @property
    def nmol(self) -> float: ...
    
    @nmol.setter
    def nmol(self, value: float) -> None: ...
    
    @property
    def time(self) -> float: ...
    
    @time.setter
    def time(self, value: float) -> None: ...

def create_cno_schedule(
    mg: float,
    start_time: float,
    repeat: Optional[int] = None,
    interval: Optional[float] = None
) -> List[Dose]: ...

