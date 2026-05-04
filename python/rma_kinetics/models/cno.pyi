"""
Clozapine-N-oxide/clozapine pharmacokinetic model.
"""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .. import Solution
    from ..solvers import Solver

class Model:
    """
    CNO PK model.
    """
    def __init__(
        self,
        doses: List["Dose"] = ...,
        cno_absorption: float = 23.94,
        cno_elimination: float = 5.51e-2,
        cno_reverse_metabolism: float = 1.44,
        clz_metabolism: float = 3e-1,
        clz_elimination: float = 3.94,
        cno_brain_transport: float = 2.33,
        cno_plasma_transport: float = 71.85,
        clz_brain_transport: float = 35.61,
        clz_plasma_transport: float = 34.07,
        cno_plasma_vd: float = 3.99e-2,
        cno_brain_vd: float = 0.21,
        clz_plasma_vd: float = 0.24,
        clz_brain_vd: float = 8.87e-2,
    ) -> None: ...
    def solve(
        self, t0: float, tf: float, dt: float, init_state: State, solver: Solver
    ) -> Solution: ...

class State:
    """
    CNO model state.
    """

    def __init__(
        self,
        peritoneal_cno: float = 0.0,
        plasma_cno: float = 0.0,
        brain_cno: float = 0.0,
        plasma_clz: float = 0.0,
        brain_clz: float = 0.0,
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
    interval: Optional[float] = None,
) -> List[Dose]: ...
