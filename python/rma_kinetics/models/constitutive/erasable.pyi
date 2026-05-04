from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ... import Solution
    from ...solvers import Solver

class Model:
    def __init__(
        self,
        doses: list[Dose] = ...,
        rma_prod: float = 0.2,
        rma_bbb_transport: float = 0.6,
        rma_deg: float = 0.007,
        tev_plasma_vd: float = 1.0,
        tev_deg: float = 0.1,
        tev_cut_rate: float = 0.01,
    ) -> None: ...
    def solve(
        self, t0: float, tf: float, dt: float, init_state: State, solver: Solver
    ) -> Solution: ...

class State:
    def __init__(
        self,
        brain_rma: float = 0.0,
        plasma_rma: float = 0.0,
        plasma_tev: float = 0.0,
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
    def plasma_tev(self) -> float: ...
    @plasma_tev.setter
    def plasma_tev(self, value: float) -> None: ...

class Dose:
    def __init__(self, nmol: float, time: float) -> None: ...
    @property
    def nmol(self) -> float: ...
    @nmol.setter
    def nmol(self, value: float) -> None: ...
    @property
    def time(self) -> float: ...
    @time.setter
    def time(self, value: float) -> None: ...

def create_tev_schedule(
    nmol: float,
    start_time: float,
    repeat: Optional[int] = None,
    interval: Optional[float] = None,
) -> list[Dose]: ...
