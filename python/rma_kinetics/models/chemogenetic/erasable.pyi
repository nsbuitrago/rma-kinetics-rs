from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ... import Solution
    from ...solvers import Solver
    from ..cno import Model as CNOModel
    from ..dox import Model as DoxModel

class TevDose:
    def __init__(self, nmol: float, time: float) -> None: ...
    @property
    def nmol(self) -> float: ...
    @nmol.setter
    def nmol(self, value: float) -> None: ...
    @property
    def time(self) -> float: ...
    @time.setter
    def time(self, value: float) -> None: ...

class Model:
    def __init__(
        self,
        rma_prod: float = 0.428,
        leaky_rma_prod: float = 7.01e-3,
        rma_bbb_transport: float = 0.727,
        rma_deg: float = 5.5e-3,
        tta_prod: float = 12.46,
        leaky_tta_prod: float = 1.22e-1,
        tta_deg: float = 2.81e-2,
        tta_kd: float = 4.19,
        tta_cooperativity: float = 2.0,
        dox_pk_model: DoxModel = ...,
        dox_tta_kd: float = 5.27,
        cno_pk_model: CNOModel = ...,
        cno_ec50: float = 7.94,
        clz_ec50: float = 4.34,
        cno_cooperativity: float = 1.0,
        clz_cooperativity: float = 1.0,
        dreadd_prod: float = 8.05,
        dreadd_deg: float = 1.0,
        dreadd_ec50: float = 6.79,
        dreadd_cooperativity: float = 1.0,
        tev_doses: list[TevDose] = ...,
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
        tta: float = 0.0,
        plasma_dox: float = 0.0,
        brain_dox: float = 0.0,
        dreadd: float = 0.0,
        peritoneal_cno: float = 0.0,
        plasma_cno: float = 0.0,
        brain_cno: float = 0.0,
        plasma_clz: float = 0.0,
        brain_clz: float = 0.0,
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
    def tta(self) -> float: ...
    @tta.setter
    def tta(self, value: float) -> None: ...
    @property
    def plasma_dox(self) -> float: ...
    @plasma_dox.setter
    def plasma_dox(self, value: float) -> None: ...
    @property
    def brain_dox(self) -> float: ...
    @brain_dox.setter
    def brain_dox(self, value: float) -> None: ...
    @property
    def dreadd(self) -> float: ...
    @dreadd.setter
    def dreadd(self, value: float) -> None: ...
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
    @property
    def plasma_tev(self) -> float: ...
    @plasma_tev.setter
    def plasma_tev(self, value: float) -> None: ...

def create_tev_schedule(
    nmol: float,
    start_time: float,
    repeat: Optional[int] = None,
    interval: Optional[float] = None,
) -> list[TevDose]: ...
