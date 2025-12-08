"""
Chemogenetic RMA expression model.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import Solution
    from ..solvers import Solver
    from .cno import Model as CNOModel
    from .dox import Model as DoxModel

class Model:
    """Chemogenetic RMA expression model."""

    def __init__(
        self,
        rma_prod: float = 0.2,
        leaky_rma_prod: float = 0.002,
        rma_bbb_transport: float = 0.6,
        rma_deg: float = 0.007,
        tta_prod: float = 10.0,
        leaky_tta_prod: float = 0.002,
        tta_deg: float = 1.0,
        tta_kd: float = 10.0,
        tta_cooperativity: float = 2.0,
        dox_pk_model: DoxModel = ...,
        dox_tta_kd: float = 10.0,
        cno_pk_model: CNOModel = ...,
        cno_ec50: float = 1000.0,
        clz_ec50: float = 1000.0,
        cno_cooperativity: float = 2.0,
        clz_cooperativity: float = 2.0,
        dreadd_prod: float = 0.2,
        dreadd_deg: float = 0.007,
        dreadd_ec50: float = 1000.0,
        dreadd_cooperativity: float = 2.0,
    ) -> None: ...
    def solve(
        self, t0: float, tf: float, dt: float, init_state: State, solver: Solver
    ) -> Solution: ...

class State:
    """Chemogenetic RMA expression model state."""

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
