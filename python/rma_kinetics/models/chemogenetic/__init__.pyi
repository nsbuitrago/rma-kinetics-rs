"""
Chemogenetic RMA expression model.
"""

from typing import TYPE_CHECKING

from . import erasable as erasable

if TYPE_CHECKING:
    import numpy as np

    from ... import Solution
    from ...solvers import Solver
    from ..cno import Model as CNOModel
    from ..dox import Model as DoxModel

class Model:
    """Chemogenetic RMA expression model."""

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

class SensitivityEngine:
    def __init__(
        self,
        mouse_id: np.ndarray,
        obs_time: np.ndarray,
        n_mice: int,
        dox_pk_model: DoxModel,
        cno_pk_model: CNOModel,
        plasma_dox_ss: float,
        brain_dox_ss: float,
        dt_sub: float = 0.25,
    ) -> None: ...
    def predict_with_jacobian(
        self,
        prod_re: np.ndarray,
        leaky_re: np.ndarray,
        rma_prod_pop: float,
        leaky_rma_prod_pop: float,
        rma_bbb_transport: float,
        rma_deg: float,
        tta_prod: float,
        tta_deg: float,
        dox_tta_kd: float,
        cno_ec50: float,
        clz_ec50: float,
        dreadd_prod: float,
        dreadd_ec50: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
