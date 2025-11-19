from dataclasses import dataclass, field

@dataclass
class Solver:
    solver_type: str
    rtol: float = 1e-6
    atol: float = 1e-6
    dt0: float = 0
    min_dt: float = 0
    max_dt: float = float('inf')
    max_steps: float = 10000
    max_rejected_steps: float = 100
    safety_factor: float = 0.9
    min_scale: float = 0.2
    max_scale: float = 10

@dataclass
class Dopri5(Solver):
    """
    Dormand-Prince 5(4) Explicit Runge-Kutta method.
    """
    solver_type: str = field(
          default="dopri5",
          init=False,
          repr=False,
      )
