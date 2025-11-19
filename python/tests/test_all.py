#from pytest import assert_equal
from rma_kinetics import models, solvers

T0 = 0
T1 = 168
DT = 1

solver = solvers.Dopri5()

def test_constitutive_model_creation():
    models.constitutive.Model() # default model
    models.constitutive.Model(0.4, 0.5, 0.005) # custom rates

def test_constitutive_state_creation():
    state = models.constitutive.State() # default state
    assert state.brain_rma == 0 and state.plasma_rma == 0

    # updating state
    state.brain_rma = 10
    assert state.brain_rma == 10

    # custom state
    custom_state = models.constitutive.State(brain_rma=20, plasma_rma=10)
    assert custom_state.brain_rma == 20 and custom_state.plasma_rma == 10

def test_constitutive_solve():
    model = models.constitutive.Model()
    state = models.constitutive.State()
    solver = solvers.Dopri5()

    solution = model.solve(T0, T1, DT, state, solver)
    expected_shape = (T1+1,)
    assert solution.ts.shape == expected_shape

    plasma_rma = solution.plasma_rma
    brain_rma = solution.brain_rma
    assert plasma_rma.shape == expected_shape
    assert brain_rma.shape == expected_shape
    assert plasma_rma[-1] > brain_rma[-1]
