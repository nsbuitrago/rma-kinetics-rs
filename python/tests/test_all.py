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

    # test other solvers
    # solver = solvers.Kvaerno3()
    # model.solve(T0, T1, DT, state, solver)

def test_dox_model_creation():
    models.dox.Model() # default model
    models.dox.Model(bioavailability=0.87)

def test_dox_state_creation():
    state = models.dox.State() # default state
    assert state.plasma_dox == 0 and state.brain_dox == 0

    # custom state
    custom_state = models.dox.State(plasma_dox=10, brain_dox=20)
    assert custom_state.plasma_dox == 10 and custom_state.brain_dox == 20

def test_dox_schedule_creation():
    schedule = models.dox.create_dox_schedule(40., 0, 24)  # single period
    assert len(schedule) == 1

    repeated_schedule = models.dox.create_dox_schedule(40., 0, 24, repeat=1)
    assert len(repeated_schedule) == 2
    assert repeated_schedule[0].dose == 40.
    assert repeated_schedule[0].start_time == 0
    assert repeated_schedule[0].stop_time == 24
    assert repeated_schedule[1].dose == 40.
    assert repeated_schedule[1].start_time == 24
    assert repeated_schedule[1].stop_time == 48

    # repeated schedule with interval
    repeated_schedule_with_interval = models.dox.create_dox_schedule(40., 0, 24, repeat=1, interval=24)
    assert len(repeated_schedule_with_interval) == 2
    assert repeated_schedule_with_interval[0].dose == 40.
    assert repeated_schedule_with_interval[0].start_time == 0
    assert repeated_schedule_with_interval[0].stop_time == 24
    assert repeated_schedule_with_interval[1].dose == 40.
    assert repeated_schedule_with_interval[1].start_time == 48
    assert repeated_schedule_with_interval[1].stop_time == 72

def test_dox_model_solve():
    model = models.dox.Model()
    state = models.dox.State()
    solver = solvers.Dopri5()

    solution = model.solve(T0, T1, DT, state, solver)
    expected_shape = (T1+1,)
    assert solution.ts.shape == expected_shape
    assert solution.plasma_dox.shape == expected_shape
    assert solution.brain_dox.shape == expected_shape
    assert solution.plasma_dox[-1] == 0
    assert solution.brain_dox[-1] == 0

    # adding dose
    period = models.dox.AccessPeriod(dose=40., start_time=0, stop_time=24)
    model = models.dox.Model(schedule=[period])
    solution = model.solve(T0, T1, DT, state, solver)
    assert solution.plasma_dox[10] > 0
    assert solution.brain_dox[10] > 0

    # test other solvers
    # solver = solvers.Kvaerno3()
    # model.solve(T0, T1, DT, state, solver)

def test_tetoff_state_creation():
    state = models.tetoff.State() # default state
    assert state.brain_rma == 0
    assert state.plasma_rma == 0
    assert state.tta == 0
    assert state.brain_dox == 0
    assert state.plasma_dox == 0

    # custom state
    custom_state = models.tetoff.State(brain_rma=10, plasma_rma=20, tta=30, brain_dox=40, plasma_dox=50)
    assert custom_state.brain_rma == 10
    assert custom_state.plasma_rma == 20
    assert custom_state.tta == 30
    assert custom_state.brain_dox == 40
    assert custom_state.plasma_dox == 50

def test_tetoff_model_creation():
    models.tetoff.Model() # default model
    models.tetoff.Model(rma_prod=0.5) # custom model

def test_tetoff_solve():
    model = models.tetoff.Model()
    state = models.tetoff.State()
    solver = solvers.Dopri5()

    solution = model.solve(T0, T1, DT, state, solver)
    expected_shape = (T1+1,)
    assert solution.ts.shape == expected_shape

    plasma_rma = solution.plasma_rma
    brain_rma = solution.brain_rma
    assert plasma_rma.shape == expected_shape
    assert brain_rma.shape == expected_shape
    assert plasma_rma[-1] > brain_rma[-1]

    assert solution.tta[-1] > 0

    # test other solvers
    # solver = solvers.Kvaerno3()
    # model.solve(T0, T1, DT, state, solver)