# from pytest import assert_equal
from rma_kinetics import models, solvers

T0 = 0
T1 = 168
DT = 1

dopri5 = solvers.Dopri5()
kvaerno3 = solvers.Kvaerno3()


def test_constitutive_model_creation():
    models.constitutive.Model()  # default model
    models.constitutive.Model(0.4, 0.5, 0.005)  # custom rates


def test_constitutive_state_creation():
    state = models.constitutive.State()  # default state
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

    solution = model.solve(T0, T1, DT, state, dopri5)
    expected_shape = (T1 + 1,)
    assert solution.ts.shape == expected_shape

    plasma_rma = solution.plasma_rma
    brain_rma = solution.brain_rma
    assert plasma_rma.shape == expected_shape
    assert brain_rma.shape == expected_shape
    assert plasma_rma[-1] > brain_rma[-1]

    # test other solvers
    solution = model.solve(T0, T1, DT, state, kvaerno3)
    assert solution.ts.shape == expected_shape
    assert solution.plasma_rma.shape == expected_shape
    assert solution.brain_rma.shape == expected_shape
    assert solution.plasma_rma[-1] > solution.brain_rma[-1]


def test_dox_model_creation():
    models.dox.Model()  # default model
    models.dox.Model(bioavailability=0.87)


def test_dox_state_creation():
    state = models.dox.State()  # default state
    assert state.plasma_dox == 0 and state.brain_dox == 0

    # custom state
    custom_state = models.dox.State(plasma_dox=10, brain_dox=20)
    assert custom_state.plasma_dox == 10 and custom_state.brain_dox == 20


def test_dox_schedule_creation():
    schedule = models.dox.create_dox_schedule(40.0, 0, 24)  # single period
    assert len(schedule) == 1

    repeated_schedule = models.dox.create_dox_schedule(40.0, 0, 24, repeat=1)
    assert len(repeated_schedule) == 2
    assert repeated_schedule[0].dose == 40.0
    assert repeated_schedule[0].start_time == 0
    assert repeated_schedule[0].stop_time == 24
    assert repeated_schedule[1].dose == 40.0
    assert repeated_schedule[1].start_time == 24
    assert repeated_schedule[1].stop_time == 48

    # repeated schedule with interval
    repeated_schedule_with_interval = models.dox.create_dox_schedule(
        40.0, 0, 24, repeat=1, interval=24
    )
    assert len(repeated_schedule_with_interval) == 2
    assert repeated_schedule_with_interval[0].dose == 40.0
    assert repeated_schedule_with_interval[0].start_time == 0
    assert repeated_schedule_with_interval[0].stop_time == 24
    assert repeated_schedule_with_interval[1].dose == 40.0
    assert repeated_schedule_with_interval[1].start_time == 48
    assert repeated_schedule_with_interval[1].stop_time == 72


def test_dox_model_solve():
    model = models.dox.Model()
    state = models.dox.State()

    solution = model.solve(T0, T1, DT, state, dopri5)
    expected_shape = (T1 + 1,)
    assert solution.ts.shape == expected_shape
    assert solution.plasma_dox.shape == expected_shape
    assert solution.brain_dox.shape == expected_shape
    assert solution.plasma_dox[-1] == 0
    assert solution.brain_dox[-1] == 0

    # adding dose
    period = models.dox.AccessPeriod(dose=40.0, start_time=0, stop_time=24)
    model = models.dox.Model(schedule=[period])
    solution = model.solve(T0, T1, DT, state, dopri5)
    assert solution.plasma_dox[10] > 0
    assert solution.brain_dox[10] > 0

    # test other solvers
    solution = model.solve(T0, T1, DT, state, kvaerno3)
    assert solution.ts.shape == expected_shape
    assert solution.plasma_dox.shape == expected_shape
    assert solution.brain_dox.shape == expected_shape
    assert solution.plasma_dox[10] > 0
    assert solution.brain_dox[10] > 0


def test_tetoff_state_creation():
    state = models.tetoff.State()  # default state
    assert state.brain_rma == 0
    assert state.plasma_rma == 0
    assert state.tta == 0
    assert state.brain_dox == 0
    assert state.plasma_dox == 0

    # custom state
    custom_state = models.tetoff.State(
        brain_rma=10, plasma_rma=20, tta=30, brain_dox=40, plasma_dox=50
    )
    assert custom_state.brain_rma == 10
    assert custom_state.plasma_rma == 20
    assert custom_state.tta == 30
    assert custom_state.brain_dox == 40
    assert custom_state.plasma_dox == 50


def test_tetoff_model_creation():
    models.tetoff.Model()  # default model
    models.tetoff.Model(rma_prod=0.5)  # custom model


def test_tetoff_solve():
    model = models.tetoff.Model()
    state = models.tetoff.State()

    solution = model.solve(T0, T1, DT, state, dopri5)
    expected_shape = (T1 + 1,)
    assert solution.ts.shape == expected_shape

    plasma_rma = solution.plasma_rma
    brain_rma = solution.brain_rma
    assert plasma_rma.shape == expected_shape
    assert brain_rma.shape == expected_shape
    assert plasma_rma[-1] > brain_rma[-1]

    assert solution.tta[-1] > 0

    # test other solvers
    solution = model.solve(T0, T1, DT, state, kvaerno3)
    assert solution.ts.shape == expected_shape
    assert solution.plasma_rma.shape == expected_shape
    assert solution.brain_rma.shape == expected_shape
    assert solution.plasma_rma[-1] > solution.brain_rma[-1]
    assert solution.tta[-1] > 0


def test_cno_dose_creation():
    dose = models.cno.Dose(0.03, 0)
    assert dose.mg == 0.03
    assert dose.time == 0

    dose.mg = 0.04
    assert dose.mg == 0.04

    dose.time = 1
    assert dose.time == 1

    dose_schedule = models.cno.create_cno_schedule(0.03, 0)
    assert len(dose_schedule) == 1

    repeated_dose_schedule = models.cno.create_cno_schedule(
        0.03, 0, repeat=1, interval=24
    )
    assert len(repeated_dose_schedule) == 2
    assert repeated_dose_schedule[0].mg == 0.03
    assert repeated_dose_schedule[0].time == 0
    assert repeated_dose_schedule[1].time == 24


def test_cno_state_creation():
    state = models.cno.State()
    assert state.peritoneal_cno == 0
    assert state.plasma_cno == 0
    assert state.brain_cno == 0
    assert state.plasma_clz == 0
    assert state.brain_clz == 0

    custom_state = models.cno.State(
        peritoneal_cno=10, plasma_cno=20, brain_cno=30, plasma_clz=40, brain_clz=50
    )
    assert custom_state.peritoneal_cno == 10
    assert custom_state.plasma_cno == 20
    assert custom_state.brain_cno == 30
    assert custom_state.plasma_clz == 40
    assert custom_state.brain_clz == 50

    custom_state.peritoneal_cno = 15
    assert custom_state.peritoneal_cno == 15


def test_cno_model_creation():
    dose = models.cno.Dose(0.03, 0)
    model = models.cno.Model([dose])

    assert len(model.doses) == 1

    # custom model
    custom_model = models.cno.Model([dose], cno_absorption=25)
    assert custom_model.cno_absorption == 25

    # model with schedule
    schedule = models.cno.create_cno_schedule(0.03, 0, repeat=1, interval=24)
    model_with_schedule = models.cno.Model(schedule)
    assert len(model_with_schedule.doses) == 2


def test_cno_model_simulation():
    # default model - dose (0.03 mg) applied at t=0
    cno_model = models.cno.Model()
    cno_state = models.cno.State()

    solution = cno_model.solve(T0, T1, DT, cno_state, dopri5)
    assert solution.peritoneal_cno.shape == (T1 + 1,)

    # test other solvers
    solution = cno_model.solve(T0, T1, DT, cno_state, kvaerno3)
    assert solution.peritoneal_cno.shape == (T1 + 1,)
    assert solution.plasma_cno.shape == (T1 + 1,)
    assert solution.brain_cno.shape == (T1 + 1,)
    assert solution.plasma_clz.shape == (T1 + 1,)
    assert solution.brain_clz.shape == (T1 + 1,)


def test_chemogenetic_state_creation():
    state = models.chemogenetic.State()
    assert state.brain_rma == 0
    assert state.plasma_rma == 0
    assert state.tta == 0
    assert state.plasma_dox == 0
    assert state.brain_dox == 0
    assert state.dreadd == 0
    assert state.peritoneal_cno == 0
    assert state.plasma_cno == 0
    assert state.brain_cno == 0
    assert state.plasma_clz == 0
    assert state.brain_clz == 0

    custom_state = models.chemogenetic.State(
        brain_rma=10,
        plasma_rma=20,
        tta=30,
        plasma_dox=40,
        brain_dox=50,
        dreadd=60,
        peritoneal_cno=70,
        plasma_cno=80,
        brain_cno=90,
        plasma_clz=100,
        brain_clz=110,
    )

    assert custom_state.brain_rma == 10
    assert custom_state.plasma_rma == 20
    assert custom_state.tta == 30
    assert custom_state.plasma_dox == 40 and custom_state.brain_dox == 50
    assert custom_state.dreadd == 60 and custom_state.peritoneal_cno == 70
    assert custom_state.plasma_cno == 80 and custom_state.brain_cno == 90
    assert custom_state.plasma_clz == 100 and custom_state.brain_clz == 110


def test_chemogenetic_model_creation():
    models.chemogenetic.Model()  # default model
    model = models.chemogenetic.Model(rma_prod=0.5)  # custom model

    assert model.rma_prod == 0.5


def test_chemogenetic_model_simulation():
    model = models.chemogenetic.Model()
    state = models.chemogenetic.State()

    solution = model.solve(T0, T1, DT, state, dopri5)
    assert solution.ts.shape == (T1 + 1,)
    assert solution.brain_rma.shape == (T1 + 1,)
    assert solution.plasma_rma.shape == (T1 + 1,)
    assert solution.tta.shape == (T1 + 1,)
    assert solution.plasma_dox.shape == (T1 + 1,)
    assert solution.brain_dox.shape == (T1 + 1,)
    assert solution.dreadd.shape == (T1 + 1,)
    assert solution.peritoneal_cno.shape == (T1 + 1,)


def test_oscillation_model_creation():
    models.oscillation.Model()  # default model
    models.oscillation.Model(0.4, 0.1, 0.5, 0.005)  # custom rates


def test_oscillation_state_creation():
    state = models.oscillation.State()  # default state
    assert state.brain_rma == 0 and state.plasma_rma == 0

    # updating state
    state.brain_rma = 10
    assert state.brain_rma == 10

    # custom state
    custom_state = models.oscillation.State(brain_rma=20, plasma_rma=10)
    assert custom_state.brain_rma == 20 and custom_state.plasma_rma == 10


def test_oscillation_solve():
    model = models.oscillation.Model()
    state = models.oscillation.State()

    solution = model.solve(T0, T1, DT, state, dopri5)
    expected_shape = (T1 + 1,)
    assert solution.ts.shape == expected_shape

    plasma_rma = solution.plasma_rma
    brain_rma = solution.brain_rma
    assert plasma_rma.shape == expected_shape
    assert brain_rma.shape == expected_shape
    assert plasma_rma[-1] > brain_rma[-1]

    # apply noise
    solution.apply_noise(0.1)

    # test other solvers
    solution = model.solve(T0, T1, DT, state, kvaerno3)
    assert solution.ts.shape == expected_shape
    assert solution.plasma_rma.shape == expected_shape
    assert solution.brain_rma.shape == expected_shape
    assert solution.plasma_rma[-1] > solution.brain_rma[-1]
