use crate::Solve;
use differential_equations::{
    derive::State as StateTrait,
    error::Error,
    ode::{ODE, ODEProblem, OrdinaryNumericalMethod},
    prelude::{ControlFlag, Interpolation, Solution},
    solout::{EvenSolout, Solout},
};

#[cfg(feature = "py")]
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pyfunction, pymethods};

#[cfg(feature = "py")]
use crate::solve::{InnerSolution, PySolution, PySolver};

const CNO_MW: f64 = 342.8; // g/mol

/// Defines a CNO dose given an amount in mg and administration time.
/// Assumes this is an instantaneous injection.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "py", pyclass)]
pub struct Dose {
    pub mg: f64,
    pub nmol: f64,
    pub time: f64,
}

impl Dose {
    /// Create a new `Dose` given an amount in mg and administration time.
    pub fn new(mg: f64, time: f64) -> Self {
        let nmol = mg / CNO_MW * 1e6;
        Self { mg, nmol, time }
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl Dose {
    /// Create a new `Dose` given an amount in mg and administration time.
    #[new]
    pub fn create(mg: f64, time: f64) -> Self {
        Self::new(mg, time)
    }

    /// Get amount in mg.
    #[getter]
    pub fn get_mg(&self) -> f64 {
        self.mg
    }

    /// Get amount in nmol.
    #[getter]
    pub fn get_nmol(&self) -> f64 {
        self.nmol
    }

    /// Get administration time.
    #[getter]
    pub fn get_time(&self) -> f64 {
        self.time
    }

    /// Set amount in mg.
    #[setter]
    pub fn set_mg(&mut self, mg: f64) -> PyResult<()> {
        self.mg = mg;
        Ok(())
    }

    /// Set amount in nmol.
    #[setter]
    pub fn set_nmol(&mut self, nmol: f64) -> PyResult<()> {
        self.nmol = nmol;
        Ok(())
    }

    /// Set administration time.
    #[setter]
    pub fn set_time(&mut self, time: f64) -> PyResult<()> {
        self.time = time;
        Ok(())
    }
}

/// Create a CNO schedule given an amount in mg, start time, number of times to repeat,
/// and the interval between administrations.
#[cfg_attr(feature = "py", pyfunction)]
#[cfg_attr(feature = "py", pyo3(signature = (mg, start_time, repeat=None, interval=None)))]
pub fn create_cno_schedule(
    mg: f64,
    start_time: f64,
    repeat: Option<usize>,
    interval: Option<f64>,
) -> Vec<Dose> {
    let mut schedule = Vec::new();
    let mut current_time = start_time;
    let interval = interval.unwrap_or(0.);

    for _ in 0..repeat.unwrap_or(0) + 1 {
        schedule.push(Dose::new(mg, current_time));
        current_time += interval;
    }
    schedule
}

/// CNO model state
#[derive(StateTrait)]
pub struct State<T> {
    pub peritoneal_cno: T,
    pub plasma_cno: T,
    pub brain_cno: T,
    pub plasma_clz: T,
    pub brain_clz: T,
}

impl State<f64> {
    /// Get a CNO model state where all concentrations are set to 0.
    pub fn zeros() -> Self {
        Self {
            peritoneal_cno: 0.,
            plasma_cno: 0.,
            brain_cno: 0.,
            plasma_clz: 0.,
            brain_clz: 0.,
        }
    }

    /// Create a new CNO model state.
    pub fn new(
        peritoneal_cno: f64,
        plasma_cno: f64,
        brain_cno: f64,
        plasma_clz: f64,
        brain_clz: f64,
    ) -> Self {
        Self {
            peritoneal_cno,
            plasma_cno,
            brain_cno,
            plasma_clz,
            brain_clz,
        }
    }
}

impl Default for State<f64> {
    /// Default CNO model state where all concentrations are set to 0.
    fn default() -> Self {
        Self::zeros()
    }
}

#[cfg(feature = "py")]
#[pyclass(name = "State")]
#[derive(Clone)]
pub struct PyState {
    pub inner: State<f64>,
}

#[cfg(feature = "py")]
#[pymethods]
impl PyState {
    #[new]
    #[pyo3(signature = (peritoneal_cno=0., plasma_cno=0., brain_cno=0., plasma_clz=0., brain_clz=0.))]
    pub fn new(
        peritoneal_cno: f64,
        plasma_cno: f64,
        brain_cno: f64,
        plasma_clz: f64,
        brain_clz: f64,
    ) -> Self {
        Self {
            inner: State::new(peritoneal_cno, plasma_cno, brain_cno, plasma_clz, brain_clz),
        }
    }

    #[getter]
    fn get_peritoneal_cno(&self) -> f64 {
        self.inner.peritoneal_cno
    }

    #[getter]
    fn get_plasma_cno(&self) -> f64 {
        self.inner.plasma_cno
    }

    #[getter]
    fn get_brain_cno(&self) -> f64 {
        self.inner.brain_cno
    }

    #[getter]
    fn get_plasma_clz(&self) -> f64 {
        self.inner.plasma_clz
    }

    #[getter]
    fn get_brain_clz(&self) -> f64 {
        self.inner.brain_clz
    }

    #[setter]
    fn set_peritoneal_cno(&mut self, value: f64) -> PyResult<()> {
        self.inner.peritoneal_cno = value;
        Ok(())
    }

    #[setter]
    fn set_plasma_cno(&mut self, value: f64) -> PyResult<()> {
        self.inner.plasma_cno = value;
        Ok(())
    }

    #[setter]
    fn set_brain_cno(&mut self, value: f64) -> PyResult<()> {
        self.inner.brain_cno = value;
        Ok(())
    }

    #[setter]
    fn set_plasma_clz(&mut self, value: f64) -> PyResult<()> {
        self.inner.plasma_clz = value;
        Ok(())
    }

    #[setter]
    fn set_brain_clz(&mut self, value: f64) -> PyResult<()> {
        self.inner.brain_clz = value;
        Ok(())
    }
}

/// Trait for types that contain CNO-related fields.
/// This enables the CNO model to use any state type that provides
/// CNO and CLZ species without manual state construction.
pub trait CNOFields {
    fn peritoneal_cno(&self) -> f64;
    fn plasma_cno(&self) -> f64;
    fn brain_cno(&self) -> f64;
    fn plasma_clz(&self) -> f64;
    fn brain_clz(&self) -> f64;
    fn peritoneal_cno_mut(&mut self) -> &mut f64;
    fn plasma_cno_mut(&mut self) -> &mut f64;
    fn brain_cno_mut(&mut self) -> &mut f64;
    fn plasma_clz_mut(&mut self) -> &mut f64;
    fn brain_clz_mut(&mut self) -> &mut f64;
}

impl CNOFields for State<f64> {
    fn peritoneal_cno(&self) -> f64 {
        self.peritoneal_cno
    }
    fn plasma_cno(&self) -> f64 {
        self.plasma_cno
    }
    fn brain_cno(&self) -> f64 {
        self.brain_cno
    }
    fn plasma_clz(&self) -> f64 {
        self.plasma_clz
    }
    fn brain_clz(&self) -> f64 {
        self.brain_clz
    }
    fn peritoneal_cno_mut(&mut self) -> &mut f64 {
        &mut self.peritoneal_cno
    }
    fn plasma_cno_mut(&mut self) -> &mut f64 {
        &mut self.plasma_cno
    }
    fn brain_cno_mut(&mut self) -> &mut f64 {
        &mut self.brain_cno
    }
    fn plasma_clz_mut(&mut self) -> &mut f64 {
        &mut self.plasma_clz
    }
    fn brain_clz_mut(&mut self) -> &mut f64 {
        &mut self.brain_clz
    }
}

const DEFAULT_DOSE: f64 = 0.03;
const DEFAULT_DOSE_TIME: f64 = 0.;
const DEFAULT_CNO_ABSORPTION: f64 = 23.94;
const DEFAULT_CNO_ELIMINATION: f64 = 5.51e-2;
const DEFAULT_CNO_REVERSE_METABOLISM: f64 = 1.44;
const DEFAULT_CLZ_METABOLISM: f64 = 3e-1;
const DEFAULT_CLZ_ELIMINATION: f64 = 3.94;
const DEFAULT_CNO_BRAIN_TRANSPORT: f64 = 2.33;
const DEFAULT_CNO_PLASMA_TRANSPORT: f64 = 71.85;
const DEFAULT_CLZ_BRAIN_TRANSPORT: f64 = 35.61;
const DEFAULT_CLZ_PLASMA_TRANSPORT: f64 = 34.07;
const DEFAULT_CNO_PLASMA_VD: f64 = 3.99e-2;
const DEFAULT_CNO_BRAIN_VD: f64 = 0.21;
const DEFAULT_CLZ_PLASMA_VD: f64 = 0.24;
const DEFAULT_CLZ_BRAIN_VD: f64 = 8.87e-2;

/// CNO PK model
#[cfg_attr(feature = "py", pyclass)]
#[derive(Debug, Clone)]
pub struct Model {
    doses: Vec<Dose>,
    cno_absorption: f64,
    cno_elimination: f64,
    cno_reverse_metabolism: f64,
    clz_metabolism: f64,
    clz_elimination: f64,
    cno_brain_transport: f64,
    cno_plasma_transport: f64,
    clz_brain_transport: f64,
    clz_plasma_transport: f64,
    cno_plasma_vd: f64,
    cno_brain_vd: f64,
    clz_plasma_vd: f64,
    clz_brain_vd: f64,
}

impl Model {
    /// Create a new CNO model builder.
    pub fn builder(doses: Vec<Dose>) -> ModelBuilder {
        ModelBuilder::new(doses)
    }

    pub fn diff_with<S: CNOFields>(&self, _t: f64, y: &S, dydt: &mut S) {
        let peritoneal_efflux = self.cno_absorption * y.peritoneal_cno();
        let brain_cno_influx = self.cno_brain_transport * y.plasma_cno();
        let brain_cno_efflux = self.cno_plasma_transport * y.brain_cno();
        let plasma_clz_influx = self.cno_reverse_metabolism * y.plasma_cno();
        let plasma_clz_efflux = self.clz_metabolism * y.plasma_clz();
        let brain_clz_influx = self.clz_brain_transport * y.plasma_clz();
        let brain_clz_efflux = self.clz_plasma_transport * y.brain_clz();

        *dydt.peritoneal_cno_mut() = -peritoneal_efflux;

        *dydt.plasma_cno_mut() =
            peritoneal_efflux - (self.cno_elimination * y.plasma_cno()) - brain_cno_influx
                + brain_cno_efflux
                - plasma_clz_influx
                + plasma_clz_efflux;

        *dydt.brain_cno_mut() = brain_cno_influx - brain_cno_efflux;

        *dydt.plasma_clz_mut() = plasma_clz_influx
            - plasma_clz_efflux
            - (self.clz_elimination * y.plasma_clz())
            - brain_clz_influx
            + brain_clz_efflux;

        *dydt.brain_clz_mut() = brain_clz_influx - brain_clz_efflux;
    }
}

impl Default for Model {
    /// Default CNO model with default parameters.
    fn default() -> Self {
        ModelBuilder::default().build()
    }
}

impl ODE<f64, State<f64>> for Model {
    fn diff(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        self.diff_with(t, y, dydt);
    }
}

impl Solve for Model {
    type State = State<f64>;

    fn solve<S>(
        &self,
        t0: f64,
        tf: f64,
        dt: f64,
        init_state: Self::State,
        solver: &mut S,
    ) -> Result<Solution<f64, Self::State>, Error<f64, Self::State>>
    where
        S: OrdinaryNumericalMethod<f64, Self::State> + Interpolation<f64, Self::State>,
    {
        // pre-apply any doses at t0 to the initial state
        let mut adjusted_init_state = init_state;
        let mut start_dose_idx = 0;
        let n_applied_doses = &self
            .doses
            .iter()
            .filter(|dose| (dose.time - t0).abs() < 1e-10)
            .map(|dose| *adjusted_init_state.peritoneal_cno_mut() += dose.nmol)
            .count();
        start_dose_idx += n_applied_doses;

        let mut dosing_solout =
            DoseApplyingSolout::new(self.doses[start_dose_idx..].to_vec(), t0, tf, dt);
        let problem = ODEProblem::new(self, t0, tf, adjusted_init_state);
        let mut solution = problem.solout(&mut dosing_solout).solve(solver)?;

        // return concentrations using given Vd (except for peritoneal compartment)
        let y = solution
            .y
            .iter()
            .map(|state| State {
                peritoneal_cno: state.peritoneal_cno(),
                plasma_cno: state.plasma_cno() / self.cno_plasma_vd,
                brain_cno: state.brain_cno() / self.cno_brain_vd,
                plasma_clz: state.plasma_clz() / self.clz_plasma_vd,
                brain_clz: state.brain_clz() / self.clz_brain_vd,
            })
            .collect::<Vec<State<f64>>>();

        solution.y = y;
        Ok(solution)
    }
}

/// Custom solout for applying CNO doses and evenly-spaced output points
pub struct DoseApplyingSolout {
    doses: Vec<Dose>,
    t0: f64,
    tf: f64,
    dt: f64,
    next_dose_index: usize,
    last_output_time: Option<f64>,
    direction: f64,
}

/// Check if two f64 values are close.
/// Here, arbitrarily used 1e-10 for some small precision tolerance.
macro_rules! is_close {
    ($a:expr, $b:expr) => {
        ($a - $b).abs() < 1e-10
    };
}

impl DoseApplyingSolout {
    pub fn new(doses: Vec<Dose>, t0: f64, tf: f64, dt: f64) -> Self {
        let direction = (tf - t0).signum();
        Self {
            doses,
            t0,
            tf,
            dt,
            next_dose_index: 0,
            last_output_time: None,
            direction,
        }
    }

    fn get_pending_dose<I: Interpolation<f64, State<f64>>>(
        &mut self,
        t_curr: f64,
        t_prev: f64,
        y_curr: &State<f64>,
        interpolator: &mut I,
    ) -> Option<(f64, State<f64>)> {
        if self.next_dose_index >= self.doses.len() {
            return None;
        }

        let dose = &self.doses[self.next_dose_index];

        if dose.time > t_prev && dose.time <= t_curr {
            let mut y_dose = if is_close!(dose.time, t_curr) {
                y_curr.clone()
            } else {
                // this is safe to unwrap since we checked that dose.time is in
                // the range [t_prev, t_curr], so we can never get a OutofBounds error.
                interpolator.interpolate(dose.time).unwrap()
            };

            // apply the dose to the peritoneal compartment
            *y_dose.peritoneal_cno_mut() += dose.nmol;
            self.next_dose_index += 1;
            Some((dose.time, y_dose))
        } else {
            None
        }
    }
}

impl Solout<f64, State<f64>> for DoseApplyingSolout {
    fn solout<I>(
        &mut self,
        t_curr: f64,
        t_prev: f64,
        y_curr: &State<f64>,
        y_prev: &State<f64>,
        interpolator: &mut I,
        solution: &mut Solution<f64, State<f64>>,
    ) -> ControlFlag<f64, State<f64>>
    where
        I: Interpolation<f64, State<f64>>,
    {
        let pending_dose = self.get_pending_dose(t_curr, t_prev, y_curr, interpolator);
        let next_output_time = match self.last_output_time {
            Some(t) => t + self.dt,
            None => t_prev,
        };

        // Check if there's a pending dose that matches the output time
        if let Some((dose_time, dosed_state)) = pending_dose {
            // If dose time matches the output time, use the dosed state for output
            if is_close!(dose_time, next_output_time) && next_output_time <= t_curr {
                solution.push(next_output_time, dosed_state.clone());
                self.last_output_time = Some(next_output_time);
                return ControlFlag::ModifyState(dose_time, dosed_state);
            }

            // also output state if the dose time is between the evenly-spaced output points
            let dose_already_output = match self.last_output_time {
                Some(last_t) => is_close!(dose_time, last_t) || dose_time < last_t,
                None => is_close!(dose_time, t_prev),
            };

            if !dose_already_output && dose_time <= t_curr {
                solution.push(dose_time, dosed_state);
            }

            return ControlFlag::ModifyState(dose_time, dosed_state);
        }
        // From EvenSolout to handle evenly-spaced output points
        // Modified to use f64 directly for T which is safe given our type constraints on other structs.
        // Determine the alignment offset (remainder when divided by dt)
        let offset = self.t0 % self.dt;

        // Tolerance for comparing time points to avoid near-duplicates from FP error
        // Scales with dt and also includes a small absolute epsilon
        let tol = self.dt.abs() * 1e-12 + f64::EPSILON * 10.0;

        // Start from the last output point if available, otherwise from t_prev
        let start_t = match self.last_output_time {
            Some(t) => t + self.dt * self.direction,
            None => {
                // First time through, we need to include t0
                if (t_prev - self.t0).abs() < f64::EPSILON {
                    solution.push(self.t0, *y_prev);
                    self.last_output_time = Some(self.t0);
                    self.t0 + self.dt * self.direction
                } else {
                    // Find the next aligned point after t_prev
                    let rem = (t_prev - offset) % self.dt;

                    if self.direction > 0.0 {
                        // For forward integration
                        if rem.abs() < f64::EPSILON {
                            t_prev
                        } else {
                            t_prev + (self.dt - rem)
                        }
                    } else {
                        // For backward integration
                        if rem.abs() < f64::EPSILON {
                            t_prev
                        } else {
                            t_prev - rem
                        }
                    }
                }
            }
        };

        let mut ti = start_t;

        // Interpolate between steps
        while (self.direction > 0.0 && ti <= t_curr) || (self.direction < 0.0 && ti >= t_curr) {
            // Only output if the point falls within the current step
            if (self.direction > 0.0 && ti >= t_prev && ti <= t_curr)
                || (self.direction < 0.0 && ti <= t_prev && ti >= t_curr)
            {
                // Skip if this ti is essentially the same as the last output
                if self
                    .last_output_time
                    .map(|t_last| (ti - t_last).abs() <= tol)
                    .unwrap_or(false)
                {
                    // Do nothing; advance to next ti
                } else {
                    let yi = interpolator.interpolate(ti).unwrap();
                    solution.push(ti, yi);
                    self.last_output_time = Some(ti);
                }
            }

            // Move to the next point
            ti += self.dt * self.direction;
        }

        // Include final point if this step reaches tf and we haven't added it yet
        if t_curr == self.tf {
            match self.last_output_time {
                Some(t_last) => {
                    if (t_last - self.tf).abs() <= tol {
                        // Replace the near-duplicate last point with the exact final time
                        let _ = solution.pop();
                        solution.push(self.tf, *y_curr);
                        self.last_output_time = Some(self.tf);
                    } else if t_last != self.tf {
                        solution.push(self.tf, *y_curr);
                        self.last_output_time = Some(self.tf);
                    }
                }
                None => {
                    solution.push(self.tf, *y_curr);
                    self.last_output_time = Some(self.tf);
                }
            }
        }

        // Continue the integration
        ControlFlag::Continue
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (doses=vec![Dose::new(DEFAULT_DOSE, DEFAULT_DOSE_TIME)], cno_absorption=DEFAULT_CNO_ABSORPTION, cno_elimination=DEFAULT_CNO_ELIMINATION, cno_reverse_metabolism=DEFAULT_CNO_REVERSE_METABOLISM, clz_metabolism=DEFAULT_CLZ_METABOLISM, clz_elimination=DEFAULT_CLZ_ELIMINATION, cno_brain_transport=DEFAULT_CNO_BRAIN_TRANSPORT, cno_plasma_transport=DEFAULT_CNO_PLASMA_TRANSPORT, clz_brain_transport=DEFAULT_CLZ_BRAIN_TRANSPORT, clz_plasma_transport=DEFAULT_CLZ_PLASMA_TRANSPORT, cno_plasma_vd=DEFAULT_CNO_PLASMA_VD, cno_brain_vd=DEFAULT_CNO_BRAIN_VD, clz_plasma_vd=DEFAULT_CLZ_PLASMA_VD, clz_brain_vd=DEFAULT_CLZ_BRAIN_VD))]
    pub fn create(
        doses: Vec<Dose>,
        cno_absorption: f64,
        cno_elimination: f64,
        cno_reverse_metabolism: f64,
        clz_metabolism: f64,
        clz_elimination: f64,
        cno_brain_transport: f64,
        cno_plasma_transport: f64,
        clz_brain_transport: f64,
        clz_plasma_transport: f64,
        cno_plasma_vd: f64,
        cno_brain_vd: f64,
        clz_plasma_vd: f64,
        clz_brain_vd: f64,
    ) -> Self {
        Self {
            doses,
            cno_absorption,
            cno_elimination,
            cno_reverse_metabolism,
            clz_metabolism,
            clz_elimination,
            cno_brain_transport,
            cno_plasma_transport,
            clz_brain_transport,
            clz_plasma_transport,
            cno_plasma_vd,
            cno_brain_vd,
            clz_plasma_vd,
            clz_brain_vd,
        }
    }

    #[pyo3(name = "solve")]
    fn py_solve(
        &self,
        t0: f64,
        tf: f64,
        dt: f64,
        init_state: PyState,
        solver: PySolver,
    ) -> PyResult<PySolution> {
        let mut internal_solver = match solver.solver_type.as_str() {
            "dopri5" => differential_equations::methods::ExplicitRungeKutta::dopri5()
                .rtol(solver.rtol)
                .atol(solver.atol)
                .h0(solver.dt0)
                .h_min(solver.min_dt)
                .h_max(solver.max_dt)
                .max_steps(solver.max_steps)
                .max_rejects(solver.max_rejected_steps)
                .safety_factor(solver.safety_factor)
                .min_scale(solver.min_scale)
                .max_scale(solver.max_scale),
            _ => panic!("Solver not supported"),
        };

        match self.solve(t0, tf, dt, init_state.inner, &mut internal_solver) {
            Ok(solution) => Ok(PySolution {
                inner: InnerSolution::CNO(solution),
            }),
            Err(e) => Err(PyValueError::new_err("Failed to solve")), // TODO: add context from _e
        }
    }

    #[getter]
    fn get_doses(&self) -> Vec<Dose> {
        self.doses.clone()
    }
    #[getter]
    fn get_cno_absorption(&self) -> f64 {
        self.cno_absorption
    }
    #[getter]
    fn get_cno_elimination(&self) -> f64 {
        self.cno_elimination
    }
    #[getter]
    fn get_cno_reverse_metabolism(&self) -> f64 {
        self.cno_reverse_metabolism
    }
    #[getter]
    fn get_clz_metabolism(&self) -> f64 {
        self.clz_metabolism
    }
    #[getter]
    fn get_clz_elimination(&self) -> f64 {
        self.clz_elimination
    }
    #[getter]
    fn get_cno_brain_transport(&self) -> f64 {
        self.cno_brain_transport
    }
    #[getter]
    fn get_cno_plasma_transport(&self) -> f64 {
        self.cno_plasma_transport
    }
    #[getter]
    fn get_clz_brain_transport(&self) -> f64 {
        self.clz_brain_transport
    }
    #[getter]
    fn get_clz_plasma_transport(&self) -> f64 {
        self.clz_plasma_transport
    }
    #[getter]
    fn get_cno_plasma_vd(&self) -> f64 {
        self.cno_plasma_vd
    }
    #[getter]
    fn get_cno_brain_vd(&self) -> f64 {
        self.cno_brain_vd
    }
    #[getter]
    fn get_clz_plasma_vd(&self) -> f64 {
        self.clz_plasma_vd
    }
    #[getter]
    fn get_clz_brain_vd(&self) -> f64 {
        self.clz_brain_vd
    }
    #[setter]
    fn set_doses(&mut self, doses: Vec<Dose>) -> PyResult<()> {
        self.doses = doses;
        Ok(())
    }
    #[setter]
    fn set_cno_absorption(&mut self, absorption: f64) -> PyResult<()> {
        self.cno_absorption = absorption;
        Ok(())
    }
    #[setter]
    fn set_cno_elimination(&mut self, elimination: f64) -> PyResult<()> {
        self.cno_elimination = elimination;
        Ok(())
    }
    #[setter]
    fn set_cno_reverse_metabolism(&mut self, metabolism: f64) -> PyResult<()> {
        self.cno_reverse_metabolism = metabolism;
        Ok(())
    }
    #[setter]
    fn set_clz_metabolism(&mut self, metabolism: f64) -> PyResult<()> {
        self.clz_metabolism = metabolism;
        Ok(())
    }
    #[setter]
    fn set_clz_elimination(&mut self, elimination: f64) -> PyResult<()> {
        self.clz_elimination = elimination;
        Ok(())
    }
    #[setter]
    fn set_cno_brain_transport(&mut self, transport: f64) -> PyResult<()> {
        self.cno_brain_transport = transport;
        Ok(())
    }
    #[setter]
    fn set_cno_plasma_transport(&mut self, transport: f64) -> PyResult<()> {
        self.cno_plasma_transport = transport;
        Ok(())
    }
    #[setter]
    fn set_clz_brain_transport(&mut self, transport: f64) -> PyResult<()> {
        self.clz_brain_transport = transport;
        Ok(())
    }
    #[setter]
    fn set_clz_plasma_transport(&mut self, transport: f64) -> PyResult<()> {
        self.clz_plasma_transport = transport;
        Ok(())
    }
    #[setter]
    fn set_cno_plasma_vd(&mut self, vd: f64) -> PyResult<()> {
        self.cno_plasma_vd = vd;
        Ok(())
    }
    #[setter]
    fn set_cno_brain_vd(&mut self, vd: f64) -> PyResult<()> {
        self.cno_brain_vd = vd;
        Ok(())
    }
    #[setter]
    fn set_clz_plasma_vd(&mut self, vd: f64) -> PyResult<()> {
        self.clz_plasma_vd = vd;
        Ok(())
    }
    #[setter]
    fn set_clz_brain_vd(&mut self, vd: f64) -> PyResult<()> {
        self.clz_brain_vd = vd;
        Ok(())
    }
}

/// CNO PK model builder
pub struct ModelBuilder {
    pub doses: Vec<Dose>,
    cno_absorption: f64,
    cno_elimination: f64,
    cno_reverse_metabolism: f64,
    clz_metabolism: f64,
    clz_elimination: f64,
    cno_brain_transport: f64,
    cno_plasma_transport: f64,
    clz_brain_transport: f64,
    clz_plasma_transport: f64,
    cno_plasma_vd: f64,
    cno_brain_vd: f64,
    clz_plasma_vd: f64,
    clz_brain_vd: f64,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            doses: vec![Dose::new(DEFAULT_DOSE, DEFAULT_DOSE_TIME)],
            cno_absorption: DEFAULT_CNO_ABSORPTION,
            cno_elimination: DEFAULT_CNO_ELIMINATION,
            cno_reverse_metabolism: DEFAULT_CNO_REVERSE_METABOLISM,
            clz_metabolism: DEFAULT_CLZ_METABOLISM,
            clz_elimination: DEFAULT_CLZ_ELIMINATION,
            cno_brain_transport: DEFAULT_CNO_BRAIN_TRANSPORT,
            cno_plasma_transport: DEFAULT_CNO_PLASMA_TRANSPORT,
            clz_brain_transport: DEFAULT_CLZ_BRAIN_TRANSPORT,
            clz_plasma_transport: DEFAULT_CLZ_PLASMA_TRANSPORT,
            cno_plasma_vd: DEFAULT_CNO_PLASMA_VD,
            cno_brain_vd: DEFAULT_CNO_BRAIN_VD,
            clz_plasma_vd: DEFAULT_CLZ_PLASMA_VD,
            clz_brain_vd: DEFAULT_CLZ_BRAIN_VD,
        }
    }
}

impl ModelBuilder {
    pub fn new(doses: Vec<Dose>) -> Self {
        Self {
            doses,
            cno_absorption: DEFAULT_CNO_ABSORPTION,
            cno_elimination: DEFAULT_CNO_ELIMINATION,
            cno_reverse_metabolism: DEFAULT_CNO_REVERSE_METABOLISM,
            clz_metabolism: DEFAULT_CLZ_METABOLISM,
            clz_elimination: DEFAULT_CLZ_ELIMINATION,
            cno_brain_transport: DEFAULT_CNO_BRAIN_TRANSPORT,
            cno_plasma_transport: DEFAULT_CNO_PLASMA_TRANSPORT,
            clz_brain_transport: DEFAULT_CLZ_BRAIN_TRANSPORT,
            clz_plasma_transport: DEFAULT_CLZ_PLASMA_TRANSPORT,
            cno_plasma_vd: DEFAULT_CNO_PLASMA_VD,
            cno_brain_vd: DEFAULT_CNO_BRAIN_VD,
            clz_plasma_vd: DEFAULT_CLZ_PLASMA_VD,
            clz_brain_vd: DEFAULT_CLZ_BRAIN_VD,
        }
    }
    /// Set the CNO absorption rate.
    pub fn cno_absorption(&mut self, absorption: f64) -> &mut Self {
        self.cno_absorption = absorption;
        self
    }

    /// Set the CNO elimination rate.
    pub fn cno_elimination(&mut self, elimination: f64) -> &mut Self {
        self.cno_elimination = elimination;
        self
    }

    /// Set the CNO reverse metabolism rate.
    pub fn cno_reverse_metabolism(&mut self, metabolism: f64) -> &mut Self {
        self.cno_reverse_metabolism = metabolism;
        self
    }

    /// Set the CLZ metabolism rate.
    pub fn clz_metabolism(&mut self, metabolism: f64) -> &mut Self {
        self.clz_metabolism = metabolism;
        self
    }

    /// Set the CLZ elimination rate.
    pub fn clz_elimination(&mut self, elimination: f64) -> &mut Self {
        self.clz_elimination = elimination;
        self
    }

    /// Set the CNO brain transport rate.
    pub fn cno_brain_transport(&mut self, transport: f64) -> &mut Self {
        self.cno_brain_transport = transport;
        self
    }

    /// Set the CNO plasma transport rate.
    pub fn cno_plasma_transport(&mut self, transport: f64) -> &mut Self {
        self.cno_plasma_transport = transport;
        self
    }

    /// Set the CLZ brain transport rate.
    pub fn clz_brain_transport(&mut self, transport: f64) -> &mut Self {
        self.clz_brain_transport = transport;
        self
    }

    /// Set the CLZ plasma transport rate.
    pub fn clz_plasma_transport(&mut self, transport: f64) -> &mut Self {
        self.clz_plasma_transport = transport;
        self
    }

    /// Set the CNO plasma volume of distribution.
    pub fn cno_plasma_vd(&mut self, vd: f64) -> &mut Self {
        self.cno_plasma_vd = vd;
        self
    }

    /// Set the CNO brain volume of distribution.
    pub fn cno_brain_vd(&mut self, vd: f64) -> &mut Self {
        self.cno_brain_vd = vd;
        self
    }

    /// Set the CLZ plasma volume of distribution.
    pub fn clz_plasma_vd(&mut self, vd: f64) -> &mut Self {
        self.clz_plasma_vd = vd;
        self
    }

    /// Set the CLZ brain volume of distribution.
    pub fn clz_brain_vd(&mut self, vd: f64) -> &mut Self {
        self.clz_brain_vd = vd;
        self
    }

    /// Build the CNO model.
    pub fn build(&self) -> Model {
        Model {
            doses: self.doses.clone(),
            cno_absorption: self.cno_absorption,
            cno_elimination: self.cno_elimination,
            cno_reverse_metabolism: self.cno_reverse_metabolism,
            clz_metabolism: self.clz_metabolism,
            clz_elimination: self.clz_elimination,
            cno_brain_transport: self.cno_brain_transport,
            cno_plasma_transport: self.cno_plasma_transport,
            clz_brain_transport: self.clz_brain_transport,
            clz_plasma_transport: self.clz_plasma_transport,
            cno_plasma_vd: self.cno_plasma_vd,
            cno_brain_vd: self.cno_brain_vd,
            clz_plasma_vd: self.clz_plasma_vd,
            clz_brain_vd: self.clz_brain_vd,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use differential_equations::{methods::ExplicitRungeKutta, status::Status};

    #[test]
    fn cno_dose_creation() {
        let single_dose = Dose::new(0.03, 0.);
        assert_eq!(single_dose.mg, 0.03);
        assert_eq!(single_dose.nmol, 0.03 / CNO_MW * 1e6);
        assert_eq!(single_dose.time, 0.);

        let schedule = create_cno_schedule(0.03, 0., None, None);
        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule[0].mg, 0.03);
        assert_eq!(schedule[0].time, 0.);

        let stacked_doses = create_cno_schedule(0.03, 0., Some(1), None);
        assert_eq!(stacked_doses.len(), 2);
        assert_eq!(stacked_doses[0].mg, 0.03);
        assert_eq!(stacked_doses[0].time, 0.);
        assert_eq!(stacked_doses[1].mg, 0.03);
        assert_eq!(stacked_doses[1].time, 0.);

        let repeated_doses = create_cno_schedule(0.03, 0., Some(1), Some(24.));
        assert_eq!(repeated_doses.len(), 2);
        assert_eq!(repeated_doses[0].mg, 0.03);
        assert_eq!(repeated_doses[0].time, 0.);
        assert_eq!(repeated_doses[1].mg, 0.03);
        assert_eq!(repeated_doses[1].time, 24.);
    }

    #[test]
    fn dox_state_creation() {
        let zero_state = State::zeros();
        assert_eq!(zero_state.peritoneal_cno, 0.);
        assert_eq!(zero_state.plasma_cno, 0.);
        assert_eq!(zero_state.brain_cno, 0.);
        assert_eq!(zero_state.plasma_clz, 0.);
        assert_eq!(zero_state.brain_clz, 0.);

        let default_state = State::default();
        assert_eq!(default_state.peritoneal_cno, 0.);
        assert_eq!(default_state.plasma_cno, 0.);
        assert_eq!(default_state.brain_cno, 0.);
        assert_eq!(default_state.plasma_clz, 0.);
        assert_eq!(default_state.brain_clz, 0.);

        let custom_state = State::new(10., 20., 30., 40., 50.);
        assert_eq!(custom_state.peritoneal_cno, 10.);
        assert_eq!(custom_state.plasma_cno, 20.);
        assert_eq!(custom_state.brain_cno, 30.);
        assert_eq!(custom_state.plasma_clz, 40.);
        assert_eq!(custom_state.brain_clz, 50.);
    }

    #[test]
    fn cno_model_creation() {
        let default_model = Model::default();
        assert_eq!(default_model.doses.len(), 1);
        assert_eq!(default_model.cno_absorption, DEFAULT_CNO_ABSORPTION);
        assert_eq!(default_model.cno_elimination, DEFAULT_CNO_ELIMINATION);

        let dose = Dose::new(0.03, 0.);
        let model_with_dose = Model::builder(vec![dose]).build();
        assert_eq!(model_with_dose.doses.len(), 1);
        assert_eq!(model_with_dose.doses[0].mg, 0.03);
        assert_eq!(model_with_dose.doses[0].time, 0.);

        let schedule = create_cno_schedule(0.03, 0., Some(1), Some(24.));
        let model_with_schedule = Model::builder(schedule).build();
        assert_eq!(model_with_schedule.doses.len(), 2);
    }

    #[test]
    fn cno_model_simulation() {
        let mut solver = ExplicitRungeKutta::dopri5();
        let t0 = 0.;
        let tf = 24.;
        let dt = 1.;

        // test default model - dose (0.03 mg) applied at t=0
        let default_model = Model::default();
        let init_state = State::zeros();
        let solution = default_model.solve(t0, tf, dt, init_state, &mut solver);

        assert!(solution.is_ok());
        let solution = solution.unwrap();
        assert!(matches!(solution.status, Status::Complete));
        assert!(solution.y[0].peritoneal_cno > 0.);

        // apply dose at t=1
        let dose = Dose::new(0.03, 1.);
        let custom_model = Model::builder(vec![dose.clone()]).build();
        let solution = custom_model.solve(t0, tf, dt, init_state, &mut solver);

        assert!(solution.is_ok());
        let solution = solution.unwrap();
        assert!(matches!(solution.status, Status::Complete));
        assert_eq!(solution.y[1].peritoneal_cno, dose.clone().nmol);
    }

    #[test]
    fn small_dt() {
        let model = Model::builder(vec![Dose::new(0.03, 1.)]).build();
        let mut solver = ExplicitRungeKutta::dopri5();
        let init_state = State::zeros();

        let solution = model.solve(0., 10., 0.1, init_state, &mut solver);
        assert!(solution.is_ok());
    }

    #[test]
    fn expected_ts() {
        let model = Model::builder(vec![Dose::new(0.03, 1.)]).build();
        let dt = 1.;
        let t0 = 0.;
        let tf = 10.;
        let init_state = State::zeros();
        let mut solver = ExplicitRungeKutta::dopri5();

        let solution = model.solve(t0, tf, dt, init_state, &mut solver);
        assert!(solution.is_ok());
        let solution = solution.unwrap();
        assert!(matches!(solution.status, Status::Complete));
        let expected_len = ((tf - t0) / dt).ceil() as usize + 1;
        assert_eq!(solution.y.len(), expected_len);
        println!("{:?}", solution.t);

        let model = Model::builder(vec![Dose::new(0.03, 1.5)]).build();
        let solution = model.solve(t0, tf, dt, init_state, &mut solver);
        assert!(solution.is_ok());
        let solution = solution.unwrap();
        assert!(matches!(solution.status, Status::Complete));
        let uneven_expected_len = ((tf - t0) / dt).ceil() as usize + 2;
        assert_eq!(solution.y.len(), uneven_expected_len);

        assert_eq!(solution.t[0], t0);
        assert_eq!(solution.t[2], 1.5);
        assert_eq!(solution.t[3], 2.0);
        assert_eq!(solution.t[4], 3.0);
    }
}
