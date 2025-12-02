use crate::Solve;
use differential_equations::{
    control::ControlFlag,
    derive::State as StateTrait,
    error::Error,
    ode::{ODE, ODEProblem, OrdinaryNumericalMethod},
    prelude::{Interpolation, Solution},
    solout::Solout,
};
use std::cell::Cell;

#[cfg(feature = "py")]
use pyo3::{PyResult, pyclass, pyfunction, pymethods};

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
    pub fn set_mg(&mut self, mg: f64) {
        self.mg = mg;
    }

    /// Set amount in nmol.
    #[setter]
    pub fn set_nmol(&mut self, nmol: f64) {
        self.nmol = nmol;
    }
}

/// Helper to create a CNO schedule given an amount in mg, start time, number of times to repeat,
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

    /// Create a new CNO model state given peritoneal, plasma and brain CNO; plasma and brain CLZ concentrations.
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
        State::zeros()
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
/// peritoneal, plasma and brain CNO; plasma and brain CLZ concentrations
/// without manual state construction.
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

/// Custom Solout that handles dose events using ControlFlag::ModifyState.
/// Combines evenly-spaced output with dose application at specific times.
pub struct DosingSolout<'a> {
    doses: &'a [Dose],
    next_dose_index: Cell<usize>,
    dt: f64,
    next_output_time: Cell<f64>,
}

impl<'a> DosingSolout<'a> {
    /// Create a new DosingSolout with the given doses and output time step.
    pub fn new(doses: &'a [Dose], t0: f64, dt: f64) -> Self {
        Self {
            doses,
            next_dose_index: Cell::new(0),
            dt,
            next_output_time: Cell::new(t0),
        }
    }
}

impl<'a> Solout<f64, State<f64>> for DosingSolout<'a> {
    fn solout<I>(
        &mut self,
        t_curr: f64,
        t_prev: f64,
        y_curr: &State<f64>,
        _y_prev: &State<f64>,
        interpolator: &mut I,
        solution: &mut Solution<f64, State<f64>>,
    ) -> ControlFlag<f64, State<f64>>
    where
        I: Interpolation<f64, State<f64>>,
    {
        // First, check if we crossed a dose time during this step
        let dose_idx = self.next_dose_index.get();
        if dose_idx < self.doses.len() {
            let dose = &self.doses[dose_idx];
            if dose.time > t_prev && dose.time <= t_curr {
                let mut dosed_state = match interpolator.interpolate(dose.time) {
                    Ok(state) => state,
                    Err(_) => y_curr.clone(), // Fallback to current state on interpolation error
                };
                // Apply the dose to peritoneal compartment
                *dosed_state.peritoneal_cno_mut() += dose.nmol;
                self.next_dose_index.set(dose_idx + 1);
                return ControlFlag::ModifyState(dose.time, dosed_state);
            }
        }

        // Handle evenly-spaced output
        let mut next_t = self.next_output_time.get();
        while next_t <= t_curr {
            if next_t >= t_prev {
                let y_out = if (next_t - t_curr).abs() < 1e-10 {
                    y_curr.clone()
                } else if next_t > t_prev {
                    match interpolator.interpolate(next_t) {
                        Ok(state) => state,
                        Err(_) => continue,
                    }
                } else {
                    continue;
                };
                solution.t.push(next_t);
                solution.y.push(y_out);
            }
            next_t += self.dt;
        }
        self.next_output_time.set(next_t);

        ControlFlag::Continue
    }
}

impl Model {
    /// Create a new CNO model builder.
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
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

impl Solve for Model {
    type State = State<f64>;

    fn solve<S>(
        &self,
        t0: f64,
        tf: f64,
        dt: f64,
        init_state: State<f64>,
        solver: &mut S,
    ) -> Result<Solution<f64, State<f64>>, Error<f64, State<f64>>>
    where
        S: OrdinaryNumericalMethod<f64, State<f64>> + Interpolation<f64, State<f64>>,
    {
        // Create custom solout that handles both evenly-spaced output and dose events
        let mut dosing_solout = DosingSolout::new(&self.doses, t0, dt);

        // Solve the ODE problem with our custom solout
        let problem = ODEProblem::new(self, t0, tf, init_state);
        problem.solout(&mut dosing_solout).solve(solver)
    }
}

impl ODE<f64, State<f64>> for Model {
    fn diff(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        self.diff_with(t, y, dydt);
    }
}

pub struct ModelBuilder {
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

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            doses: Vec::new(),
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
    /// Set the CNO doses.
    pub fn doses(&mut self, doses: Vec<Dose>) -> &mut Self {
        self.doses = doses;
        self
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
    use differential_equations::methods::ExplicitRungeKutta;
    use differential_equations::status::Status;

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
        assert_eq!(default_model.doses.len(), 0);
        assert_eq!(default_model.cno_absorption, DEFAULT_CNO_ABSORPTION);
        assert_eq!(default_model.cno_elimination, DEFAULT_CNO_ELIMINATION);

        let dose = Dose::new(0.03, 0.);
        let model_with_dose = Model::builder().doses(vec![dose]).build();
        assert_eq!(model_with_dose.doses.len(), 1);
        assert_eq!(model_with_dose.doses[0].mg, 0.03);
        assert_eq!(model_with_dose.doses[0].time, 0.);
    }

    #[test]
    fn cno_model_dose_at_t0() {
        let dose = Dose::new(0.5, 0.0);
        let model = Model::builder().doses(vec![dose]).build();
        let mut solver = ExplicitRungeKutta::dopri5();
        let init_state = State::zeros();

        let solution = model.solve(0., 24., 0.1, init_state, &mut solver);
        assert!(solution.is_ok());

        let solution = solution.unwrap();
        assert!(!solution.t.is_empty());
        assert!(matches!(solution.status, Status::Complete));
    }

    #[test]
    fn cno_model_dose_post_t0() {
        let zero_model = Model::default();
        let mut solver = ExplicitRungeKutta::dopri5();
        let init_state = State::zeros();
        let solution = zero_model.solve(0., 24., 0.1, init_state, &mut solver);
        assert!(solution.is_ok());

        let unwrapped_solution = solution.unwrap();
        assert_eq!(unwrapped_solution.y.last().unwrap().peritoneal_cno, 0.);
        assert_eq!(unwrapped_solution.y.last().unwrap().plasma_cno, 0.);
        assert_eq!(unwrapped_solution.y.last().unwrap().brain_cno, 0.);
        assert_eq!(unwrapped_solution.y.last().unwrap().plasma_clz, 0.);
        assert_eq!(unwrapped_solution.y.last().unwrap().brain_clz, 0.);

        let dose = Dose::new(0.5, 1.);
        let model_with_dose = Model::builder().doses(vec![dose]).build();
        let solution = model_with_dose.solve(0., 24., 0.1, init_state, &mut solver);
        assert!(solution.is_ok());

        let solution = solution.unwrap();
        let post_dose_idx = solution.t.iter().position(|&t| t > 1.0).unwrap_or(0);
        assert!(post_dose_idx > 0);
    }

    #[test]
    fn cno_model_multiple_doses() {
        // Test multiple doses at different times
        let doses = vec![
            Dose::new(0.5, 1.0),  // First dose at t=1
            Dose::new(0.5, 12.0), // Second dose at t=12
        ];
        let model = Model::builder().doses(doses).build();
        let mut solver = ExplicitRungeKutta::dopri5();
        let init_state = State::zeros();

        let solution = model.solve(0., 24., 0.1, init_state, &mut solver);
        assert!(solution.is_ok());

        let solution = solution.unwrap();
        assert!(matches!(solution.status, Status::Complete));
    }
}
