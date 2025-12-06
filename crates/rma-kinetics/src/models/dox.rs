//! Doxycycline pharmacokinetic model.
//!
//! A simple pharmacokinetic model describing doxycyline dynamics in the brain and plasma
//! following food or water intake.
//!
//! ## Parameters
//!
//! Doxycycline is assumed to be adminstered via food or water.
//! The vehicle intake rate is the input of food or water per unit time.
//!
//! Bioavailability is the fraction of the dose that is absorbed into the plasma.
//!
//! The following rates are used to desribe absorption, elimination, and transport between the plasma and brain.
//! - absorption: the rate at which doxycycline is absorbed into the plasma.
//! - elimination: the rate at which doxycycline is eliminated from the plasma.
//! - brain_transport: the rate at which doxycycline is transported from the plasma to the brain.
//! - plasma_transport: the rate at which doxycycline is transported from the brain to the plasma.
//! - plasma_vd: the volume of distribution of doxycycline in the plasma.
//!
//! By default, the model does not set any dox administration.
//! To set the periods of dox administration, use the `schedule` method on the model builder.
//!
//! ## Usage
//!
//! To solve the model over a given period of time, we use the solvers provided by
//! the `differential_equations` dependency. From here, we can use the provided `Solve`
//! trait and use the `solve` method on our model.
//!
//! ```rust
//! use rma_kinetics::{models::dox, pk::Error, Solve};
//! use differential_equations::methods::ExplicitRungeKutta;
//!
//! let dox_access_period = dox::AccessPeriod::new(40., 0.0..=24.);
//! let model = dox::Model::builder().schedule(vec![dox_access_period]).build()?;
//! let init_state = dox::State::zeros();
//! let mut solver = ExplicitRungeKutta::dopri5();
//!
//! let solution = model.solve(0., 48., 1., init_state, &mut solver);
//! assert!(solution.is_ok());
//! Ok::<(), Error>(())
//! ```
//!

use crate::pk::Error;
use differential_equations::{derive::State as StateTrait, ode::ODE, prelude::Matrix};
use rma_kinetics_derive::Solve;
use std::ops::RangeInclusive;

#[cfg(feature = "py")]
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pyfunction, pymethods};

#[cfg(feature = "py")]
use rma_kinetics_derive::PySolve;

const DOX_MW: f64 = 444.4; // g/mol

/// Defines the concentration and period of access of dox food or water.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "py", pyclass(name = "AccessPeriod"))]
pub struct AccessPeriod {
    pub dose: f64,
    pub time: RangeInclusive<f64>,
}

impl AccessPeriod {
    /// Create a new `AccessPeriod` given a dose and time range.
    pub fn new(dose: f64, time: RangeInclusive<f64>) -> Self {
        Self { dose, time }
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl AccessPeriod {
    #[new]
    pub fn create(dose: f64, start_time: f64, stop_time: f64) -> Self {
        let time = start_time..=stop_time;
        Self::new(dose, time)
    }

    #[getter]
    fn get_dose(&self) -> f64 {
        self.dose
    }

    #[getter]
    fn get_start_time(&self) -> f64 {
        *self.time.start()
    }

    #[getter]
    fn get_stop_time(&self) -> f64 {
        *self.time.end()
    }
}

/// Create a dox schedule given a dose, start time, duration of the access period, interval between access periods, and number of repeated administrations.
///
/// ## Usage
/// ```rust
/// use rma_kinetics::models::dox::create_dox_schedule;
///
/// // creates a schedule with two access periods with a
/// // duration of 24 hours and an interval of 24 hours.
/// let schedule = create_dox_schedule(40., 0., 24., Some(1), Some(24.));
/// assert_eq!(schedule.len(), 2);
/// ```
#[cfg_attr(feature = "py", pyfunction)]
#[cfg_attr(feature = "py", pyo3(signature = (dose, start_time, duration, repeat=None, interval=None)))]
pub fn create_dox_schedule(
    dose: f64,
    start_time: f64,
    duration: f64,
    repeat: Option<usize>,
    interval: Option<f64>,
) -> Vec<AccessPeriod> {
    let mut schedule = Vec::new();
    let mut current_time = start_time;
    let interval = interval.unwrap_or(0.);

    for _ in 0..repeat.unwrap_or(0) + 1 {
        schedule.push(AccessPeriod::new(
            dose,
            current_time..=current_time + duration,
        ));
        current_time += duration + interval;
    }

    schedule
}

/// Dox model state
#[derive(StateTrait)]
pub struct State<T> {
    pub plasma_dox: T,
    pub brain_dox: T,
}

impl State<f64> {
    /// Get a dox model state where brain and plasma dox concentration are set to 0.
    pub fn zeros() -> Self {
        Self {
            plasma_dox: 0.,
            brain_dox: 0.,
        }
    }

    /// Create a new constitutive model state given plasma and brain dox concentrations.
    pub fn new(plasma_dox: f64, brain_dox: f64) -> Self {
        Self {
            plasma_dox,
            brain_dox,
        }
    }
}

impl Default for State<f64> {
    /// Default dox model state where plasma and brain dox concentration are set to 0.
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
    #[pyo3(signature = (plasma_dox=0., brain_dox=0.))]
    pub fn new(plasma_dox: f64, brain_dox: f64) -> Self {
        Self {
            inner: State {
                plasma_dox,
                brain_dox,
            },
        }
    }

    /// Get plasma dox state
    #[getter]
    fn get_plasma_dox(&self) -> f64 {
        self.inner.plasma_dox
    }
    /// Get brain dox state
    #[getter]
    fn get_brain_dox(&self) -> f64 {
        self.inner.brain_dox
    }
    /// Set plasma dox state
    #[setter]
    fn set_plasma_dox(&mut self, value: f64) -> PyResult<()> {
        self.inner.plasma_dox = value;
        Ok(())
    }
    /// Set brain dox state
    #[setter]
    fn set_brain_dox(&mut self, value: f64) -> PyResult<()> {
        self.inner.brain_dox = value;
        Ok(())
    }
}

/// Trait for types that contain dox-related fields.
/// This enables the Dox model to use any state type that provides
/// plasma and brain dox concentrations without manual state construction.
pub trait DoxFields {
    fn plasma_dox(&self) -> f64;
    fn brain_dox(&self) -> f64;
    fn plasma_dox_mut(&mut self) -> &mut f64;
    fn brain_dox_mut(&mut self) -> &mut f64;
}

impl DoxFields for State<f64> {
    fn plasma_dox(&self) -> f64 {
        self.plasma_dox
    }

    fn brain_dox(&self) -> f64 {
        self.brain_dox
    }

    fn plasma_dox_mut(&mut self) -> &mut f64 {
        &mut self.plasma_dox
    }

    fn brain_dox_mut(&mut self) -> &mut f64 {
        &mut self.brain_dox
    }
}

/// Dox PK model
#[cfg_attr(feature = "py", pyclass)]
#[cfg_attr(feature = "py", derive(PySolve))]
#[cfg_attr(feature = "py", py_solve(variant = "Dox"))]
#[derive(Debug, Solve, Clone)]
pub struct Model {
    /// Vehicle (food or water) intake rate.
    pub vehicle_intake: f64,
    /// Bioavailability in the range [0, 1].
    pub bioavailability: f64,
    /// Plasma absorption rate.
    pub absorption: f64,
    /// Plasma elimination rate.
    pub elimination: f64,
    /// Plasma to brain transport rate.
    pub brain_transport: f64,
    /// Brain to plasma transport rate.
    pub plasma_transport: f64,
    /// Plasma volume of distribution.
    pub plasma_vd: f64,
    /// Dox administration schedule.
    pub schedule: Vec<AccessPeriod>,
    /// Dose concentration for each access period (typically in nM if dose is given in mg).
    pub dose_concentration: Vec<f64>,
}

const DEFAULT_VEHICLE_INTAKE: f64 = 1.875e-4; // mg chow / hr (4.5 g chow / day)
const DEFAULT_BIOAVAILABILITY: f64 = 0.9;
const DEFAULT_ABSORPTION: f64 = 0.8;
const DEFAULT_ELIMINATION: f64 = 0.2;
const DEFAULT_BRAIN_TRANSPORT: f64 = 0.2;
const DEFAULT_PLASMA_TRANSPORT: f64 = 1.0;
const DEFAULT_PLASMA_VD: f64 = 0.21;

#[cfg(feature = "py")]
#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (vehicle_intake=DEFAULT_VEHICLE_INTAKE, bioavailability=DEFAULT_BIOAVAILABILITY, absorption=DEFAULT_ABSORPTION, elimination=DEFAULT_ELIMINATION, brain_transport=DEFAULT_BRAIN_TRANSPORT, plasma_transport=DEFAULT_PLASMA_TRANSPORT, plasma_vd=DEFAULT_PLASMA_VD, schedule=Vec::new()))]
    pub fn create(
        vehicle_intake: f64,
        bioavailability: f64,
        absorption: f64,
        elimination: f64,
        brain_transport: f64,
        plasma_transport: f64,
        plasma_vd: f64,
        schedule: Vec<AccessPeriod>,
    ) -> Self {
        let dose_concentration = schedule
            .iter()
            .map(|period| {
                period.dose * bioavailability * vehicle_intake / (DOX_MW * plasma_vd) * 1e6
            })
            .collect();

        Self {
            vehicle_intake,
            bioavailability,
            absorption,
            elimination,
            brain_transport,
            plasma_transport,
            plasma_vd,
            schedule,
            dose_concentration,
        }
    }

    #[pyo3(name = "solve")]
    fn py_solve(
        &self,
        t0: f64,
        tf: f64,
        dt: f64,
        init_state: PyState,
        solver: crate::solve::PySolver,
    ) -> PyResult<crate::solve::PySolution> {
        let result = crate::solve::PySolve::solve(self, t0, tf, dt, init_state.inner, solver);
        match result {
            Ok(solution) => Ok(solution),
            Err(e) => Err(PyValueError::new_err(format!("Failed to solve: {:?}", e))),
        }
    }
}

impl Model {
    /// Create a new dox model builder.
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
    }

    fn intake(&self, t: f64) -> f64 {
        if self.schedule.is_empty() {
            return 0.;
        }

        self.schedule
            .iter()
            .enumerate()
            .filter(|(_, period)| period.time.contains(&t))
            .map(|(index, _)| self.dose_concentration[index])
            .sum::<f64>()
    }

    /// System of differential equations describing dox dynamics. Works with any state type that implements DoxFields.
    pub fn diff_with<S: DoxFields>(&self, t: f64, y: &S, dydt: &mut S) {
        let plasma_efflux = self.brain_transport * y.plasma_dox();
        let brain_efflux = self.plasma_transport * y.brain_dox();
        *dydt.plasma_dox_mut() = (self.absorption * self.intake(t))
            - (self.elimination * y.plasma_dox())
            - plasma_efflux
            + brain_efflux;
        *dydt.brain_dox_mut() = plasma_efflux - brain_efflux;
    }

    pub fn jacobian_with<S: DoxFields>(&self, _t: f64, _y: &S, j: &mut Matrix<f64>) {
        j[(0, 0)] = -self.elimination - self.brain_transport;
        j[(0, 1)] = self.plasma_transport;
        j[(1, 0)] = self.brain_transport;
        j[(1, 1)] = -self.plasma_transport;
    }
}

impl Default for Model {
    fn default() -> Self {
        ModelBuilder::default().build().unwrap()
    }
}

impl ODE<f64, State<f64>> for Model {
    fn diff(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        self.diff_with(t, y, dydt);
    }
}

/// Dox model builder.
pub struct ModelBuilder {
    pub vehicle_intake: f64,
    pub bioavailability: f64,
    pub absorption: f64,
    pub elimination: f64,
    pub brain_transport: f64,
    pub plasma_transport: f64,
    pub plasma_vd: f64,
    pub schedule: Vec<AccessPeriod>,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            vehicle_intake: DEFAULT_VEHICLE_INTAKE,
            bioavailability: DEFAULT_BIOAVAILABILITY,
            absorption: DEFAULT_ABSORPTION,
            elimination: DEFAULT_ELIMINATION,
            brain_transport: DEFAULT_BRAIN_TRANSPORT,
            plasma_transport: DEFAULT_PLASMA_TRANSPORT,
            plasma_vd: DEFAULT_PLASMA_VD,
            schedule: Vec::new(),
        }
    }
}

impl ModelBuilder {
    /// Set the vehicle (food or water) intake rate (mg/hr)
    pub fn vehicle_intake(&mut self, intake: f64) -> &mut Self {
        self.vehicle_intake = intake;
        self
    }

    /// Set the bioavailability of the vehicle (food or water) intake (0-1)
    pub fn bioavailability(&mut self, bioavailability: f64) -> Result<&mut Self, Error> {
        if !(0. ..=1.).contains(&bioavailability) {
            return Err(Error::InvalidBioavailability(bioavailability));
        }

        self.bioavailability = bioavailability;
        Ok(self)
    }

    /// Set plasma absorption rate.
    pub fn absorption(&mut self, absorption: f64) -> &mut Self {
        self.absorption = absorption;
        self
    }

    /// Set plasma elimination rate.
    pub fn elimination(&mut self, elimination: f64) -> &mut Self {
        self.elimination = elimination;
        self
    }

    /// Set plasma to brain transport rate.
    pub fn brain_transport(&mut self, transport: f64) -> &mut Self {
        self.brain_transport = transport;
        self
    }

    /// Set brain to plasma transport rate.
    pub fn plasma_transport(&mut self, transport: f64) -> &mut Self {
        self.plasma_transport = transport;
        self
    }

    /// Set the volume of distribution.
    pub fn plasma_vd(&mut self, vd: f64) -> &mut Self {
        self.plasma_vd = vd;
        self
    }

    /// Set dox administration schedules
    pub fn schedule(&mut self, access_periods: Vec<AccessPeriod>) -> &mut Self {
        self.schedule.extend(access_periods);
        self
    }

    pub fn build(&self) -> Result<Model, Error> {
        let dose_concentration = self
            .schedule
            .iter()
            .map(|period| {
                period.dose * self.bioavailability * self.vehicle_intake / (DOX_MW * self.plasma_vd)
                    * 1e6
            })
            .collect();

        Ok(Model {
            vehicle_intake: self.vehicle_intake,
            bioavailability: self.bioavailability,
            absorption: self.absorption,
            elimination: self.elimination,
            brain_transport: self.brain_transport,
            plasma_transport: self.plasma_transport,
            plasma_vd: self.plasma_vd,
            schedule: self.schedule.clone(),
            dose_concentration,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::Solve;
    use differential_equations::methods::ExplicitRungeKutta;

    #[test]
    fn dox_access_period_creation() {
        let access_period = AccessPeriod::new(40., 0.0..=24.);
        assert_eq!(access_period.dose, 40.);
        assert_eq!(access_period.time, 0.0..=24.);
    }

    #[test]
    fn dox_schedule_creation() {
        let single_period_schedule = create_dox_schedule(40., 0., 24., None, None);
        assert_eq!(single_period_schedule.len(), 1);

        // here we create a schedule with a duration of 24 hours and repeat it once. Since the interval is not specified, it is assumed to be 0.
        let long_schedule = create_dox_schedule(40., 0., 24., Some(1), None);
        assert_eq!(long_schedule.len(), 2);
        assert_eq!(long_schedule[0].dose, 40.);
        assert_eq!(long_schedule[0].time, 0.0..=24.);
        assert_eq!(long_schedule[1].dose, 40.);
        assert_eq!(long_schedule[1].time, 24.0..=48.);

        // create a schedule with access period durations of 24 hours and we repeat it once with an interval of 24 hours.
        let repeated_schedule = create_dox_schedule(40., 0., 24., Some(1), Some(24.));
        assert_eq!(repeated_schedule.len(), 2);
        assert_eq!(repeated_schedule[0].dose, 40.);
        assert_eq!(repeated_schedule[0].time, 0.0..=24.);
        assert_eq!(repeated_schedule[1].dose, 40.);
        assert_eq!(repeated_schedule[1].time, 48.0..=72.);
    }

    #[test]
    fn dox_state_creation() {
        let zero_state = State::zeros();
        assert_eq!(zero_state.plasma_dox, 0.);
        assert_eq!(zero_state.brain_dox, 0.);

        // also defaults to 0.

        let default_state = State::default();
        assert_eq!(default_state.plasma_dox, 0.);
        assert_eq!(default_state.brain_dox, 0.);

        let custom_state = State::new(10., 20.);
        assert_eq!(custom_state.plasma_dox, 10.);
        assert_eq!(custom_state.brain_dox, 20.);
    }

    #[test]
    fn dox_model_creation() -> Result<(), Error> {
        let default_model = Model::default();
        assert_eq!(default_model.schedule.len(), 0); // no schedule set for the default model

        let model_with_schedule = Model::builder()
            .schedule(vec![AccessPeriod::new(40., 0.0..=24.)])
            .build()?;
        assert_eq!(model_with_schedule.schedule.len(), 1);
        assert_eq!(model_with_schedule.absorption, DEFAULT_ABSORPTION);

        Ok(())
    }

    #[test]
    fn dox_model_simulation() -> Result<(), Error> {
        let zero_model = Model::default();
        let mut solver = ExplicitRungeKutta::dopri5();
        let init_state = State::zeros();
        let solution = zero_model.solve(0., 24., 1., init_state, &mut solver);
        assert!(solution.is_ok());
        let unwrapped_solution = solution.unwrap();
        assert_eq!(unwrapped_solution.y.last().unwrap().plasma_dox, 0.);
        assert_eq!(unwrapped_solution.y.last().unwrap().brain_dox, 0.);

        // add dox administration period
        let custom_model = Model::builder()
            .schedule(vec![AccessPeriod::new(40., 0.0..=24.)])
            .build()?;
        let solution = custom_model.solve(0., 24., 1., init_state, &mut solver);
        assert!(solution.is_ok());
        let unwrapped_solution = solution.unwrap();
        assert!(unwrapped_solution.y.last().unwrap().plasma_dox > 0.);
        assert!(unwrapped_solution.y.last().unwrap().brain_dox > 0.);

        Ok(())
    }
}
