use crate::pk::{ScheduledDose, ScheduledStateUpdate};

#[cfg(feature = "py")]
use pyo3::{pyclass, pyfunction, pymethods, PyResult};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Default TEV dose amount in nmol.
pub const DEFAULT_TEV_DOSE_NMOL: f64 = 0.;
/// Default TEV dose administration time.
pub const DEFAULT_TEV_DOSE_TIME: f64 = 0.;
/// Default TEV plasma volume of distribution.
pub const DEFAULT_TEV_PLASMA_VD: f64 = 1.;
/// Default TEV degradation rate.
pub const DEFAULT_TEV_DEG: f64 = 0.1;
/// Default TEV-mediated cleavage rate.
pub const DEFAULT_TEV_CUT_RATE: f64 = 0.01;

/// Trait for states that expose a plasma TEV amount field.
///
/// This allows TEV dosing logic to be shared across erasable models with
/// different state layouts.
pub trait TevFields {
    fn plasma_tev(&self) -> f64;
    fn plasma_tev_mut(&mut self) -> &mut f64;
}

/// Defines a TEV dose in nmol and administration time.
#[cfg_attr(feature = "py", pyclass(name = "TevDose"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone)]
pub struct TevDose {
    pub nmol: f64,
    pub time: f64,
}

impl TevDose {
    pub fn new(nmol: f64, time: f64) -> Self {
        Self { nmol, time }
    }
}

impl ScheduledDose for TevDose {
    fn time(&self) -> f64 {
        self.time
    }

    fn amount(&self) -> f64 {
        self.nmol
    }
}

impl<S: TevFields> ScheduledStateUpdate<S> for TevDose {
    fn time(&self) -> f64 {
        self.time
    }

    fn apply(&self, state: &mut S) {
        *state.plasma_tev_mut() += self.nmol;
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl TevDose {
    #[new]
    pub fn create(nmol: f64, time: f64) -> Self {
        Self::new(nmol, time)
    }

    #[getter]
    pub fn get_nmol(&self) -> f64 {
        self.nmol
    }

    #[getter]
    pub fn get_time(&self) -> f64 {
        self.time
    }

    #[setter]
    pub fn set_nmol(&mut self, nmol: f64) -> PyResult<()> {
        self.nmol = nmol;
        Ok(())
    }

    #[setter]
    pub fn set_time(&mut self, time: f64) -> PyResult<()> {
        self.time = time;
        Ok(())
    }
}

/// Create a TEV schedule given an amount in nmol, start time, number of repeats,
/// and interval between administrations.
#[cfg_attr(feature = "py", pyfunction)]
#[cfg_attr(
    feature = "py",
    pyo3(signature = (nmol, start_time, repeat=None, interval=None))
)]
pub fn create_tev_schedule(
    nmol: f64,
    start_time: f64,
    repeat: Option<usize>,
    interval: Option<f64>,
) -> Vec<TevDose> {
    let mut schedule = Vec::new();
    let mut current_time = start_time;
    let interval = interval.unwrap_or(0.);

    for _ in 0..repeat.unwrap_or(0) + 1 {
        schedule.push(TevDose::new(nmol, current_time));
        current_time += interval;
    }

    schedule
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Default)]
    struct DummyState {
        plasma_tev: f64,
    }

    impl TevFields for DummyState {
        fn plasma_tev(&self) -> f64 {
            self.plasma_tev
        }

        fn plasma_tev_mut(&mut self) -> &mut f64 {
            &mut self.plasma_tev
        }
    }

    #[test]
    fn tev_schedule_creation() {
        let schedule = create_tev_schedule(20., 4., Some(2), Some(24.));
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0].nmol, 20.);
        assert_eq!(schedule[0].time, 4.);
        assert_eq!(schedule[1].time, 28.);
        assert_eq!(schedule[2].time, 52.);
    }

    #[test]
    fn tev_dose_applies_to_any_tev_state() {
        let dose = TevDose::new(12., 3.);
        let mut state = DummyState::default();

        dose.apply(&mut state);

        assert_eq!(state.plasma_tev(), 12.);
    }

    #[test]
    fn tev_defaults_are_stable() {
        assert_eq!(DEFAULT_TEV_DOSE_NMOL, 0.);
        assert_eq!(DEFAULT_TEV_DOSE_TIME, 0.);
        assert_eq!(DEFAULT_TEV_PLASMA_VD, 1.);
        assert_eq!(DEFAULT_TEV_DEG, 0.1);
        assert_eq!(DEFAULT_TEV_CUT_RATE, 0.01);
    }
}
