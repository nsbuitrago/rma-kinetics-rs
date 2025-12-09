//! Oscillating RMA expression model.
//!
//! The oscillation model is a model that describes the expression of a synthetic serum reporter
//! in the brain tissue and blood-brain barrier transport to the plasma, with time-varying
//! production following a sine wave pattern.
//!
//! ## Parameters
//! - Production rate: 0.2 nM/hr (base rate, oscillates around this value)
//! - Frequency: 0.01 Hz (oscillation frequency)
//! - Blood-brain barrier transport rate: 0.6 1/hr
//! - Degradation rate: 0.007 1/hr
//!
//! ## Usage
//!
//! To solve the model over a given period of time, we use the solvers provided by
//! the `differential_equations` dependency. From here, we can use the provided `Solve`
//! trait and use the `solve` method on our model.
//!
//! ```rust
//! use rma_kinetics::{models::oscillation, Solve};
//! use differential_equations::methods::ExplicitRungeKutta;
//!
//! let model = oscillation::Model::default();
//! let init_state = oscillation::State::zeros();
//! let mut solver = ExplicitRungeKutta::dopri5();
//!
//! let solution = model.solve(0., 100., 1., init_state, &mut solver);
//! assert!(solution.is_ok());
//!
//! let solution = solution.unwrap();
//! println!("{:?}", solution.y);
//! ```

use crate::{impl_solution_access_basic_rma, solve::ApplyNoise};
use derive_builder::Builder;
use differential_equations::{
    derive::State as StateTrait,
    ode::ODE,
    prelude::{Matrix, Solution},
};
use rand::Rng;
use rand_distr::StandardNormal;
use rma_kinetics_derive::Solve;

#[cfg(feature = "py")]
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pymethods};

#[cfg(feature = "py")]
use rma_kinetics_derive::PySolve;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Oscillation model state.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(StateTrait)]
pub struct State<T> {
    pub brain_rma: T,
    pub plasma_rma: T,
}

impl State<f64> {
    /// Get an oscillation model state where brain and plasma RMA concentration
    /// are set to 0.
    pub fn zeros() -> Self {
        Self {
            brain_rma: 0.,
            plasma_rma: 0.,
        }
    }

    /// Create a new oscillation model state given brain and plasma RMA concentrations.
    pub fn new(brain_rma: f64, plasma_rma: f64) -> Self {
        Self {
            brain_rma,
            plasma_rma,
        }
    }
}

impl Default for State<f64> {
    /// Default oscillation model state where brain and plasma RMA concentration
    /// are set to 0.
    fn default() -> Self {
        State::zeros()
    }
}

impl_solution_access_basic_rma!(Solution<f64, State<f64>>, State<f64>);

#[cfg(feature = "py")]
macro_rules! create_interface {
    ($name: ident, $type: ident) => {
        #[derive(Clone)]
        #[pyclass(name = "State")]
        pub struct $name {
            pub inner: State<$type>,
        }
        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (brain_rma=0., plasma_rma=0.))]
            pub fn new(brain_rma: $type, plasma_rma: $type) -> Self {
                Self {
                    inner: State {
                        brain_rma,
                        plasma_rma,
                    },
                }
            }

            #[getter]
            fn get_brain_rma(&self) -> f64 {
                self.inner.brain_rma
            }

            #[getter]
            fn get_plasma_rma(&self) -> f64 {
                self.inner.plasma_rma
            }

            #[setter]
            fn set_brain_rma(&mut self, value: f64) -> PyResult<()> {
                self.inner.brain_rma = value;
                Ok(())
            }

            #[setter]
            fn set_plasma_rma(&mut self, value: f64) -> PyResult<()> {
                self.inner.plasma_rma = value;
                Ok(())
            }
        }
    };
}

#[cfg(feature = "py")]
create_interface!(PyState, f64);

/// Default oscillating RMA production rate.
const DEFAULT_PROD: f64 = 0.2;
/// Default oscillation frequency.
const DEFAULT_FREQ: f64 = 0.01;
/// Default oscillating RMA blood-brain barrier transport rate.
const DEFAULT_BBB_TRANSPORT: f64 = 0.6;
/// Default oscillating RMA degradation rate.
const DEFAULT_DEG: f64 = 0.007;

/// Oscillating RMA expression model.
///
/// The [`default`](Model::default), [`new`](Model::new) or [`builder`](Model::builder)
/// methods can be used to create a new model instance. See `solve` for more
/// information on integration.
#[cfg_attr(feature = "py", pyclass)]
#[cfg_attr(feature = "py", derive(PySolve))]
#[cfg_attr(feature = "py", py_solve(variant = "Oscillation"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Solve, Builder)]
#[builder(derive(Debug))]
pub struct Model {
    /// RMA production rate (base rate, oscillates around this value).
    #[builder(default = "DEFAULT_PROD")]
    pub prod: f64,
    /// Oscillation frequency (Hz).
    #[builder(default = "DEFAULT_FREQ")]
    pub freq: f64,
    /// RMA blood-brain barrier transport rate.
    #[builder(default = "DEFAULT_BBB_TRANSPORT")]
    pub bbb_transport: f64,
    /// RMA degradation rate.
    #[builder(default = "DEFAULT_DEG")]
    pub deg: f64,
}

#[cfg(feature = "py")]
#[pymethods]
impl Model {
    /// Create a new oscillating expression model given RMA production, frequency,
    /// blood-brain barrier transport, and degradation rates.
    #[new]
    #[pyo3(signature = (prod=DEFAULT_PROD, freq=DEFAULT_FREQ, bbb_transport=DEFAULT_BBB_TRANSPORT, deg=DEFAULT_DEG))]
    pub fn create(prod: f64, freq: f64, bbb_transport: f64, deg: f64) -> Self {
        Self {
            prod,
            freq,
            bbb_transport,
            deg,
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
    /// Create a new oscillating expression model given RMA production, frequency,
    /// blood-brain barrier transport, and degradation rates.
    pub fn new(prod: f64, freq: f64, bbb_transport: f64, deg: f64) -> Self {
        Self {
            prod,
            freq,
            bbb_transport,
            deg,
        }
    }

    /// Create a new ModelBuilder for constructing a model instance. This is useful
    /// if you need to update a single rate parameter for example.
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
    }
}

impl Default for Model {
    /// Create a new oscillation model instance with the default parameters
    /// for CA1 hippocampus expression driven by a human-synapsin promoter.
    fn default() -> Self {
        ModelBuilder::default().build().unwrap()
    }
}

impl ODE<f64, State<f64>> for Model {
    /// System of differential equations describing oscillating RMA expression
    /// in the brain tissue and blood-brain barrier transport to the plasma.
    /// Production varies with time as: prod * (1 + sin(2*pi*freq*t))
    fn diff(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        let production = self.prod * (1.0 + (2.0 * std::f64::consts::PI * self.freq * t).sin());
        let brain_efflux = self.bbb_transport * y.brain_rma;
        dydt.brain_rma = production - brain_efflux;
        dydt.plasma_rma = brain_efflux - (self.deg * y.plasma_rma);
    }

    fn jacobian(&self, _t: f64, _y: &State<f64>, j: &mut Matrix<f64>) {
        j[(0, 0)] = -self.bbb_transport;
        j[(0, 1)] = 0.;
        j[(1, 0)] = self.bbb_transport;
        j[(1, 1)] = -self.deg;
    }
}

impl ApplyNoise for Solution<f64, State<f64>> {
    fn apply_noise(&mut self, strength: f64) {
        self.y.iter_mut().for_each(|state| {
            state.plasma_rma +=
                strength * rand::rng().sample::<f64, StandardNormal>(StandardNormal);
        });
    }
}

#[cfg(test)]
mod tests {
    use crate::solve::{SolutionAccess, Solve};

    use super::*;
    use differential_equations::methods::ExplicitRungeKutta;

    const T0: f64 = 0.;
    const TF: f64 = 504.;
    const DT: f64 = 1.;

    #[test]
    fn default_simulation() {
        let model = Model::default();
        let mut solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(T0, TF, DT, State::default(), &mut solver);

        assert!(solution.is_ok());
        let mut unwrapped_solution = solution.unwrap();
        assert!(unwrapped_solution.plasma_rma().is_ok());
        assert!(unwrapped_solution.plasma_dox().is_err());

        // apply noise
        unwrapped_solution.apply_noise(0.1);
    }

    #[test]
    fn custom_rates() {
        let model = Model::new(0.5, 0.02, 0.7, 0.005);
        let mut solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(T0, TF, DT, State::default(), &mut solver);

        assert!(solution.is_ok());
    }

    #[test]
    fn builder_pattern() -> Result<(), Box<dyn std::error::Error>> {
        let model = Model::builder()
            .prod(0.5)
            .freq(0.02)
            .bbb_transport(0.7)
            .build()?;
        let mut solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(T0, TF, DT, State::default(), &mut solver);

        assert!(solution.is_ok());
        Ok(())
    }
}
