use differential_equations::{derive::State as StateTrait, ode::ODE, prelude::Matrix};

#[cfg(feature = "py")]
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pymethods};

#[cfg(feature = "py")]
use rma_kinetics_derive::PySolve;

use rma_kinetics_derive::Solve;

/// Constitutive model state.
#[derive(StateTrait)]
pub struct State<T> {
    pub brain_rma: T,
    pub plasma_rma: T,
}

impl State<f64> {
    /// Get a constitutive model state where brain and plasma RMA concentration
    /// are set to 0.
    pub fn zeros() -> Self {
        Self {
            brain_rma: 0.,
            plasma_rma: 0.,
        }
    }

    /// Create a new constitutive model state given brain and plasma RMA concentrations.
    pub fn new(brain_rma: f64, plasma_rma: f64) -> Self {
        Self {
            brain_rma,
            plasma_rma,
        }
    }
}

impl Default for State<f64> {
    /// Default constitutive model state where brain and plasma RMA concentration
    /// are set to 0.
    fn default() -> Self {
        State::zeros()
    }
}

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

/// Default constitutive RMA production rate.
const DEFAULT_PROD: f64 = 0.2;
/// Default constitutive RMA blood-brain barrier transport rate.
const DEFAULT_BBB_TRANSPORT: f64 = 0.6;
/// Default constitutive RMA degradation rate.
const DEFAULT_DEG: f64 = 0.007;

/// Constitutive RMA expression model.
#[cfg_attr(feature = "py", pyclass)]
#[cfg_attr(feature = "py", derive(PySolve))]
#[cfg_attr(feature = "py", py_solve(variant = "Constitutive"))]
#[derive(Solve)]
pub struct Model {
    prod: f64,
    bbb_transport: f64,
    deg: f64,
}

#[cfg(feature = "py")]
#[pymethods]
impl Model {
    /// Create a new constitutive expression model given RMA production, blood-brain
    /// barrier transport, and degradation rates.
    #[new]
    #[pyo3(signature = (prod=DEFAULT_PROD, bbb_transport=DEFAULT_BBB_TRANSPORT, deg=DEFAULT_DEG))]
    pub fn create(prod: f64, bbb_transport: f64, deg: f64) -> Self {
        Self {
            prod,
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
            Err(e) => Err(PyValueError::new_err("Failed to solve")), // TODO: add context from e
        }
    }
}

impl Model {
    /// Create a new constitutive expression model given RMA production, blood-brain
    /// barrier transport, and degradation rates.
    pub fn new(prod: f64, bbb_transport: f64, deg: f64) -> Self {
        Self {
            prod,
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
    /// Create a new constitutive model instance with the default parameters
    /// for CA1 hippocampus expression driven by a human-synapsin promoter.
    fn default() -> Self {
        ModelBuilder::default().build()
    }
}

impl ODE<f64, State<f64>> for Model {
    /// System of differential equations describing constitutive RMA expression
    /// in the brain tissue and blood-brain barrier transport to the plasma.
    fn diff(&self, _t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        let brain_efflux = self.bbb_transport * y.brain_rma;
        dydt.brain_rma = self.prod - brain_efflux;
        dydt.plasma_rma = brain_efflux - (self.deg * y.plasma_rma);
    }

    fn jacobian(&self, _t: f64, _y: &State<f64>, j: &mut Matrix<f64>) {
        j[(0, 0)] = -self.bbb_transport;
        j[(0, 1)] = 0.;
        j[(1, 0)] = self.bbb_transport;
        j[(1, 1)] = -self.deg;
    }
}

/// Constitutive expression model builder.
pub struct ModelBuilder {
    pub prod: f64,
    pub bbb_transport: f64,
    pub deg: f64,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            prod: DEFAULT_PROD,
            bbb_transport: DEFAULT_BBB_TRANSPORT,
            deg: DEFAULT_DEG,
        }
    }
}

impl ModelBuilder {
    /// Create a new constitutive model builder instance.
    pub fn new() -> Self {
        ModelBuilder::default()
    }

    /// Set RMA production rate (concentration/time)
    pub fn prod_rate(&mut self, rate: f64) -> &Self {
        self.prod = rate;
        self
    }

    /// Set the blood-brain barrier transport rate (1/time)
    pub fn bbb_transport_rate(&mut self, rate: f64) -> &Self {
        self.bbb_transport = rate;
        self
    }

    /// Set the RMA degradation rate (1/time)
    pub fn deg_rate(&mut self, rate: f64) -> &Self {
        self.deg = rate;
        self
    }

    /// Build the constitutive expression model
    pub fn build(&self) -> Model {
        Model {
            prod: self.prod,
            bbb_transport: self.bbb_transport,
            deg: self.deg,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::solve::Solve;

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
    }

    #[test]
    fn custom_rates() {
        let model = Model::new(0.5, 0.7, 0.005);
        let mut solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(T0, TF, DT, State::default(), &mut solver);

        assert!(solution.is_ok());
    }

    #[test]
    fn builder_pattern() {
        let model = Model::builder().prod_rate(0.5).build();
        let mut solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(T0, TF, DT, State::default(), &mut solver);

        assert!(solution.is_ok());
    }
}
