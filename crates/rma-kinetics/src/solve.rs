#[cfg(feature = "py")]
use pyo3::{Bound, FromPyObject, Python, pyclass, pymethods};

#[cfg(feature = "py")]
use numpy::PyArray1;

use differential_equations::{
    error::Error, interpolate::Interpolation, ode::OrdinaryNumericalMethod, solution::Solution,
    traits,
};

#[cfg(feature = "py")]
pub use crate::models::constitutive;

pub trait Solve {
    type State: traits::State<f64>;

    fn solve<S>(
        &self,
        t0: f64,
        tf: f64,
        sample_rate: f64,
        init_state: Self::State,
        solver: &mut S,
    ) -> Result<Solution<f64, Self::State>, Error<f64, Self::State>>
    where
        S: OrdinaryNumericalMethod<f64, Self::State> + Interpolation<f64, Self::State>;
}

#[cfg(feature = "py")]
pub trait PySolve {
    type State: traits::State<f64>;

    fn solve(
        &self,
        t0: f64,
        tf: f64,
        dt: f64,
        init_state: Self::State,
        solver: PySolver,
    ) -> Result<PySolution, Error<f64, Self::State>>;
}

#[cfg(feature = "py")]
#[derive(FromPyObject)]
pub struct PySolver {
    pub rtol: f64,
    pub atol: f64,
    pub dt0: f64,
    pub min_dt: f64,
    pub max_dt: f64,
    pub max_steps: usize,
    pub max_rejected_steps: usize,
    pub safety_factor: f64,
    pub min_scale: f64,
    pub max_scale: f64,
    pub solver_type: String,
}

#[cfg(feature = "py")]
pub enum InnerSolution {
    Constitutive(Solution<f64, constitutive::State<f64>>),
}

// A macro to access any field that exists on ALL InnerSolution variants
#[cfg(feature = "py")]
macro_rules! access_field {
    ($self:expr, $field:ident) => {
        match $self {
            InnerSolution::Constitutive(s) => &s.$field,
        }
    };
}

// A macro to access plasma RMA on InnerSolution variants
#[cfg(feature = "py")]
macro_rules! get_common_species {
    ($self:expr, $species:ident) => {{
        let ys = access_field!($self, y);
        ys.iter().map(|state| state.$species).collect::<Vec<f64>>()
    }};
}

#[cfg(feature = "py")]
impl InnerSolution {
    fn ts(&self) -> &Vec<f64> {
        access_field!(self, t)
    }

    fn elapsed(&self) -> f64 {
        access_field!(self, timer).elapsed()
    }
}

#[cfg(feature = "py")]
#[pyclass(name = "Solution")]
pub struct PySolution {
    pub inner: InnerSolution,
}

#[cfg(feature = "py")]
#[pymethods]
impl PySolution {
    /// Get time points.
    #[getter]
    fn ts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let ts = self.inner.ts().to_vec();
        PyArray1::from_vec(py, ts)
    }

    /// Get plasma RMA array.
    #[getter]
    fn plasma_rma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let species = get_common_species!(&self.inner, plasma_rma);
        PyArray1::from_vec(py, species)
    }

    /// Returns the elapsed time in seconds
    fn elapsed_time(&self) -> f64 {
        self.inner.elapsed()
    }
}
