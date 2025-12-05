#[cfg(feature = "py")]
use pyo3::{Bound, FromPyObject, PyResult, Python, exceptions::PyValueError, pyclass, pymethods};

#[cfg(feature = "py")]
use numpy::PyArray1;

use differential_equations::{
    error::Error, interpolate::Interpolation, ode::OrdinaryNumericalMethod, solution::Solution,
    traits,
};

#[cfg(feature = "py")]
pub use crate::models::chemogenetic;
#[cfg(feature = "py")]
pub use crate::models::cno;
#[cfg(feature = "py")]
pub use crate::models::constitutive;
#[cfg(feature = "py")]
pub use crate::models::dox;
#[cfg(feature = "py")]
pub use crate::models::tetoff;

/// Solve trait for kinetic models.
pub trait Solve {
    type State: traits::State<f64>;

    fn solve<S>(
        &self,
        t0: f64,
        tf: f64,
        dt: f64,
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
    Dox(Solution<f64, dox::State<f64>>),
    TetOff(Solution<f64, tetoff::State<f64>>),
    CNO(Solution<f64, cno::State<f64>>),
    Chemogenetic(Solution<f64, chemogenetic::State<f64>>),
}

/// Trait for accessing the states vector from Solution types with different State types.
/// This allows type-safe access to the `y` field which contains different State types
/// across different Solution variants.
#[cfg(feature = "py")]
trait SolutionAccess {
    /// Get a reference to the states vector.
    /// The return type is a reference to Vec<State<f64>> where State varies by implementation.
    fn states(&self) -> &Vec<Self::StateType>
    where
        Self: Sized;

    /// Associated type for the State type in this Solution.
    type StateType: traits::State<f64>;
}

#[cfg(feature = "py")]
impl SolutionAccess for Solution<f64, constitutive::State<f64>> {
    type StateType = constitutive::State<f64>;

    fn states(&self) -> &Vec<constitutive::State<f64>> {
        &self.y
    }
}

#[cfg(feature = "py")]
impl SolutionAccess for Solution<f64, dox::State<f64>> {
    type StateType = dox::State<f64>;

    fn states(&self) -> &Vec<dox::State<f64>> {
        &self.y
    }
}

#[cfg(feature = "py")]
impl SolutionAccess for Solution<f64, tetoff::State<f64>> {
    type StateType = tetoff::State<f64>;

    fn states(&self) -> &Vec<tetoff::State<f64>> {
        &self.y
    }
}

#[cfg(feature = "py")]
impl SolutionAccess for Solution<f64, cno::State<f64>> {
    type StateType = cno::State<f64>;

    fn states(&self) -> &Vec<cno::State<f64>> {
        &self.y
    }
}

#[cfg(feature = "py")]
impl SolutionAccess for Solution<f64, chemogenetic::State<f64>> {
    type StateType = chemogenetic::State<f64>;

    fn states(&self) -> &Vec<chemogenetic::State<f64>> {
        &self.y
    }
}

// A macro to access fields that exist on ALL InnerSolution variants with the same type.
// For fields with different types (like the `y` field containing different State types),
// use the SolutionAccess trait's `states()` method instead.
#[cfg(feature = "py")]
macro_rules! access_field {
    ($self:expr, $field:ident) => {
        match $self {
            InnerSolution::Constitutive(s) => &s.$field,
            InnerSolution::Dox(s) => &s.$field,
            InnerSolution::TetOff(s) => &s.$field,
            InnerSolution::CNO(s) => &s.$field,
            InnerSolution::Chemogenetic(s) => &s.$field,
        }
    };
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
    fn plasma_rma<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // let species = get_common_species!(&self.inner, plasma_rma);
        let plasma_rma = match &self.inner {
            InnerSolution::Constitutive(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_rma)
                .collect::<Vec<f64>>(),
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "plasma RMA is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_rma)
                .collect::<Vec<f64>>(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "plasma RMA is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_rma)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, plasma_rma))
    }

    /// Get brain RMA array.
    #[getter]
    fn brain_rma<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let brain_rma = match &self.inner {
            InnerSolution::Constitutive(s) => s
                .states()
                .iter()
                .map(|state| state.brain_rma)
                .collect::<Vec<f64>>(),
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "brain RMA is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(s) => s
                .states()
                .iter()
                .map(|state| state.brain_rma)
                .collect::<Vec<f64>>(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "brain RMA is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.brain_rma)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, brain_rma))
    }

    /// Get tTA array.
    #[getter]
    fn tta<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let tta = match &self.inner {
            InnerSolution::Constitutive(_) => {
                return Err(PyValueError::new_err(
                    "tTA is not available for the constitutive model",
                ));
            }
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "tTA is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(s) => s
                .states()
                .iter()
                .map(|state| state.tta)
                .collect::<Vec<f64>>(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "tTA is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.tta)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, tta))
    }

    /// Get plasma dox array.
    #[getter]
    fn plasma_dox<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let plasma_dox = match &self.inner {
            InnerSolution::Constitutive(_) => {
                return Err(PyValueError::new_err(
                    "plasma dox is not available for the constitutive model",
                ));
            }
            InnerSolution::Dox(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_dox)
                .collect::<Vec<f64>>(),
            InnerSolution::TetOff(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_dox)
                .collect::<Vec<f64>>(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "plasma dox is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_dox)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, plasma_dox))
    }

    /// Get brain dox array.
    #[getter]
    fn brain_dox<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let brain_dox = match &self.inner {
            InnerSolution::Constitutive(_) => {
                return Err(PyValueError::new_err(
                    "brain dox is not available for the constitutive model",
                ));
            }
            InnerSolution::Dox(s) => s
                .states()
                .iter()
                .map(|state| state.brain_dox)
                .collect::<Vec<f64>>(),
            InnerSolution::TetOff(s) => s
                .states()
                .iter()
                .map(|state| state.brain_dox)
                .collect::<Vec<f64>>(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "brain dox is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.brain_dox)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, brain_dox))
    }

    /// Get dreadd array.
    #[getter]
    fn dreadd<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let dreadd = match &self.inner {
            InnerSolution::Constitutive(_) => {
                return Err(PyValueError::new_err(
                    "dreadd is not available for the constitutive model",
                ));
            }
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "dreadd is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(_) => {
                return Err(PyValueError::new_err(
                    "dreadd is not available for the tetoff model",
                ));
            }
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "dreadd is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.dreadd)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, dreadd))
    }

    /// Get peritoneal CNO array (returned as nmol).
    #[getter]
    fn peritoneal_cno<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let peritoneal_cno = match &self.inner {
            InnerSolution::Constitutive(_) => {
                return Err(PyValueError::new_err(
                    "peritoneal CNO is not available for the constitutive model",
                ));
            }
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "peritoneal CNO is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(_) => {
                return Err(PyValueError::new_err(
                    "peritoneal CNO is not available for the tetoff model",
                ));
            }
            InnerSolution::CNO(s) => s
                .states()
                .iter()
                .map(|state| state.peritoneal_cno)
                .collect::<Vec<f64>>(),
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.peritoneal_cno)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, peritoneal_cno))
    }

    /// Get plasma CNO array.
    #[getter]
    fn plasma_cno<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let plasma_cno = match &self.inner {
            InnerSolution::Constitutive(_) => {
                return Err(PyValueError::new_err(
                    "plasma CNO is not available for the constitutive model",
                ));
            }
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "plasma CNO is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(_) => {
                return Err(PyValueError::new_err(
                    "plasma CNO is not available for the tetoff model",
                ));
            }
            InnerSolution::CNO(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_cno)
                .collect::<Vec<f64>>(),
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_cno)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, plasma_cno))
    }

    /// Get brain CNO array.
    #[getter]
    fn brain_cno<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let brain_cno = match &self.inner {
            InnerSolution::Constitutive(_) => {
                return Err(PyValueError::new_err(
                    "brain CNO is not available for the constitutive model",
                ));
            }
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "brain CNO is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(_) => {
                return Err(PyValueError::new_err(
                    "brain CNO is not available for the tetoff model",
                ));
            }
            InnerSolution::CNO(s) => s
                .states()
                .iter()
                .map(|state| state.brain_cno)
                .collect::<Vec<f64>>(),
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.brain_cno)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, brain_cno))
    }

    /// Get plasma CLZ array.
    #[getter]
    fn plasma_clz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let plasma_clz = match &self.inner {
            InnerSolution::Constitutive(_) => {
                return Err(PyValueError::new_err(
                    "plasma CLZ is not available for the constitutive model",
                ));
            }
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "plasma CLZ is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(_) => {
                return Err(PyValueError::new_err(
                    "plasma CLZ is not available for the tetoff model",
                ));
            }
            InnerSolution::CNO(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_clz)
                .collect::<Vec<f64>>(),
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.plasma_clz)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, plasma_clz))
    }

    /// Get brain CLZ array.
    #[getter]
    fn brain_clz<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let brain_clz = match &self.inner {
            InnerSolution::Constitutive(_) => {
                return Err(PyValueError::new_err(
                    "brain CLZ is not available for the constitutive model",
                ));
            }
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "brain CLZ is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(_) => {
                return Err(PyValueError::new_err(
                    "brain CLZ is not available for the tetoff model",
                ));
            }
            InnerSolution::CNO(s) => s
                .states()
                .iter()
                .map(|state| state.brain_clz)
                .collect::<Vec<f64>>(),
            InnerSolution::Chemogenetic(s) => s
                .states()
                .iter()
                .map(|state| state.brain_clz)
                .collect::<Vec<f64>>(),
        };

        Ok(PyArray1::from_vec(py, brain_clz))
    }

    /// Returns the elapsed time in seconds
    fn elapsed_time(&self) -> f64 {
        self.inner.elapsed()
    }
}
