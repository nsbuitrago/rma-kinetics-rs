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
pub use crate::models::oscillation;
#[cfg(feature = "py")]
pub use crate::models::tetoff;

#[derive(Debug)]
pub enum SpeciesAccessError {
    NoBrainRMA,
    NoPlasmaRMA,
    NoTta,
    NoPlasmaDox,
    NoBrainDox,
    NoDreadd,
    NoPeritonealCno,
    NoPlasmaCno,
    NoBrainCno,
    NoPlasmaClz,
    NoBrainClz,
}

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
    Oscillation(Solution<f64, oscillation::State<f64>>),
}

/// Trait for accessing the species vectors from Solution types with different State types.
pub trait SolutionAccess {
    fn brain_rma(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn plasma_rma(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn tta(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn plasma_dox(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn brain_dox(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn dreadd(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn peritoneal_cno(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn plasma_cno(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn brain_cno(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn plasma_clz(&self) -> Result<Vec<f64>, SpeciesAccessError>;
    fn brain_clz(&self) -> Result<Vec<f64>, SpeciesAccessError>;
}

/// Macro to implement SolutionAccess for models that only have brain_rma and plasma_rma.
/// Used by constitutive and oscillation models.
#[macro_export]
macro_rules! impl_solution_access_basic_rma {
    ($solution_type:ty, $state_type:ty) => {
        impl crate::solve::SolutionAccess for $solution_type {
            fn brain_rma(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Ok(self
                    .y
                    .iter()
                    .map(|state| state.brain_rma)
                    .collect::<Vec<f64>>())
            }

            fn plasma_rma(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Ok(self
                    .y
                    .iter()
                    .map(|state| state.plasma_rma)
                    .collect::<Vec<f64>>())
            }

            fn tta(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Err(crate::solve::SpeciesAccessError::NoTta)
            }
            fn plasma_dox(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Err(crate::solve::SpeciesAccessError::NoPlasmaDox)
            }
            fn brain_dox(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Err(crate::solve::SpeciesAccessError::NoBrainDox)
            }
            fn dreadd(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Err(crate::solve::SpeciesAccessError::NoDreadd)
            }
            fn peritoneal_cno(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Err(crate::solve::SpeciesAccessError::NoPeritonealCno)
            }
            fn plasma_cno(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Err(crate::solve::SpeciesAccessError::NoPlasmaCno)
            }
            fn brain_cno(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Err(crate::solve::SpeciesAccessError::NoBrainCno)
            }
            fn plasma_clz(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Err(crate::solve::SpeciesAccessError::NoPlasmaClz)
            }
            fn brain_clz(&self) -> Result<Vec<f64>, crate::solve::SpeciesAccessError> {
                Err(crate::solve::SpeciesAccessError::NoBrainClz)
            }
        }
    };
}

pub trait ApplyNoise {
    fn apply_noise(&mut self, strength: f64);
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
            InnerSolution::Oscillation(s) => &s.$field,
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
            InnerSolution::Constitutive(s) => s.plasma_rma().unwrap(),
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "plasma RMA is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(s) => s.plasma_rma().unwrap(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "plasma RMA is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s.plasma_rma().unwrap(),
            InnerSolution::Oscillation(s) => s.plasma_rma().unwrap(),
        };

        Ok(PyArray1::from_vec(py, plasma_rma))
    }

    /// Get brain RMA array.
    #[getter]
    fn brain_rma<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let brain_rma = match &self.inner {
            InnerSolution::Constitutive(s) => s.brain_rma().unwrap(),
            InnerSolution::Dox(_) => {
                return Err(PyValueError::new_err(
                    "brain RMA is not available for the dox model",
                ));
            }
            InnerSolution::TetOff(s) => s.brain_rma().unwrap(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "brain RMA is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s.brain_rma().unwrap(),
            InnerSolution::Oscillation(s) => s.brain_rma().unwrap(),
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
            InnerSolution::TetOff(s) => s.tta().unwrap(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "tTA is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s.tta().unwrap(),
            InnerSolution::Oscillation(_) => {
                return Err(PyValueError::new_err(
                    "tTA is not available for the oscillation model",
                ));
            }
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
            InnerSolution::Dox(s) => s.plasma_dox().unwrap(),
            InnerSolution::TetOff(s) => s.plasma_dox().unwrap(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "plasma dox is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s.plasma_dox().unwrap(),
            InnerSolution::Oscillation(_) => {
                return Err(PyValueError::new_err(
                    "plasma dox is not available for the oscillation model",
                ));
            }
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
            InnerSolution::Dox(s) => s.brain_dox().unwrap(),
            InnerSolution::TetOff(s) => s.brain_dox().unwrap(),
            InnerSolution::CNO(_) => {
                return Err(PyValueError::new_err(
                    "brain dox is not available for the cno model",
                ));
            }
            InnerSolution::Chemogenetic(s) => s.brain_dox().unwrap(),
            InnerSolution::Oscillation(_) => {
                return Err(PyValueError::new_err(
                    "brain dox is not available for the oscillation model",
                ));
            }
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
            InnerSolution::Chemogenetic(s) => s.dreadd().unwrap(),
            InnerSolution::Oscillation(_) => {
                return Err(PyValueError::new_err(
                    "dreadd is not available for the oscillation model",
                ));
            }
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
            InnerSolution::CNO(s) => s.peritoneal_cno().unwrap(),
            InnerSolution::Chemogenetic(s) => s.peritoneal_cno().unwrap(),
            InnerSolution::Oscillation(_) => {
                return Err(PyValueError::new_err(
                    "peritoneal CNO is not available for the oscillation model",
                ));
            }
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
            InnerSolution::CNO(s) => s.plasma_cno().unwrap(),
            InnerSolution::Chemogenetic(s) => s.plasma_cno().unwrap(),
            InnerSolution::Oscillation(_) => {
                return Err(PyValueError::new_err(
                    "plasma CNO is not available for the oscillation model",
                ));
            }
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
            InnerSolution::CNO(s) => s.brain_cno().unwrap(),
            InnerSolution::Chemogenetic(s) => s.brain_cno().unwrap(),
            InnerSolution::Oscillation(_) => {
                return Err(PyValueError::new_err(
                    "brain CNO is not available for the oscillation model",
                ));
            }
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
            InnerSolution::CNO(s) => s.plasma_clz().unwrap(),
            InnerSolution::Chemogenetic(s) => s.plasma_clz().unwrap(),
            InnerSolution::Oscillation(_) => {
                return Err(PyValueError::new_err(
                    "plasma CLZ is not available for the oscillation model",
                ));
            }
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
            InnerSolution::CNO(s) => s.brain_clz().unwrap(),
            InnerSolution::Chemogenetic(s) => s.brain_clz().unwrap(),
            InnerSolution::Oscillation(_) => {
                return Err(PyValueError::new_err(
                    "brain CLZ is not available for the oscillation model",
                ));
            }
        };

        Ok(PyArray1::from_vec(py, brain_clz))
    }

    /// Returns the elapsed time in seconds
    fn elapsed_time(&self) -> f64 {
        self.inner.elapsed()
    }

    /// Applies standard normal noise of a given scale to the plasma RMA array.
    /// Only available for the oscillation model.
    fn apply_noise(&mut self, scale: f64) -> PyResult<()> {
        match &mut self.inner {
            InnerSolution::Oscillation(s) => s.apply_noise(scale),
            _ => {
                return Err(PyValueError::new_err(
                    "apply_noise is not available for this model",
                ));
            }
        }

        Ok(())
    }
}
