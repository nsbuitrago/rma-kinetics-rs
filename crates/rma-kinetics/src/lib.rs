//! RMA Kinetics is a library of synthetic serum reporter models and associated
//! simulation methods. The synthetic serum reporters modeled here specifically are
//! the Released Markers of Activity or RMAs.
//!
//! For a detailed description of the models, see the model documentation or accompanying [paper](https://doi.org/10.1101/2025.11.17.688787).
//!
//! ## Models
//!
//! This crate supports the following core models:
//! 1. Constitutive - a constitutively expressed synthetic serum reporter
//! 2. TetOff - serum reporter expressed under the tetracycline responsive operator
//! 3. Chemogenetic - neuronal activity induced + doxycycline gated serum reporter expression
//! 4. Oscillating - an artifically oscillating reporter for proxies of rapidly changing gene expression monitoring
//!
//! The Tet-Off and Chemogenetic models additionally use doxycycline and clozapine pharmacokinetic models.
//!
//! ## Getting Started
//!
//! Each submodule in `models` contains at least a `Model` struct and a `State` struct.
//! For example, to model simple constitutive marker expression,
//!
//! ```rust
//! use rma_kinetics::{models::constitutive, Solve};
//! use differential_equations::methods::ExplicitRungeKutta;
//!
//! let model = constitutive::Model::default();
//! let init_state = constitutive::State::zeros();
//! let mut solver = ExplicitRungeKutta::dopri5();
//!
//! let solution = model.solve(0., 100., 1., init_state, &mut solver);
//! assert!(solution.is_ok());
//! ```
//!
//! The returned solution is the [`Solution`](https://docs.rs/differential-equations/latest/differential_equations/solution/struct.Solution.html)
//! struct from the `differential_equations` crate, where the `y` field is the corresponding `State` struct.

pub mod models;
pub mod pk;
mod solve;

pub use solve::{ApplyNoise, SolutionAccess, Solve};

#[cfg(feature = "py")]
use pyo3::prelude::*;

/// RMA kinetics Python module
#[cfg(feature = "py")]
#[pymodule]
mod _rma_kinetics {
    #[pymodule_export]
    use super::py_models;
    #[pymodule_export]
    use super::solve::PySolution;
}

/// Kinetic models Python module
#[cfg(feature = "py")]
#[pymodule(name = "models")]
mod py_models {
    #[pymodule_export]
    use super::py_chemogenetic;
    #[pymodule_export]
    use super::py_cno;
    #[pymodule_export]
    use super::py_constitutive;
    #[pymodule_export]
    use super::py_dox;
    #[pymodule_export]
    use super::py_oscillation;
    #[pymodule_export]
    use super::py_tetoff;
}

/// Constitutive model Python module
#[cfg(feature = "py")]
#[pymodule(name = "constitutive")]
mod py_constitutive {
    #[pymodule_export]
    use super::models::constitutive::Model;
    #[pymodule_export]
    use super::models::constitutive::PyState;
}

/// Oscillation model Python module
#[cfg(feature = "py")]
#[pymodule(name = "oscillation")]
mod py_oscillation {
    #[pymodule_export]
    use super::models::oscillation::Model;
    #[pymodule_export]
    use super::models::oscillation::PyState;
}

/// TetOff model python module
#[cfg(feature = "py")]
#[pymodule(name = "tetoff")]
mod py_tetoff {
    #[pymodule_export]
    use super::models::tetoff::Model;
    #[pymodule_export]
    use super::models::tetoff::PyState;
}

// Dox model python module
#[cfg(feature = "py")]
#[pymodule(name = "dox")]
mod py_dox {
    #[pymodule_export]
    use super::models::dox::AccessPeriod;
    #[pymodule_export]
    use super::models::dox::Model;
    #[pymodule_export]
    use super::models::dox::PyState;
    #[pymodule_export]
    use super::models::dox::create_dox_schedule;
}

// CNO model python module
#[cfg(feature = "py")]
#[pymodule(name = "cno")]
mod py_cno {
    #[pymodule_export]
    use super::models::cno::Dose;
    #[pymodule_export]
    use super::models::cno::Model;
    #[pymodule_export]
    use super::models::cno::PyState;
    #[pymodule_export]
    use super::models::cno::create_cno_schedule;
}

// Chemogenetic model python module
#[cfg(feature = "py")]
#[pymodule(name = "chemogenetic")]
mod py_chemogenetic {
    #[pymodule_export]
    use super::models::chemogenetic::Model;
    #[pymodule_export]
    use super::models::chemogenetic::PyState;
}
