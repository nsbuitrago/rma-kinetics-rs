//! RMA Kinetics is a library of synthetic serum reporter models and associated
//! methods for simulation. The synthetic serum reporters modeled here are
//! specifically the Released Markers of Activity or RMAs.
//!
//! For a detailed description of the models, see the accompanying [paper]().
//! The original work described there was implemented in Python. For better performance,
//! we have since developed this library to be used directly in Rust or Python.
//!
//! ## Models
//!
//! This crate supports four core models listed below:
//! 1. Constitutive - a constitutively expressed synthetic serum reporter
//! 2. TetOff - serum reporter expressed under the tetracycline responsive operator
//! 3. Chemogenetic - neuronal activity induced + doxycycline gated serum reporter expression
//! 4. Oscillating - an artifically oscillating reporter for proxies of rapidly changing gene expression monitoring
//!
//! ## Getting Started

pub mod models;
mod solve;

pub use solve::Solve;

#[cfg(feature = "py")]
use pyo3::prelude::*;

/// RMA kinetics Python module
#[cfg(feature = "py")]
#[pymodule]
mod _rma_kinetics {
    #[pymodule_export]
    use super::py_models;
}

/// Kinetic models Python module
#[cfg(feature = "py")]
#[pymodule(name = "models")]
mod py_models {
    #[pymodule_export]
    use super::py_constitutive;
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
