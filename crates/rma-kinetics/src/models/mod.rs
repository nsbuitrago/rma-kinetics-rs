//! Released Marker of Activity Kinetic models
//!
//! This module contains the kinetic models for the Released Markers of Activity or RMAs.
//!
//! ## Models
//!
//! The following models are supported:
//! - Constitutive - a constitutively expressed synthetic serum reporter
//! - TetOff - serum reporter expressed under the tetracycline responsive operator
//! - Chemogenetic - neuronal activity induced + doxycycline gated serum reporter expression
//! - Dox - doxycycline pharmacokinetic model
//! - CNO - clozapine-N-oxide/clozapine pharmacokinetic model

pub mod cno;
pub mod constitutive;
pub mod dox;
pub mod tetoff;
