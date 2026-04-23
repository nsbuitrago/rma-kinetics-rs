//! Chemogenetic model.
//!
//! A model describing the genetic circuit for monitoring neuronal activity
//! with released markers of activity ([Lee et al., 2024](https://doi.org/10.1038/s41587-023-02087-x), [Buitrago et al., 2025](https://doi.org/10.1101/2025.11.17.688787))
//!
//! ## Usage
//!
//! This model makes use of the [CNO ](crate::models::cno::Model) and [Doxycycline](crate::models::dox::Model) pharmacokinetic models
//! to describe the dynamics of CNO/CLZ and doxycycline in the brain and plasma.
//!
//! ## Usage
//!
//! ```rust
//! use rma_kinetics::{models::{chemogenetic, cno}, Solve};
//! use differential_equations::methods::ExplicitRungeKutta;
//!
//! let dose = cno::Dose::new(0.03, 0.);
//! let cno_pk = cno::Model::builder().doses(vec![dose]).build()?;
//! let model = chemogenetic::Model::builder().cno_pk_model(cno_pk).build()?;
//! let init_state = chemogenetic::State::zeros();
//! let mut solver = ExplicitRungeKutta::dopri5();
//!
//! let solution = model.solve(0., 48., 1., init_state, &mut solver);
//! assert!(solution.is_ok());
//! Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod erasable;

use crate::{
    SolutionAccess, Solve,
    models::{
        cno::{CNOFields, CNOPKAccess, Dose, Model as CNOModel},
        dox::{DoxFields, Model as DoxModel},
    },
    pk::{DoseApplyingSolout, ScheduledStateUpdate},
    solve::SpeciesAccessError,
};

pub trait ChemogeneticCoreFields: DoxFields + CNOFields {
    fn tta(&self) -> f64;
    fn dreadd(&self) -> f64;
    fn tta_mut(&mut self) -> &mut f64;
    fn dreadd_mut(&mut self) -> &mut f64;
}
use derive_builder::Builder;
use differential_equations::{
    derive::State as StateTrait,
    error::Error,
    ode::{ODE, ODEProblem, OrdinaryNumericalMethod},
    prelude::{Interpolation, Solution},
};

#[cfg(feature = "py")]
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pymethods};

#[cfg(feature = "py")]
use differential_equations::methods::DiagonallyImplicitRungeKutta;

#[cfg(feature = "py")]
use crate::models::{cno::State as CNOState, dox::State as DoxState};

#[cfg(feature = "py")]
use numpy::{PyArray1, PyArray2, PyReadonlyArray1};

#[cfg(feature = "py")]
use pyo3::{Bound, Python};

#[cfg(feature = "py")]
use crate::solve::{InnerSolution, PySolution, PySolver};

#[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
use crate::solve::ToDataFrame;

#[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
use polars::{error::PolarsError, frame::DataFrame};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Chemogenetic model state.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(StateTrait, Builder)]
#[builder(derive(Debug))]
pub struct State<T> {
    pub brain_rma: T,
    pub plasma_rma: T,
    pub tta: T,
    pub plasma_dox: T,
    pub brain_dox: T,
    pub dreadd: T,
    pub peritoneal_cno: T,
    pub plasma_cno: T,
    pub brain_cno: T,
    pub plasma_clz: T,
    pub brain_clz: T,
}

impl State<f64> {
    /// Create a new chemogenetic model state where all concentrations are set to zero.
    pub fn zeros() -> Self {
        Self {
            brain_rma: 0.,
            plasma_rma: 0.,
            tta: 0.,
            plasma_dox: 0.,
            brain_dox: 0.,
            dreadd: 0.,
            peritoneal_cno: 0.,
            plasma_cno: 0.,
            brain_cno: 0.,
            plasma_clz: 0.,
            brain_clz: 0.,
        }
    }

    /// Create a new chemogenetic model state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        brain_rma: f64,
        plasma_rma: f64,
        tta: f64,
        plasma_dox: f64,
        brain_dox: f64,
        dreadd: f64,
        peritoneal_cno: f64,
        plasma_cno: f64,
        brain_cno: f64,
        plasma_clz: f64,
        brain_clz: f64,
    ) -> Self {
        Self {
            brain_rma,
            plasma_rma,
            tta,
            plasma_dox,
            brain_dox,
            dreadd,
            peritoneal_cno,
            plasma_cno,
            brain_cno,
            plasma_clz,
            brain_clz,
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for State<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "brain_rma={:.3}, plasma_rma={:.3}, tta={:.3}, plasma_dox={:.3}, brain_dox={:.3}, dreadd={:.3}, peritoneal_cno={:.3}, plasma_cno={:.3}, brain_cno={:.3}, plasma_clz={:.3}, brain_clz={:.3}",
            self.brain_rma,
            self.plasma_rma,
            self.tta,
            self.plasma_dox,
            self.brain_dox,
            self.dreadd,
            self.peritoneal_cno,
            self.plasma_cno,
            self.brain_cno,
            self.plasma_clz,
            self.brain_clz
        )
    }
}

impl SolutionAccess for Solution<f64, State<f64>> {
    fn brain_rma(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.brain_rma)
            .collect::<Vec<f64>>())
    }

    fn max_brain_rma(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, brain_rma))
    }

    fn plasma_rma(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.plasma_rma)
            .collect::<Vec<f64>>())
    }

    fn max_plasma_rma(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, plasma_rma))
    }

    fn tta(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self.y.iter().map(|state| state.tta).collect::<Vec<f64>>())
    }

    fn max_tta(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, tta))
    }

    fn plasma_dox(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.plasma_dox)
            .collect::<Vec<f64>>())
    }

    fn max_plasma_dox(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, plasma_dox))
    }

    fn brain_dox(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.brain_dox)
            .collect::<Vec<f64>>())
    }

    fn max_brain_dox(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, brain_dox))
    }

    fn dreadd(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.dreadd)
            .collect::<Vec<f64>>())
    }

    fn max_dreadd(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, dreadd))
    }

    fn peritoneal_cno(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.peritoneal_cno)
            .collect::<Vec<f64>>())
    }

    fn max_peritoneal_cno(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, peritoneal_cno))
    }

    fn plasma_cno(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.plasma_cno)
            .collect::<Vec<f64>>())
    }

    fn max_plasma_cno(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, plasma_cno))
    }

    fn brain_cno(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.brain_cno)
            .collect::<Vec<f64>>())
    }

    fn max_brain_cno(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, brain_cno))
    }

    fn plasma_clz(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.plasma_clz)
            .collect::<Vec<f64>>())
    }

    fn max_plasma_clz(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, plasma_clz))
    }

    fn brain_clz(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Ok(self
            .y
            .iter()
            .map(|state| state.brain_clz)
            .collect::<Vec<f64>>())
    }

    fn max_brain_clz(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, brain_clz))
    }

    fn plasma_tev(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoPlasmaTev)
    }

    fn max_plasma_tev(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoPlasmaTev)
    }
}

#[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
impl ToDataFrame for Solution<f64, State<f64>> {
    fn to_dataframe(self) -> Result<DataFrame, PolarsError> {
        use crate::struct_to_dataframe;

        struct_to_dataframe!(
            self,
            [
                brain_rma,
                plasma_rma,
                tta,
                plasma_dox,
                brain_dox,
                dreadd,
                peritoneal_cno,
                plasma_cno,
                brain_cno,
                plasma_clz,
                brain_clz
            ]
        )
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
    #[pyo3(signature = (brain_rma=0., plasma_rma=0., tta=0., plasma_dox=0., brain_dox=0., dreadd=0., peritoneal_cno=0., plasma_cno=0., brain_cno=0., plasma_clz=0., brain_clz=0.))]
    pub fn new(
        brain_rma: f64,
        plasma_rma: f64,
        tta: f64,
        plasma_dox: f64,
        brain_dox: f64,
        dreadd: f64,
        peritoneal_cno: f64,
        plasma_cno: f64,
        brain_cno: f64,
        plasma_clz: f64,
        brain_clz: f64,
    ) -> Self {
        Self {
            inner: State::new(
                brain_rma,
                plasma_rma,
                tta,
                plasma_dox,
                brain_dox,
                dreadd,
                peritoneal_cno,
                plasma_cno,
                brain_cno,
                plasma_clz,
                brain_clz,
            ),
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
    #[getter]
    fn get_tta(&self) -> f64 {
        self.inner.tta
    }
    #[getter]
    fn get_plasma_dox(&self) -> f64 {
        self.inner.plasma_dox
    }
    #[getter]
    fn get_brain_dox(&self) -> f64 {
        self.inner.brain_dox
    }
    #[getter]
    fn get_dreadd(&self) -> f64 {
        self.inner.dreadd
    }
    #[getter]
    fn get_peritoneal_cno(&self) -> f64 {
        self.inner.peritoneal_cno
    }
    #[getter]
    fn get_plasma_cno(&self) -> f64 {
        self.inner.plasma_cno
    }
    #[getter]
    fn get_brain_cno(&self) -> f64 {
        self.inner.brain_cno
    }
    #[getter]
    fn get_plasma_clz(&self) -> f64 {
        self.inner.plasma_clz
    }
    #[getter]
    fn get_brain_clz(&self) -> f64 {
        self.inner.brain_clz
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
    #[setter]
    fn set_tta(&mut self, value: f64) -> PyResult<()> {
        self.inner.tta = value;
        Ok(())
    }
    #[setter]
    fn set_plasma_dox(&mut self, value: f64) -> PyResult<()> {
        self.inner.plasma_dox = value;
        Ok(())
    }
    #[setter]
    fn set_brain_dox(&mut self, value: f64) -> PyResult<()> {
        self.inner.brain_dox = value;
        Ok(())
    }
    #[setter]
    fn set_dreadd(&mut self, value: f64) -> PyResult<()> {
        self.inner.dreadd = value;
        Ok(())
    }
    #[setter]
    fn set_peritoneal_cno(&mut self, value: f64) -> PyResult<()> {
        self.inner.peritoneal_cno = value;
        Ok(())
    }
    #[setter]
    fn set_plasma_cno(&mut self, value: f64) -> PyResult<()> {
        self.inner.plasma_cno = value;
        Ok(())
    }
    #[setter]
    fn set_brain_cno(&mut self, value: f64) -> PyResult<()> {
        self.inner.brain_cno = value;
        Ok(())
    }
    #[setter]
    fn set_plasma_clz(&mut self, value: f64) -> PyResult<()> {
        self.inner.plasma_clz = value;
        Ok(())
    }
    #[setter]
    fn set_brain_clz(&mut self, value: f64) -> PyResult<()> {
        self.inner.brain_clz = value;
        Ok(())
    }
}

#[cfg(feature = "py")]
impl std::fmt::Display for PyState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
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

impl CNOFields for State<f64> {
    fn peritoneal_cno(&self) -> f64 {
        self.peritoneal_cno
    }
    fn plasma_cno(&self) -> f64 {
        self.plasma_cno
    }
    fn brain_cno(&self) -> f64 {
        self.brain_cno
    }
    fn plasma_clz(&self) -> f64 {
        self.plasma_clz
    }
    fn brain_clz(&self) -> f64 {
        self.brain_clz
    }
    fn peritoneal_cno_mut(&mut self) -> &mut f64 {
        &mut self.peritoneal_cno
    }
    fn plasma_cno_mut(&mut self) -> &mut f64 {
        &mut self.plasma_cno
    }
    fn brain_cno_mut(&mut self) -> &mut f64 {
        &mut self.brain_cno
    }
    fn plasma_clz_mut(&mut self) -> &mut f64 {
        &mut self.plasma_clz
    }
    fn brain_clz_mut(&mut self) -> &mut f64 {
        &mut self.brain_clz
    }
}

impl ChemogeneticCoreFields for State<f64> {
    fn tta(&self) -> f64 {
        self.tta
    }

    fn dreadd(&self) -> f64 {
        self.dreadd
    }

    fn tta_mut(&mut self) -> &mut f64 {
        &mut self.tta
    }

    fn dreadd_mut(&mut self) -> &mut f64 {
        &mut self.dreadd
    }
}

impl ScheduledStateUpdate<State<f64>> for Dose {
    fn time(&self) -> f64 {
        self.time
    }

    fn apply(&self, state: &mut State<f64>) {
        state.peritoneal_cno += self.nmol;
    }
}

const DEFAULT_RMA_PROD: f64 = 0.428;
const DEFAULT_LEAKY_RMA_PROD: f64 = 7.01e-3;
const DEFAULT_RMA_BBB_TRANSPORT: f64 = 0.727;
const DEFAULT_RMA_DEG: f64 = 5.5e-3;
const DEFAULT_TTA_PROD: f64 = 12.46;
const DEFAULT_LEAKY_TTA_PROD: f64 = 1.22e-1;
const DEFAULT_TTA_DEG: f64 = 2.81e-2;
const DEFAULT_TTA_KD: f64 = 4.19;
const DEFAULT_TTA_COOPERATIVITY: f64 = 2.;
const DEFAULT_DOX_TTA_KD: f64 = 5.27;
const DEFAULT_CNO_EC50: f64 = 7.94;
const DEFAULT_CLZ_EC50: f64 = 4.34;
const DEFAULT_CNO_COOPERATIVITY: f64 = 1.;
const DEFAULT_CLZ_COOPERATIVITY: f64 = 1.;
const DEFAULT_DREADD_PROD: f64 = 8.05;
const DEFAULT_DREADD_DEG: f64 = 1.;
const DEFAULT_DREADD_EC50: f64 = 6.79;
const DEFAULT_DREADD_COOPERATIVITY: f64 = 1.;

#[cfg_attr(feature = "py", pyclass)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Builder, Debug)]
#[builder(derive(Debug))]
pub struct Model {
    #[builder(default = "DEFAULT_RMA_PROD")]
    pub rma_prod: f64,
    #[builder(default = "DEFAULT_LEAKY_RMA_PROD")]
    pub leaky_rma_prod: f64,
    #[builder(default = "DEFAULT_RMA_BBB_TRANSPORT")]
    pub rma_bbb_transport: f64,
    #[builder(default = "DEFAULT_RMA_DEG")]
    pub rma_deg: f64,
    #[builder(default = "DEFAULT_TTA_PROD")]
    pub tta_prod: f64,
    #[builder(default = "DEFAULT_LEAKY_TTA_PROD")]
    pub leaky_tta_prod: f64,
    #[builder(default = "DEFAULT_TTA_DEG")]
    pub tta_deg: f64,
    #[builder(default = "DEFAULT_TTA_KD")]
    pub tta_kd: f64,
    #[builder(default = "DEFAULT_TTA_COOPERATIVITY")]
    pub tta_cooperativity: f64,
    #[builder(default = "DoxModel::default()")]
    pub dox_pk_model: DoxModel,
    #[builder(default = "DEFAULT_DOX_TTA_KD")]
    pub dox_tta_kd: f64,
    #[builder(default = "CNOModel::default()")]
    pub cno_pk_model: CNOModel,
    #[builder(default = "DEFAULT_CNO_EC50")]
    pub cno_ec50: f64,
    #[builder(default = "DEFAULT_CLZ_EC50")]
    pub clz_ec50: f64,
    #[builder(default = "DEFAULT_CNO_COOPERATIVITY")]
    pub cno_cooperativity: f64,
    #[builder(default = "DEFAULT_CLZ_COOPERATIVITY")]
    pub clz_cooperativity: f64,
    #[builder(default = "DEFAULT_DREADD_PROD")]
    pub dreadd_prod: f64,
    #[builder(default = "DEFAULT_DREADD_DEG")]
    pub dreadd_deg: f64,
    #[builder(default = "DEFAULT_DREADD_EC50")]
    pub dreadd_ec50: f64,
    #[builder(default = "DEFAULT_DREADD_COOPERATIVITY")]
    pub dreadd_cooperativity: f64,
}

impl Default for Model {
    fn default() -> Self {
        ModelBuilder::default().build().unwrap()
    }
}

impl CNOPKAccess for Model {
    fn get_doses(&self) -> &Vec<Dose> {
        &self.cno_pk_model.doses
    }
}

impl Model {
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
    }

    pub fn diff_with<S: ChemogeneticCoreFields>(&self, t: f64, y: &S, dydt: &mut S) {
        diff_chemogenetic_core(
            t,
            y,
            dydt,
            &self.dox_pk_model,
            &self.cno_pk_model,
            self.tta_prod,
            self.leaky_tta_prod,
            self.tta_deg,
            self.cno_ec50,
            self.clz_ec50,
            self.cno_cooperativity,
            self.clz_cooperativity,
            self.dreadd_prod,
            self.dreadd_deg,
            self.dreadd_ec50,
            self.dreadd_cooperativity,
        );
    }
}

fn diff_chemogenetic_core<S: ChemogeneticCoreFields>(
    t: f64,
    y: &S,
    dydt: &mut S,
    dox_pk_model: &DoxModel,
    cno_pk_model: &CNOModel,
    tta_prod: f64,
    leaky_tta_prod: f64,
    tta_deg: f64,
    cno_ec50: f64,
    clz_ec50: f64,
    cno_cooperativity: f64,
    clz_cooperativity: f64,
    dreadd_prod: f64,
    dreadd_deg: f64,
    dreadd_ec50: f64,
    dreadd_cooperativity: f64,
) {
    dox_pk_model.diff_with(t, y, dydt);
    cno_pk_model.diff_with(t, y, dydt);

    let cno_ec50_hill =
        (y.brain_cno() / cno_pk_model.cno_brain_vd / cno_ec50).powf(cno_cooperativity);
    let clz_ec50_hill =
        (y.brain_clz() / cno_pk_model.clz_brain_vd / clz_ec50).powf(clz_cooperativity);
    let active_dreadd_frac = fractional_activation(cno_ec50_hill + clz_ec50_hill);
    let dreadd_mod = (active_dreadd_frac * y.dreadd() / dreadd_ec50).powf(dreadd_cooperativity);

    *dydt.tta_mut() = saturating_mix(leaky_tta_prod, tta_prod, dreadd_mod) - (tta_deg * y.tta());
    *dydt.dreadd_mut() = dreadd_prod - (dreadd_deg * y.dreadd());
}

#[inline]
fn saturating_mix(leaky: f64, induced: f64, activation: f64) -> f64 {
    (leaky + (induced * activation)) / (1.0 + activation)
}

#[inline]
#[cfg(feature = "py")]
fn saturating_mix_derivative(leaky: f64, induced: f64, activation: f64) -> f64 {
    (induced - leaky) / ((1.0 + activation) * (1.0 + activation))
}

#[inline]
fn fractional_activation(total_signal: f64) -> f64 {
    total_signal / (1.0 + total_signal)
}

#[inline]
#[cfg(feature = "py")]
fn fractional_activation_derivative(total_signal: f64) -> f64 {
    1.0 / ((1.0 + total_signal) * (1.0 + total_signal))
}

impl ODE<f64, State<f64>> for Model {
    fn diff(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        self.diff_with(t, y, dydt);

        // tet inducible RMA expression
        let active_tta = 1. / (1. + y.brain_dox / self.dox_tta_kd);
        let tta_hill = (active_tta * y.tta / self.tta_kd).powf(self.tta_cooperativity);
        dydt.brain_rma = saturating_mix(self.leaky_rma_prod, self.rma_prod, tta_hill)
            - (self.rma_bbb_transport * y.brain_rma);

        let brain_efflux = self.rma_bbb_transport * y.brain_rma;
        dydt.plasma_rma = brain_efflux - (self.rma_deg * y.plasma_rma);
    }
}

impl Solve for Model {
    type State = State<f64>;
    fn solve<S>(
        &self,
        t0: f64,
        tf: f64,
        dt: f64,
        init_state: Self::State,
        solver: &mut S,
    ) -> Result<Solution<f64, Self::State>, Error<f64, Self::State>>
    where
        S: OrdinaryNumericalMethod<f64, Self::State> + Interpolation<f64, Self::State>,
    {
        let mut adjusted_init_state = init_state;
        let scheduled_updates = self
            .cno_pk_model
            .doses
            .iter()
            .filter_map(|dose| {
                if (dose.time - t0).abs() < 1e-10 {
                    adjusted_init_state.peritoneal_cno += dose.nmol;
                    None
                } else {
                    Some(dose.clone())
                }
            })
            .collect::<Vec<Dose>>();

        let mut dosing_solout =
            DoseApplyingSolout::<State<f64>, Dose>::new(scheduled_updates, t0, tf, dt);
        let problem = ODEProblem::new(self, t0, tf, adjusted_init_state);
        let mut solution = problem.solout(&mut dosing_solout).solve(solver)?;

        // return concentrations using given Vd (except for peritoneal compartment)
        let y = solution
            .y
            .iter()
            .map(|state| State {
                brain_rma: state.brain_rma,
                plasma_rma: state.plasma_rma,
                tta: state.tta,
                plasma_dox: state.plasma_dox(),
                brain_dox: state.brain_dox(),
                dreadd: state.dreadd,
                peritoneal_cno: state.peritoneal_cno(),
                plasma_cno: state.plasma_cno() / self.cno_pk_model.cno_plasma_vd,
                brain_cno: state.brain_cno() / self.cno_pk_model.cno_brain_vd,
                plasma_clz: state.plasma_clz() / self.cno_pk_model.clz_plasma_vd,
                brain_clz: state.brain_clz() / self.cno_pk_model.clz_brain_vd,
            })
            .collect::<Vec<State<f64>>>();
        solution.y = y;

        Ok(solution)
    }
}

#[cfg(feature = "py")]
const INFERENCE_N_STATE: usize = 4;
#[cfg(feature = "py")]
const INFERENCE_N_PARAMS: usize = 13;
#[cfg(feature = "py")]
const INFERENCE_N_GLOBAL_PARAMS: usize = 11;
#[cfg(feature = "py")]
const INFERENCE_DREADD_DEG: f64 = 1.0;

#[cfg(feature = "py")]
const IDX_LOG_PROD_LOCAL: usize = 0;
#[cfg(feature = "py")]
const IDX_LOG_LEAKY_PROD_LOCAL: usize = 1;
#[cfg(feature = "py")]
const IDX_LOG_BBB: usize = 2;
#[cfg(feature = "py")]
const IDX_LOG_DEG: usize = 3;
#[cfg(feature = "py")]
const IDX_LOG_TTA_PROD: usize = 4;
#[cfg(feature = "py")]
const IDX_LOG_TTA_LEAKY_PROD: usize = 5;
#[cfg(feature = "py")]
const IDX_LOG_TTA_DEG: usize = 6;
#[cfg(feature = "py")]
const IDX_LOG_TTA_KD: usize = 7;
#[cfg(feature = "py")]
const IDX_LOG_DOX_KD: usize = 8;
#[cfg(feature = "py")]
const IDX_LOG_CNO_EC50: usize = 9;
#[cfg(feature = "py")]
const IDX_LOG_CLZ_EC50: usize = 10;
#[cfg(feature = "py")]
const IDX_LOG_DREADD_PROD: usize = 11;
#[cfg(feature = "py")]
const IDX_LOG_DREADD_EC50: usize = 12;

#[cfg(feature = "py")]
const Y_TTA: usize = 0;
#[cfg(feature = "py")]
const Y_DREADD: usize = 1;
#[cfg(feature = "py")]
const Y_BRAIN_RMA: usize = 2;
#[cfg(feature = "py")]
const Y_PLASMA_RMA: usize = 3;

#[cfg(feature = "py")]
type ReducedState = [f64; INFERENCE_N_STATE];
#[cfg(feature = "py")]
type ReducedSensitivity = [[f64; INFERENCE_N_PARAMS]; INFERENCE_N_STATE];

#[cfg(feature = "py")]
fn pow_with_grad(base: f64, exponent: f64) -> (f64, f64) {
    if base <= 0.0 {
        if (exponent - 1.0).abs() < 1e-12 {
            (0.0, 1.0)
        } else {
            (0.0, 0.0)
        }
    } else {
        let value = base.powf(exponent);
        let grad = exponent * base.powf(exponent - 1.0);
        (value, grad)
    }
}

#[cfg(feature = "py")]
fn interpolate_scalar(ts: &[f64], ys: &[f64], t: f64) -> f64 {
    if t <= ts[0] {
        return ys[0];
    }

    let last_idx = ts.len() - 1;
    if t >= ts[last_idx] {
        return ys[last_idx];
    }

    let upper = ts.partition_point(|x| *x < t);
    let lower = upper - 1;
    let dt = ts[upper] - ts[lower];
    if dt.abs() < 1e-12 {
        return ys[lower];
    }
    let w = (t - ts[lower]) / dt;
    ys[lower] + (ys[upper] - ys[lower]) * w
}

#[cfg(feature = "py")]
fn interpolate_sensitivity(
    ts: &[f64],
    ys: &[[f64; INFERENCE_N_PARAMS]],
    t: f64,
) -> [f64; INFERENCE_N_PARAMS] {
    if t <= ts[0] {
        return ys[0];
    }

    let last_idx = ts.len() - 1;
    if t >= ts[last_idx] {
        return ys[last_idx];
    }

    let upper = ts.partition_point(|x| *x < t);
    let lower = upper - 1;
    let dt = ts[upper] - ts[lower];
    if dt.abs() < 1e-12 {
        return ys[lower];
    }

    let w = (t - ts[lower]) / dt;
    let mut out = [0.0; INFERENCE_N_PARAMS];
    for p in 0..INFERENCE_N_PARAMS {
        out[p] = ys[lower][p] + (ys[upper][p] - ys[lower][p]) * w;
    }
    out
}

#[cfg(feature = "py")]
#[derive(Clone)]
struct ForcingSeries {
    dox_t: Vec<f64>,
    dox_v: Vec<f64>,
    cno_t: Vec<f64>,
    cno_v: Vec<f64>,
    clz_t: Vec<f64>,
    clz_v: Vec<f64>,
}

#[cfg(feature = "py")]
impl ForcingSeries {
    fn at(&self, t: f64) -> (f64, f64, f64) {
        (
            interpolate_scalar(&self.dox_t, &self.dox_v, t),
            interpolate_scalar(&self.cno_t, &self.cno_v, t),
            interpolate_scalar(&self.clz_t, &self.clz_v, t),
        )
    }
}

#[cfg(feature = "py")]
#[derive(Clone, Copy)]
struct InferenceParams {
    rma_prod: f64,
    leaky_rma_prod: f64,
    rma_bbb_transport: f64,
    rma_deg: f64,
    tta_prod: f64,
    leaky_tta_prod: f64,
    tta_deg: f64,
    tta_kd: f64,
    dox_tta_kd: f64,
    cno_ec50: f64,
    clz_ec50: f64,
    dreadd_prod: f64,
    dreadd_ec50: f64,
    tta_cooperativity: f64,
    cno_cooperativity: f64,
    clz_cooperativity: f64,
    dreadd_cooperativity: f64,
}

#[cfg(feature = "py")]
fn rhs_and_jacobians(
    y: &ReducedState,
    forcing: (f64, f64, f64),
    p: &InferenceParams,
) -> (
    ReducedState,
    [[f64; INFERENCE_N_STATE]; INFERENCE_N_STATE],
    [[f64; INFERENCE_N_PARAMS]; INFERENCE_N_STATE],
) {
    let tta = y[Y_TTA];
    let dreadd = y[Y_DREADD];
    let brain_rma = y[Y_BRAIN_RMA];
    let plasma_rma = y[Y_PLASMA_RMA];

    let (brain_dox, brain_cno, brain_clz) = forcing;

    let cno_ratio = if p.cno_ec50 > 0.0 {
        brain_cno / p.cno_ec50
    } else {
        0.0
    };
    let clz_ratio = if p.clz_ec50 > 0.0 {
        brain_clz / p.clz_ec50
    } else {
        0.0
    };

    let (cno_ec50_hill, dcno_hill_dcno_ratio) = pow_with_grad(cno_ratio, p.cno_cooperativity);
    let (clz_ec50_hill, dclz_hill_dclz_ratio) = pow_with_grad(clz_ratio, p.clz_cooperativity);

    let q = cno_ec50_hill + clz_ec50_hill;
    let active_dreadd_frac = fractional_activation(q);
    let dactive_frac_dq = fractional_activation_derivative(q);

    let dreadd_mod_base = if p.dreadd_ec50 > 0.0 {
        active_dreadd_frac * dreadd / p.dreadd_ec50
    } else {
        0.0
    };
    let (dreadd_mod, ddreadd_mod_ddreadd_mod_base) =
        pow_with_grad(dreadd_mod_base, p.dreadd_cooperativity);

    let one_plus_dreadd_mod = 1.0 + dreadd_mod;
    let tta_expr = saturating_mix(p.leaky_tta_prod, p.tta_prod, dreadd_mod);

    let dtta_expr_ddreadd_mod = saturating_mix_derivative(p.leaky_tta_prod, p.tta_prod, dreadd_mod);

    let d_dreadd_mod_base_ddreadd = if p.dreadd_ec50 > 0.0 {
        active_dreadd_frac / p.dreadd_ec50
    } else {
        0.0
    };
    let d_dreadd_mod_ddreadd = ddreadd_mod_ddreadd_mod_base * d_dreadd_mod_base_ddreadd;

    let d_cno_hill_dlog_cno_ec50 = -p.cno_cooperativity * cno_ec50_hill;
    let d_clz_hill_dlog_clz_ec50 = -p.clz_cooperativity * clz_ec50_hill;

    let d_q_dlog_cno_ec50 = d_cno_hill_dlog_cno_ec50;
    let d_q_dlog_clz_ec50 = d_clz_hill_dlog_clz_ec50;

    let d_active_dreadd_frac_dlog_cno_ec50 = dactive_frac_dq * d_q_dlog_cno_ec50;
    let d_active_dreadd_frac_dlog_clz_ec50 = dactive_frac_dq * d_q_dlog_clz_ec50;

    let d_dreadd_mod_base_dlog_cno_ec50 = if p.dreadd_ec50 > 0.0 {
        (dreadd / p.dreadd_ec50) * d_active_dreadd_frac_dlog_cno_ec50
    } else {
        0.0
    };
    let d_dreadd_mod_base_dlog_clz_ec50 = if p.dreadd_ec50 > 0.0 {
        (dreadd / p.dreadd_ec50) * d_active_dreadd_frac_dlog_clz_ec50
    } else {
        0.0
    };
    let d_dreadd_mod_base_dlog_dreadd_ec50 = -dreadd_mod_base;

    let d_dreadd_mod_dlog_cno_ec50 = ddreadd_mod_ddreadd_mod_base * d_dreadd_mod_base_dlog_cno_ec50;
    let d_dreadd_mod_dlog_clz_ec50 = ddreadd_mod_ddreadd_mod_base * d_dreadd_mod_base_dlog_clz_ec50;
    let d_dreadd_mod_dlog_dreadd_ec50 =
        ddreadd_mod_ddreadd_mod_base * d_dreadd_mod_base_dlog_dreadd_ec50;

    let active_tta = if p.dox_tta_kd > 0.0 {
        1.0 / (1.0 + (brain_dox / p.dox_tta_kd))
    } else {
        0.0
    };
    let tta_hill_base = if p.tta_kd > 0.0 {
        (active_tta * tta) / p.tta_kd
    } else {
        0.0
    };
    let (tta_hill, dtta_hill_dtta_hill_base) = pow_with_grad(tta_hill_base, p.tta_cooperativity);

    let one_plus_tta_hill = 1.0 + tta_hill;
    let brain_expr = saturating_mix(p.leaky_rma_prod, p.rma_prod, tta_hill);
    let dbrain_expr_dtta_hill = saturating_mix_derivative(p.leaky_rma_prod, p.rma_prod, tta_hill);

    let dtta_hill_base_dtta = if p.tta_kd > 0.0 {
        active_tta / p.tta_kd
    } else {
        0.0
    };
    let dtta_hill_dtta = dtta_hill_dtta_hill_base * dtta_hill_base_dtta;

    let dtta_hill_dlog_tta_kd = -p.tta_cooperativity * tta_hill;
    let d_active_tta_dlog_dox_kd = active_tta * (1.0 - active_tta);
    let dtta_hill_base_dlog_dox_kd = if p.tta_kd > 0.0 {
        (tta / p.tta_kd) * d_active_tta_dlog_dox_kd
    } else {
        0.0
    };
    let dtta_hill_dlog_dox_kd = dtta_hill_dtta_hill_base * dtta_hill_base_dlog_dox_kd;

    let mut dydt = [0.0; INFERENCE_N_STATE];
    dydt[Y_TTA] = tta_expr - (p.tta_deg * tta);
    dydt[Y_DREADD] = p.dreadd_prod - (INFERENCE_DREADD_DEG * dreadd);
    dydt[Y_BRAIN_RMA] = brain_expr - (p.rma_bbb_transport * brain_rma);
    dydt[Y_PLASMA_RMA] = (p.rma_bbb_transport * brain_rma) - (p.rma_deg * plasma_rma);

    let mut jy = [[0.0; INFERENCE_N_STATE]; INFERENCE_N_STATE];
    jy[Y_TTA][Y_TTA] = -p.tta_deg;
    jy[Y_TTA][Y_DREADD] = dtta_expr_ddreadd_mod * d_dreadd_mod_ddreadd;
    jy[Y_DREADD][Y_DREADD] = -INFERENCE_DREADD_DEG;
    jy[Y_BRAIN_RMA][Y_TTA] = dbrain_expr_dtta_hill * dtta_hill_dtta;
    jy[Y_BRAIN_RMA][Y_BRAIN_RMA] = -p.rma_bbb_transport;
    jy[Y_PLASMA_RMA][Y_BRAIN_RMA] = p.rma_bbb_transport;
    jy[Y_PLASMA_RMA][Y_PLASMA_RMA] = -p.rma_deg;

    let mut jp = [[0.0; INFERENCE_N_PARAMS]; INFERENCE_N_STATE];

    jp[Y_TTA][IDX_LOG_TTA_PROD] = (p.tta_prod * dreadd_mod) / one_plus_dreadd_mod;
    jp[Y_TTA][IDX_LOG_TTA_LEAKY_PROD] = p.leaky_tta_prod / one_plus_dreadd_mod;
    jp[Y_TTA][IDX_LOG_TTA_DEG] = -p.tta_deg * tta;
    jp[Y_TTA][IDX_LOG_CNO_EC50] = dtta_expr_ddreadd_mod * d_dreadd_mod_dlog_cno_ec50;
    jp[Y_TTA][IDX_LOG_CLZ_EC50] = dtta_expr_ddreadd_mod * d_dreadd_mod_dlog_clz_ec50;
    jp[Y_TTA][IDX_LOG_DREADD_EC50] = dtta_expr_ddreadd_mod * d_dreadd_mod_dlog_dreadd_ec50;

    jp[Y_DREADD][IDX_LOG_DREADD_PROD] = p.dreadd_prod;

    jp[Y_BRAIN_RMA][IDX_LOG_PROD_LOCAL] = (p.rma_prod * tta_hill) / one_plus_tta_hill;
    jp[Y_BRAIN_RMA][IDX_LOG_LEAKY_PROD_LOCAL] = p.leaky_rma_prod / one_plus_tta_hill;
    jp[Y_BRAIN_RMA][IDX_LOG_BBB] = -p.rma_bbb_transport * brain_rma;
    jp[Y_BRAIN_RMA][IDX_LOG_TTA_KD] = dbrain_expr_dtta_hill * dtta_hill_dlog_tta_kd;
    jp[Y_BRAIN_RMA][IDX_LOG_DOX_KD] = dbrain_expr_dtta_hill * dtta_hill_dlog_dox_kd;

    jp[Y_PLASMA_RMA][IDX_LOG_BBB] = p.rma_bbb_transport * brain_rma;
    jp[Y_PLASMA_RMA][IDX_LOG_DEG] = -p.rma_deg * plasma_rma;

    let _ = dcno_hill_dcno_ratio;
    let _ = dclz_hill_dclz_ratio;

    (dydt, jy, jp)
}

#[cfg(feature = "py")]
fn augmented_derivative(
    t: f64,
    y: &ReducedState,
    s: &ReducedSensitivity,
    p: &InferenceParams,
    forcing: &ForcingSeries,
) -> (ReducedState, ReducedSensitivity) {
    let forcing_values = forcing.at(t);
    let (dy, jy, jp) = rhs_and_jacobians(y, forcing_values, p);

    let mut ds = [[0.0; INFERENCE_N_PARAMS]; INFERENCE_N_STATE];
    for i in 0..INFERENCE_N_STATE {
        for j in 0..INFERENCE_N_PARAMS {
            let mut value = jp[i][j];
            for k in 0..INFERENCE_N_STATE {
                value += jy[i][k] * s[k][j];
            }
            ds[i][j] = value;
        }
    }

    (dy, ds)
}

#[cfg(feature = "py")]
fn rk4_step_augmented(
    t: f64,
    h: f64,
    y: &ReducedState,
    s: &ReducedSensitivity,
    p: &InferenceParams,
    forcing: &ForcingSeries,
) -> (ReducedState, ReducedSensitivity) {
    let (k1y, k1s) = augmented_derivative(t, y, s, p, forcing);

    let mut y2 = [0.0; INFERENCE_N_STATE];
    let mut s2 = [[0.0; INFERENCE_N_PARAMS]; INFERENCE_N_STATE];
    for i in 0..INFERENCE_N_STATE {
        y2[i] = y[i] + 0.5 * h * k1y[i];
        for j in 0..INFERENCE_N_PARAMS {
            s2[i][j] = s[i][j] + 0.5 * h * k1s[i][j];
        }
    }

    let (k2y, k2s) = augmented_derivative(t + 0.5 * h, &y2, &s2, p, forcing);

    let mut y3 = [0.0; INFERENCE_N_STATE];
    let mut s3 = [[0.0; INFERENCE_N_PARAMS]; INFERENCE_N_STATE];
    for i in 0..INFERENCE_N_STATE {
        y3[i] = y[i] + 0.5 * h * k2y[i];
        for j in 0..INFERENCE_N_PARAMS {
            s3[i][j] = s[i][j] + 0.5 * h * k2s[i][j];
        }
    }

    let (k3y, k3s) = augmented_derivative(t + 0.5 * h, &y3, &s3, p, forcing);

    let mut y4 = [0.0; INFERENCE_N_STATE];
    let mut s4 = [[0.0; INFERENCE_N_PARAMS]; INFERENCE_N_STATE];
    for i in 0..INFERENCE_N_STATE {
        y4[i] = y[i] + h * k3y[i];
        for j in 0..INFERENCE_N_PARAMS {
            s4[i][j] = s[i][j] + h * k3s[i][j];
        }
    }

    let (k4y, k4s) = augmented_derivative(t + h, &y4, &s4, p, forcing);

    let mut y_next = [0.0; INFERENCE_N_STATE];
    let mut s_next = [[0.0; INFERENCE_N_PARAMS]; INFERENCE_N_STATE];
    for i in 0..INFERENCE_N_STATE {
        y_next[i] = y[i] + (h / 6.0) * (k1y[i] + (2.0 * k2y[i]) + (2.0 * k3y[i]) + k4y[i]);
        for j in 0..INFERENCE_N_PARAMS {
            s_next[i][j] = s[i][j]
                + (h / 6.0) * (k1s[i][j] + (2.0 * k2s[i][j]) + (2.0 * k3s[i][j]) + k4s[i][j]);
        }
    }

    (y_next, s_next)
}

#[cfg(feature = "py")]
fn to_pyarray2<'py>(
    py: Python<'py>,
    flat: &[f64],
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mut vec2 = Vec::with_capacity(rows);
    for r in 0..rows {
        let start = r * cols;
        let stop = start + cols;
        vec2.push(flat[start..stop].to_vec());
    }
    PyArray2::from_vec2(py, &vec2)
        .map_err(|e| PyValueError::new_err(format!("Failed to create numpy matrix: {}", e)))
}

#[cfg(feature = "py")]
fn build_forcing_series(
    t0: f64,
    tf: f64,
    dt_sub: f64,
    dox_pk_model: &DoxModel,
    cno_pk_model: &CNOModel,
    plasma_dox_ss: f64,
    brain_dox_ss: f64,
) -> PyResult<ForcingSeries> {
    let mut dox_solver = DiagonallyImplicitRungeKutta::kvaerno423()
        .h0(dt_sub)
        .h_max(dt_sub);
    let dox_solution = dox_pk_model
        .solve(
            t0,
            tf,
            dt_sub,
            DoxState::new(plasma_dox_ss, brain_dox_ss),
            &mut dox_solver,
        )
        .map_err(|e| PyValueError::new_err(format!("Failed to solve dox PK model: {:?}", e)))?;

    let dox_t = dox_solution.t.clone();
    let dox_v = dox_solution.brain_dox().map_err(|e| {
        PyValueError::new_err(format!("Failed to access dox concentrations: {}", e))
    })?;

    if dox_t.iter().any(|value| !value.is_finite()) || dox_v.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "Non-finite values encountered in dox forcing series",
        ));
    }

    let mut cno_solver = DiagonallyImplicitRungeKutta::kvaerno423()
        .h0(dt_sub)
        .h_max(dt_sub);
    let cno_solution = cno_pk_model
        .solve(t0, tf, dt_sub, CNOState::zeros(), &mut cno_solver)
        .map_err(|e| PyValueError::new_err(format!("Failed to solve CNO PK model: {:?}", e)))?;

    let cno_t = cno_solution.t.clone();
    let cno_v = cno_solution.brain_cno().map_err(|e| {
        PyValueError::new_err(format!("Failed to access CNO concentrations: {}", e))
    })?;
    let clz_t = cno_solution.t.clone();
    let clz_v = cno_solution.brain_clz().map_err(|e| {
        PyValueError::new_err(format!("Failed to access CLZ concentrations: {}", e))
    })?;

    if cno_t.iter().any(|value| !value.is_finite())
        || cno_v.iter().any(|value| !value.is_finite())
        || clz_t.iter().any(|value| !value.is_finite())
        || clz_v.iter().any(|value| !value.is_finite())
    {
        return Err(PyValueError::new_err(
            "Non-finite values encountered in CNO/CLZ forcing series",
        ));
    }

    Ok(ForcingSeries {
        dox_t,
        dox_v,
        cno_t,
        cno_v,
        clz_t,
        clz_v,
    })
}

#[cfg(feature = "py")]
#[pyclass(name = "SensitivityEngine")]
#[derive(Clone)]
pub struct SensitivityEngine {
    n_mice: usize,
    n_obs: usize,
    obs_time: Vec<f64>,
    obs_by_mouse: Vec<Vec<usize>>,
    t0: f64,
    tf: f64,
    dt_sub: f64,
    forcing: ForcingSeries,
    tta_cooperativity: f64,
    cno_cooperativity: f64,
    clz_cooperativity: f64,
    dreadd_cooperativity: f64,
}

#[cfg(feature = "py")]
#[pymethods]
impl SensitivityEngine {
    #[new]
    #[pyo3(signature = (mouse_id, obs_time, n_mice, dox_pk_model=DoxModel::default(), cno_pk_model=CNOModel::default(), plasma_dox_ss=0.0, brain_dox_ss=0.0, t0=0.0, dt_sub=0.25, tta_cooperativity=DEFAULT_TTA_COOPERATIVITY, cno_cooperativity=DEFAULT_CNO_COOPERATIVITY, clz_cooperativity=DEFAULT_CLZ_COOPERATIVITY, dreadd_cooperativity=DEFAULT_DREADD_COOPERATIVITY))]
    fn new(
        mouse_id: PyReadonlyArray1<'_, i64>,
        obs_time: PyReadonlyArray1<'_, f64>,
        n_mice: usize,
        dox_pk_model: DoxModel,
        cno_pk_model: CNOModel,
        plasma_dox_ss: f64,
        brain_dox_ss: f64,
        t0: f64,
        dt_sub: f64,
        tta_cooperativity: f64,
        cno_cooperativity: f64,
        clz_cooperativity: f64,
        dreadd_cooperativity: f64,
    ) -> PyResult<Self> {
        if n_mice == 0 {
            return Err(PyValueError::new_err("n_mice must be > 0"));
        }
        if dt_sub <= 0.0 {
            return Err(PyValueError::new_err("dt_sub must be > 0"));
        }

        let mouse_id_vec = mouse_id
            .as_array()
            .iter()
            .map(|value| {
                if *value < 0 {
                    Err(PyValueError::new_err("mouse_id values must be >= 0"))
                } else {
                    Ok(*value as usize)
                }
            })
            .collect::<PyResult<Vec<usize>>>()?;

        let obs_time_vec = obs_time.as_array().iter().copied().collect::<Vec<f64>>();

        if mouse_id_vec.len() != obs_time_vec.len() {
            return Err(PyValueError::new_err(
                "mouse_id and obs_time must have matching lengths",
            ));
        }
        if obs_time_vec.is_empty() {
            return Err(PyValueError::new_err("obs_time must not be empty"));
        }

        let mut obs_by_mouse = vec![Vec::<usize>::new(); n_mice];
        let mut tf = t0;
        for (idx, m) in mouse_id_vec.iter().enumerate() {
            if *m >= n_mice {
                return Err(PyValueError::new_err(format!(
                    "mouse_id {} is out of bounds for n_mice={}",
                    m, n_mice
                )));
            }
            let t = obs_time_vec[idx];
            if t < t0 {
                return Err(PyValueError::new_err(format!(
                    "obs_time at index {} is below t0: {} < {}",
                    idx, t, t0
                )));
            }
            tf = tf.max(t);
            obs_by_mouse[*m].push(idx);
        }

        if tf <= t0 {
            tf = t0 + dt_sub;
        }

        let forcing = build_forcing_series(
            t0,
            tf,
            dt_sub,
            &dox_pk_model,
            &cno_pk_model,
            plasma_dox_ss,
            brain_dox_ss,
        )?;

        Ok(Self {
            n_mice,
            n_obs: obs_time_vec.len(),
            obs_time: obs_time_vec,
            obs_by_mouse,
            t0,
            tf,
            dt_sub,
            forcing,
            tta_cooperativity,
            cno_cooperativity,
            clz_cooperativity,
            dreadd_cooperativity,
        })
    }

    #[pyo3(signature = (log_prod_mouse, log_leaky_prod_mouse, log_bbb, log_deg, log_tta_prod, log_tta_leaky_prod, log_tta_deg, log_tta_kd, log_dox_kd, log_cno_ec50, log_clz_ec50, log_dreadd_prod, log_dreadd_ec50))]
    fn predict_with_jacobian<'py>(
        &self,
        py: Python<'py>,
        log_prod_mouse: PyReadonlyArray1<'_, f64>,
        log_leaky_prod_mouse: PyReadonlyArray1<'_, f64>,
        log_bbb: f64,
        log_deg: f64,
        log_tta_prod: f64,
        log_tta_leaky_prod: f64,
        log_tta_deg: f64,
        log_tta_kd: f64,
        log_dox_kd: f64,
        log_cno_ec50: f64,
        log_clz_ec50: f64,
        log_dreadd_prod: f64,
        log_dreadd_ec50: f64,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
        Bound<'py, PyArray2<f64>>,
    )> {
        let log_prod = log_prod_mouse
            .as_array()
            .iter()
            .copied()
            .collect::<Vec<f64>>();
        let log_leaky = log_leaky_prod_mouse
            .as_array()
            .iter()
            .copied()
            .collect::<Vec<f64>>();

        if log_prod.len() != self.n_mice {
            return Err(PyValueError::new_err(format!(
                "log_prod_mouse length {} does not match n_mice {}",
                log_prod.len(),
                self.n_mice
            )));
        }
        if log_leaky.len() != self.n_mice {
            return Err(PyValueError::new_err(format!(
                "log_leaky_prod_mouse length {} does not match n_mice {}",
                log_leaky.len(),
                self.n_mice
            )));
        }

        let mut mu = vec![0.0; self.n_obs];
        let mut dmu_dlog_prod_mouse = vec![0.0; self.n_obs * self.n_mice];
        let mut dmu_dlog_leaky_prod_mouse = vec![0.0; self.n_obs * self.n_mice];
        let mut dmu_dglobal = vec![0.0; self.n_obs * INFERENCE_N_GLOBAL_PARAMS];

        for m in 0..self.n_mice {
            if self.obs_by_mouse[m].is_empty() {
                continue;
            }

            let params = InferenceParams {
                rma_prod: log_prod[m].exp(),
                leaky_rma_prod: log_leaky[m].exp(),
                rma_bbb_transport: log_bbb.exp(),
                rma_deg: log_deg.exp(),
                tta_prod: log_tta_prod.exp(),
                leaky_tta_prod: log_tta_leaky_prod.exp(),
                tta_deg: log_tta_deg.exp(),
                tta_kd: log_tta_kd.exp(),
                dox_tta_kd: log_dox_kd.exp(),
                cno_ec50: log_cno_ec50.exp(),
                clz_ec50: log_clz_ec50.exp(),
                dreadd_prod: log_dreadd_prod.exp(),
                dreadd_ec50: log_dreadd_ec50.exp(),
                tta_cooperativity: self.tta_cooperativity,
                cno_cooperativity: self.cno_cooperativity,
                clz_cooperativity: self.clz_cooperativity,
                dreadd_cooperativity: self.dreadd_cooperativity,
            };

            let mut t = self.t0;
            let mut y = [0.0; INFERENCE_N_STATE];
            y[Y_TTA] = 0.0;
            y[Y_DREADD] = params.dreadd_prod;
            y[Y_BRAIN_RMA] = 0.0;
            y[Y_PLASMA_RMA] = 0.0;

            let mut s = [[0.0; INFERENCE_N_PARAMS]; INFERENCE_N_STATE];
            s[Y_DREADD][IDX_LOG_DREADD_PROD] = params.dreadd_prod;

            let mut ts = vec![t];
            let mut plasma = vec![y[Y_PLASMA_RMA]];
            let mut plasma_sens = vec![s[Y_PLASMA_RMA]];

            while t < self.tf - 1e-12 {
                let h = self.dt_sub.min(self.tf - t);
                let (y_next, s_next) = rk4_step_augmented(t, h, &y, &s, &params, &self.forcing);
                t += h;
                y = y_next;
                s = s_next;

                if !(y[Y_PLASMA_RMA].is_finite()) {
                    return Err(PyValueError::new_err(
                        "non-finite plasma_rma produced during sensitivity integration",
                    ));
                }

                ts.push(t);
                plasma.push(y[Y_PLASMA_RMA]);
                plasma_sens.push(s[Y_PLASMA_RMA]);
            }

            for &obs_idx in &self.obs_by_mouse[m] {
                let t_obs = self.obs_time[obs_idx];
                let mu_val = interpolate_scalar(&ts, &plasma, t_obs);
                let sens_val = interpolate_sensitivity(&ts, &plasma_sens, t_obs);

                mu[obs_idx] = mu_val;
                dmu_dlog_prod_mouse[(obs_idx * self.n_mice) + m] = sens_val[IDX_LOG_PROD_LOCAL];
                dmu_dlog_leaky_prod_mouse[(obs_idx * self.n_mice) + m] =
                    sens_val[IDX_LOG_LEAKY_PROD_LOCAL];

                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 0] = sens_val[IDX_LOG_BBB];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 1] = sens_val[IDX_LOG_DEG];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 2] = sens_val[IDX_LOG_TTA_PROD];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 3] =
                    sens_val[IDX_LOG_TTA_LEAKY_PROD];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 4] = sens_val[IDX_LOG_TTA_DEG];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 5] = sens_val[IDX_LOG_TTA_KD];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 6] = sens_val[IDX_LOG_DOX_KD];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 7] = sens_val[IDX_LOG_CNO_EC50];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 8] = sens_val[IDX_LOG_CLZ_EC50];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 9] =
                    sens_val[IDX_LOG_DREADD_PROD];
                dmu_dglobal[(obs_idx * INFERENCE_N_GLOBAL_PARAMS) + 10] =
                    sens_val[IDX_LOG_DREADD_EC50];
            }
        }

        let mu_py = PyArray1::from_vec(py, mu);
        let jac_prod_py = to_pyarray2(py, &dmu_dlog_prod_mouse, self.n_obs, self.n_mice)?;
        let jac_leaky_py = to_pyarray2(py, &dmu_dlog_leaky_prod_mouse, self.n_obs, self.n_mice)?;
        let jac_global_py = to_pyarray2(py, &dmu_dglobal, self.n_obs, INFERENCE_N_GLOBAL_PARAMS)?;

        Ok((mu_py, jac_prod_py, jac_leaky_py, jac_global_py))
    }

    #[getter]
    fn get_n_obs(&self) -> usize {
        self.n_obs
    }

    #[getter]
    fn get_n_mice(&self) -> usize {
        self.n_mice
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (rma_prod=DEFAULT_RMA_PROD, leaky_rma_prod=DEFAULT_LEAKY_RMA_PROD, rma_bbb_transport=DEFAULT_RMA_BBB_TRANSPORT, rma_deg=DEFAULT_RMA_DEG, tta_prod=DEFAULT_TTA_PROD, leaky_tta_prod=DEFAULT_LEAKY_TTA_PROD, tta_deg=DEFAULT_TTA_DEG, tta_kd=DEFAULT_TTA_KD, tta_cooperativity=DEFAULT_TTA_COOPERATIVITY, dox_pk_model=DoxModel::default(), dox_tta_kd=DEFAULT_DOX_TTA_KD, cno_pk_model=CNOModel::default(), cno_ec50=DEFAULT_CNO_EC50, clz_ec50=DEFAULT_CLZ_EC50, cno_cooperativity=DEFAULT_CNO_COOPERATIVITY, clz_cooperativity=DEFAULT_CLZ_COOPERATIVITY, dreadd_prod=DEFAULT_DREADD_PROD, dreadd_deg=DEFAULT_DREADD_DEG, dreadd_ec50=DEFAULT_DREADD_EC50, dreadd_cooperativity=DEFAULT_DREADD_COOPERATIVITY))]
    pub fn create(
        rma_prod: f64,
        leaky_rma_prod: f64,
        rma_bbb_transport: f64,
        rma_deg: f64,
        tta_prod: f64,
        leaky_tta_prod: f64,
        tta_deg: f64,
        tta_kd: f64,
        tta_cooperativity: f64,
        dox_pk_model: DoxModel,
        dox_tta_kd: f64,
        cno_pk_model: CNOModel,
        cno_ec50: f64,
        clz_ec50: f64,
        cno_cooperativity: f64,
        clz_cooperativity: f64,
        dreadd_prod: f64,
        dreadd_deg: f64,
        dreadd_ec50: f64,
        dreadd_cooperativity: f64,
    ) -> Self {
        Self {
            rma_prod,
            leaky_rma_prod,
            rma_bbb_transport,
            rma_deg,
            tta_prod,
            leaky_tta_prod,
            tta_deg,
            tta_kd,
            tta_cooperativity,
            dox_pk_model,
            dox_tta_kd,
            cno_pk_model,
            cno_ec50,
            clz_ec50,
            cno_cooperativity,
            clz_cooperativity,
            dreadd_prod,
            dreadd_deg,
            dreadd_ec50,
            dreadd_cooperativity,
        }
    }

    #[pyo3(name = "solve")]
    fn py_solve(
        &self,
        t0: f64,
        tf: f64,
        dt: f64,
        init_state: PyState,
        solver: PySolver,
    ) -> PyResult<PySolution> {
        let result = match solver.solver_type.as_str() {
            "dopri5" => {
                let mut solver_instance =
                    differential_equations::methods::ExplicitRungeKutta::dopri5()
                        .rtol(solver.rtol)
                        .atol(solver.atol)
                        .h0(solver.dt0)
                        .h_min(solver.min_dt)
                        .h_max(solver.max_dt)
                        .max_steps(solver.max_steps)
                        .max_rejects(solver.max_rejected_steps)
                        .safety_factor(solver.safety_factor)
                        .min_scale(solver.min_scale)
                        .max_scale(solver.max_scale);
                self.solve(t0, tf, dt, init_state.inner, &mut solver_instance)
            }
            "kvaerno3" => {
                let mut solver_instance =
                    differential_equations::methods::DiagonallyImplicitRungeKutta::kvaerno423()
                        .rtol(solver.rtol)
                        .atol(solver.atol)
                        .h0(solver.dt0)
                        .h_min(solver.min_dt)
                        .h_max(solver.max_dt)
                        .max_steps(solver.max_steps)
                        .max_rejects(solver.max_rejected_steps)
                        .safety_factor(solver.safety_factor)
                        .min_scale(solver.min_scale)
                        .max_scale(solver.max_scale);
                self.solve(t0, tf, dt, init_state.inner, &mut solver_instance)
            }
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Solver '{}' not supported",
                    solver.solver_type
                )));
            }
        };

        match result {
            Ok(solution) => Ok(PySolution {
                inner: InnerSolution::Chemogenetic(solution),
            }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to solve: {:?}", e))),
        }
    }

    #[getter]
    fn get_rma_prod(&self) -> f64 {
        self.rma_prod
    }
    #[getter]
    fn get_leaky_rma_prod(&self) -> f64 {
        self.leaky_rma_prod
    }
    #[getter]
    fn get_rma_bbb_transport(&self) -> f64 {
        self.rma_bbb_transport
    }
    #[getter]
    fn get_rma_deg(&self) -> f64 {
        self.rma_deg
    }
    #[getter]
    fn get_tta_prod(&self) -> f64 {
        self.tta_prod
    }
    #[getter]
    fn get_leaky_tta_prod(&self) -> f64 {
        self.leaky_tta_prod
    }
    #[getter]
    fn get_tta_deg(&self) -> f64 {
        self.tta_deg
    }
    #[getter]
    fn get_tta_kd(&self) -> f64 {
        self.tta_kd
    }
    #[getter]
    fn get_tta_cooperativity(&self) -> f64 {
        self.tta_cooperativity
    }
    #[getter]
    fn get_dox_pk_model(&self) -> DoxModel {
        self.dox_pk_model.clone()
    }
    #[getter]
    fn get_dox_tta_kd(&self) -> f64 {
        self.dox_tta_kd
    }
    #[getter]
    fn get_cno_pk_model(&self) -> CNOModel {
        self.cno_pk_model.clone()
    }
    #[getter]
    fn get_cno_ec50(&self) -> f64 {
        self.cno_ec50
    }
    #[getter]
    fn get_clz_ec50(&self) -> f64 {
        self.clz_ec50
    }
    #[getter]
    fn get_cno_cooperativity(&self) -> f64 {
        self.cno_cooperativity
    }
    #[getter]
    fn get_clz_cooperativity(&self) -> f64 {
        self.clz_cooperativity
    }
    #[getter]
    fn get_dreadd_prod(&self) -> f64 {
        self.dreadd_prod
    }
    #[getter]
    fn get_dreadd_deg(&self) -> f64 {
        self.dreadd_deg
    }
    #[getter]
    fn get_dreadd_ec50(&self) -> f64 {
        self.dreadd_ec50
    }
    #[getter]
    fn get_dreadd_cooperativity(&self) -> f64 {
        self.dreadd_cooperativity
    }
}

#[cfg(test)]
mod tests {
    use differential_equations::{prelude::DiagonallyImplicitRungeKutta, status::Status};

    use super::*;

    #[test]
    fn state_creation() {
        let zero_state = State::zeros();
        assert_eq!(zero_state.brain_rma, 0.);
        assert_eq!(zero_state.plasma_rma, 0.);
        assert_eq!(zero_state.tta, 0.);
        assert_eq!(zero_state.plasma_dox, 0.);
        assert_eq!(zero_state.brain_dox, 0.);
        assert_eq!(zero_state.dreadd, 0.);
        assert_eq!(zero_state.peritoneal_cno, 0.);
        assert_eq!(zero_state.plasma_cno, 0.);
        assert_eq!(zero_state.brain_cno, 0.);
        assert_eq!(zero_state.plasma_clz, 0.);
        assert_eq!(zero_state.brain_clz, 0.);

        let custom_state = State::new(0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.);
        assert_eq!(custom_state.brain_rma, 0.);
        assert_eq!(custom_state.plasma_rma, 10.);
        assert_eq!(custom_state.tta, 20.);
        assert_eq!(custom_state.plasma_dox, 30.);
        assert_eq!(custom_state.brain_dox, 40.);
        assert_eq!(custom_state.dreadd, 50.);
        assert_eq!(custom_state.peritoneal_cno, 60.);
        assert_eq!(custom_state.plasma_cno, 70.);
        assert_eq!(custom_state.brain_cno, 80.);
    }

    #[test]
    fn model_creation() -> Result<(), ModelBuilderError> {
        let default_model = Model::default();
        assert_eq!(default_model.rma_prod, DEFAULT_RMA_PROD);

        let custom_model = Model::builder().rma_prod(0.5).tta_prod(10.).build()?;
        assert_eq!(custom_model.rma_prod, 0.5);
        assert_eq!(custom_model.tta_prod, 10.);

        Ok(())
    }

    #[test]
    fn model_simulation() -> Result<(), ModelBuilderError> {
        let model = Model::default();
        let state = State::zeros();
        let mut solver = DiagonallyImplicitRungeKutta::kvaerno423();
        let solution = model.solve(0., 48., 1., state, &mut solver);

        assert!(solution.is_ok());
        let solution = solution.unwrap();
        assert!(matches!(solution.status, Status::Complete));
        assert!(solution.y.last().unwrap().plasma_rma > 0.);
        assert!(solution.plasma_cno().is_ok());
        assert!(solution.plasma_rma().is_ok());
        assert!(solution.plasma_dox().is_ok());
        assert!(solution.max_plasma_cno().is_ok());
        assert!(solution.max_plasma_rma().is_ok());
        assert!(solution.max_plasma_dox().is_ok());

        // test simulation with high leaky rma prod
        let model = Model::builder().leaky_rma_prod(0.2).build()?;
        let solution = model.solve(0., 48., 1., state, &mut solver);
        assert!(solution.is_ok());

        Ok(())
    }

    #[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
    #[test]
    fn dataframe_conversion() -> Result<(), PolarsError> {
        let model = Model::default();
        let state = State::zeros();
        let mut solver = DiagonallyImplicitRungeKutta::kvaerno423();
        let solution = model.solve(0., 48., 1., state, &mut solver);

        assert!(solution.is_ok());
        let solution = solution.unwrap();

        let dataframe = solution.to_dataframe()?;
        assert_eq!(dataframe.shape(), (49, 12));
        assert_eq!(
            dataframe.get_column_names(),
            &[
                "time",
                "brain_rma",
                "plasma_rma",
                "tta",
                "plasma_dox",
                "brain_dox",
                "dreadd",
                "peritoneal_cno",
                "plasma_cno",
                "brain_cno",
                "plasma_clz",
                "brain_clz"
            ]
        );

        Ok(())
    }
}
