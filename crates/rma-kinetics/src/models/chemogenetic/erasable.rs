use derive_builder::Builder;
use differential_equations::{
    derive::State as StateTrait,
    error::Error,
    ivp::IVP,
    ode::{OrdinaryNumericalMethod, ODE},
    prelude::{Interpolation, Solution},
};

#[cfg(feature = "py")]
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyResult};

#[cfg(feature = "py")]
use crate::solve::{InnerSolution, PySolution, PySolver};

#[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
use polars::{error::PolarsError, frame::DataFrame};

#[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
use crate::ToDataFrame;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::{diff_chemogenetic_core, saturating_mix, ChemogeneticCoreFields};
use crate::{
    models::{
        cno::{CNOFields, CNOPKAccess, CnoDose, Model as CNOModel},
        dox::{DoxFields, Model as DoxModel},
        erasable::{
            TevDose, TevFields, DEFAULT_TEV_CUT_RATE, DEFAULT_TEV_DEG, DEFAULT_TEV_DOSE_NMOL,
            DEFAULT_TEV_DOSE_TIME, DEFAULT_TEV_PLASMA_VD,
        },
    },
    pk::{validate_unique_dose_times, DoseApplyingSolout, ScheduledStateUpdate},
    solve::SpeciesAccessError,
    SolutionAccess, Solve,
};

/// Chemogenetic model + fast erasable RMA state.
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
    pub plasma_tev: T,
}

impl State<f64> {
    /// Create a new chemogenetic feRMA model state where all concentrations are set to zero.
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
            plasma_tev: 0.,
        }
    }

    /// Create a new chemogenetic feRMA model state.
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
        plasma_tev: f64,
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
            plasma_tev,
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for State<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "brain_rma={:.3}, plasma_rma={:.3}, tta={:.3}, plasma_dox={:.3}, brain_dox={:.3}, dreadd={:.3}, peritoneal_cno={:.3}, plasma_cno={:.3}, brain_cno={:.3}, plasma_clz={:.3}, brain_clz={:.3}, plasma_tev={:.3}",
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
            self.brain_clz,
            self.plasma_tev,
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
    #[pyo3(signature = (brain_rma=0., plasma_rma=0., tta=0., plasma_dox=0., brain_dox=0., dreadd=0., peritoneal_cno=0., plasma_cno=0., brain_cno=0., plasma_clz=0., brain_clz=0., plasma_tev=0.))]
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
        plasma_tev: f64,
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
                plasma_tev,
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

    #[getter]
    fn get_plasma_tev(&self) -> f64 {
        self.inner.plasma_tev
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

    #[setter]
    fn set_plasma_tev(&mut self, value: f64) -> PyResult<()> {
        self.inner.plasma_tev = value;
        Ok(())
    }
}

#[cfg(feature = "py")]
impl std::fmt::Display for PyState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
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
        Ok(self
            .y
            .iter()
            .map(|state| state.plasma_tev)
            .collect::<Vec<f64>>())
    }

    fn max_plasma_tev(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Ok(crate::max_species!(self, plasma_tev))
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

// TODO: can extract this into a trait
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

// TODO: can extract this into a trait
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

impl TevFields for State<f64> {
    fn plasma_tev(&self) -> f64 {
        self.plasma_tev
    }

    fn plasma_tev_mut(&mut self) -> &mut f64 {
        &mut self.plasma_tev
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
#[builder(derive(Debug), build_fn(validate = "Self::validate"))]
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
    #[builder(default = "vec![TevDose::new(DEFAULT_TEV_DOSE_NMOL, DEFAULT_TEV_DOSE_TIME)]")]
    pub tev_doses: Vec<TevDose>,
    #[builder(default = "DEFAULT_TEV_PLASMA_VD")]
    pub tev_plasma_vd: f64,
    #[builder(default = "DEFAULT_TEV_DEG")]
    pub tev_deg: f64,
    #[builder(default = "DEFAULT_TEV_CUT_RATE")]
    pub tev_cut_rate: f64,
}

impl Default for Model {
    fn default() -> Self {
        ModelBuilder::default().build().unwrap()
    }
}

impl Model {
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
    }
}

impl ModelBuilder {
    /// Validate tev doses are administered at unique times
    /// CNO doses are validated within the CNO model itself.
    fn validate(&self) -> Result<(), String> {
        if let Some(doses) = self.tev_doses.as_deref() {
            validate_unique_dose_times(doses).map_err(|e| e.to_string())
        } else {
            Ok(())
        }
    }
}

impl CNOPKAccess for Model {
    fn get_doses(&self) -> &Vec<CnoDose> {
        &self.cno_pk_model.doses
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (rma_prod=DEFAULT_RMA_PROD, leaky_rma_prod=DEFAULT_LEAKY_RMA_PROD, rma_bbb_transport=DEFAULT_RMA_BBB_TRANSPORT, rma_deg=DEFAULT_RMA_DEG, tta_prod=DEFAULT_TTA_PROD, leaky_tta_prod=DEFAULT_LEAKY_TTA_PROD, tta_deg=DEFAULT_TTA_DEG, tta_kd=DEFAULT_TTA_KD, tta_cooperativity=DEFAULT_TTA_COOPERATIVITY, dox_pk_model=DoxModel::default(), dox_tta_kd=DEFAULT_DOX_TTA_KD, cno_pk_model=CNOModel::default(), cno_ec50=DEFAULT_CNO_EC50, clz_ec50=DEFAULT_CLZ_EC50, cno_cooperativity=DEFAULT_CNO_COOPERATIVITY, clz_cooperativity=DEFAULT_CLZ_COOPERATIVITY, dreadd_prod=DEFAULT_DREADD_PROD, dreadd_deg=DEFAULT_DREADD_DEG, dreadd_ec50=DEFAULT_DREADD_EC50, dreadd_cooperativity=DEFAULT_DREADD_COOPERATIVITY, tev_doses=vec![TevDose::new(DEFAULT_TEV_DOSE_NMOL, DEFAULT_TEV_DOSE_TIME)], tev_plasma_vd=DEFAULT_TEV_PLASMA_VD, tev_deg=DEFAULT_TEV_DEG, tev_cut_rate=DEFAULT_TEV_CUT_RATE))]
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
        tev_doses: Vec<TevDose>,
        tev_plasma_vd: f64,
        tev_deg: f64,
        tev_cut_rate: f64,
    ) -> PyResult<Self> {
        validate_unique_dose_times(&tev_doses).map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
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
            tev_doses,
            tev_plasma_vd,
            tev_deg,
            tev_cut_rate,
        })
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
                let solver_instance = differential_equations::methods::ExplicitRungeKutta::dopri5()
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
                self.solve(t0, tf, dt, init_state.inner, solver_instance)
            }
            "kvaerno3" => {
                let solver_instance =
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
                self.solve(t0, tf, dt, init_state.inner, solver_instance)
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
                inner: InnerSolution::ChemogeneticErasable(solution),
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

    #[getter]
    fn get_tev_doses(&self) -> Vec<TevDose> {
        self.tev_doses.clone()
    }

    #[getter]
    fn get_tev_plasma_vd(&self) -> f64 {
        self.tev_plasma_vd
    }

    #[getter]
    fn get_tev_deg(&self) -> f64 {
        self.tev_deg
    }

    #[getter]
    fn get_tev_cut_rate(&self) -> f64 {
        self.tev_cut_rate
    }
}

impl ODE<f64, State<f64>> for Model {
    fn diff(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
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

        // tet inducible RMA expression
        let active_tta = 1. / (1. + y.brain_dox / self.dox_tta_kd);
        let tta_hill = (active_tta * y.tta / self.tta_kd).powf(self.tta_cooperativity);
        let brain_efflux = self.rma_bbb_transport * y.brain_rma;
        let tev_conc = y.plasma_tev / self.tev_plasma_vd;
        let cleaved_rma = self.tev_cut_rate * y.plasma_rma * tev_conc;

        dydt.brain_rma =
            saturating_mix(self.leaky_rma_prod, self.rma_prod, tta_hill) - brain_efflux;
        dydt.plasma_rma = brain_efflux - (self.rma_deg * y.plasma_rma) - cleaved_rma;
        dydt.plasma_tev = -(self.tev_deg * y.plasma_tev);
    }
}

#[derive(Clone)]
enum ScheduledUpdate {
    Cno(CnoDose),
    Tev(TevDose),
}

impl ScheduledStateUpdate<State<f64>> for ScheduledUpdate {
    fn time(&self) -> f64 {
        match self {
            Self::Cno(dose) => dose.time,
            Self::Tev(dose) => dose.time,
        }
    }

    fn apply(&self, state: &mut State<f64>) {
        match self {
            Self::Cno(dose) => state.peritoneal_cno += dose.nmol,
            Self::Tev(dose) => state.plasma_tev += dose.nmol,
        }
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
        solver: S,
    ) -> Result<Solution<f64, Self::State>, Error<f64, Self::State>>
    where
        S: OrdinaryNumericalMethod<f64, Self::State> + Interpolation<f64, Self::State>,
    {
        let mut adjusted_init_state = init_state;

        for dose in &self.cno_pk_model.doses {
            if (dose.time - t0).abs() < 1e-10 {
                adjusted_init_state.peritoneal_cno += dose.nmol;
            }
        }

        for dose in &self.tev_doses {
            if (dose.time - t0).abs() < 1e-10 {
                adjusted_init_state.plasma_tev += dose.nmol;
            }
        }

        let mut scheduled_updates = self
            .cno_pk_model
            .doses
            .iter()
            .filter(|dose| (dose.time - t0).abs() >= 1e-10)
            .cloned()
            .map(ScheduledUpdate::Cno)
            .chain(
                self.tev_doses
                    .iter()
                    .filter(|dose| (dose.time - t0).abs() >= 1e-10)
                    .cloned()
                    .map(ScheduledUpdate::Tev),
            )
            .collect::<Vec<ScheduledUpdate>>();

        scheduled_updates.sort_by(|a, b| a.time().total_cmp(&b.time()));

        let dosing_solout =
            DoseApplyingSolout::<State<f64>, ScheduledUpdate>::new(scheduled_updates, t0, tf, dt);

        let problem = IVP::ode(self, t0, tf, adjusted_init_state);
        let mut solution = problem.solout(dosing_solout).method(solver).solve()?;

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
                plasma_tev: state.plasma_tev / self.tev_plasma_vd,
            })
            .collect::<Vec<State<f64>>>();
        solution.y = y;

        Ok(solution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use differential_equations::methods::ExplicitRungeKutta;

    #[test]
    fn simultaneous_cno_and_tev_update() {
        let cno_dose = CnoDose::new(0.03, 4.);
        let tev_dose = TevDose::new(20., 4.);

        let cno_pk_model = CNOModel::builder().doses(vec![cno_dose]).build().unwrap();
        let model = Model::builder()
            .cno_pk_model(cno_pk_model)
            .tev_doses(vec![tev_dose])
            .tev_plasma_vd(1.)
            .tev_deg(0.1)
            .tev_cut_rate(0.05)
            .build()
            .unwrap();

        let init_state = State::zeros();
        let solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(0., 8., 1., init_state, solver).unwrap();

        let dose_idx = solution
            .t
            .iter()
            .position(|t| (*t - 4.).abs() < 1e-10)
            .unwrap();

        let dose_state = &solution.y[dose_idx];
        assert!(dose_state.peritoneal_cno > 0.);
        assert!(dose_state.plasma_tev > 0.);
    }
}
