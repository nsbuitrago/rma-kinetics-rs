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

use crate::{
    SolutionAccess, Solve,
    models::{
        cno::{CNOFields, CNOPKAccess, Dose, Model as CNOModel},
        dox::{DoxFields, Model as DoxModel},
    },
    pk::DoseApplyingSolout,
    solve::SpeciesAccessError,
};
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
use crate::solve::{InnerSolution, PySolution, PySolver};

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
}

impl ODE<f64, State<f64>> for Model {
    fn diff(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        self.dox_pk_model.diff_with(t, y, dydt); // dox dynamics
        self.cno_pk_model.diff_with(t, y, dydt); // cno dynamics

        // DREADD induced tTA expression
        let cno_ec50_hill = (y.brain_cno / self.cno_pk_model.cno_brain_vd / self.cno_ec50)
            .powf(self.cno_cooperativity);
        let clz_ec50_hill = (y.brain_clz / self.cno_pk_model.clz_brain_vd / self.clz_ec50)
            .powf(self.clz_cooperativity);
        let active_dreadd_frac =
            (cno_ec50_hill + clz_ec50_hill) / (1. + cno_ec50_hill + clz_ec50_hill);
        let dreadd_mod =
            (active_dreadd_frac * y.dreadd / self.dreadd_ec50).powf(self.dreadd_cooperativity);

        dydt.tta = ((self.leaky_tta_prod + (self.tta_prod * dreadd_mod)) / (1. + dreadd_mod))
            - (self.tta_deg * y.tta);

        // constitutive DREADD expression
        dydt.dreadd = self.dreadd_prod - (self.dreadd_deg * y.dreadd);

        // tet inducible RMA expression
        let active_tta = 1. / (1. + y.brain_dox / self.dox_tta_kd);
        let tta_hill = (active_tta * y.tta / self.tta_kd).powf(self.tta_cooperativity);
        dydt.brain_rma = (self.leaky_rma_prod + (self.rma_prod * tta_hill)) / (1. + tta_hill)
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
        let mut start_dose_idx = 0;
        let n_applied_doses = &self
            .cno_pk_model
            .doses
            .iter()
            .filter(|dose| (dose.time - t0).abs() < 1e-10)
            .map(|dose| *adjusted_init_state.peritoneal_cno_mut() += dose.nmol)
            .count();
        start_dose_idx += n_applied_doses;

        let mut dosing_solout = DoseApplyingSolout::<State<f64>>::new(
            self.get_doses()[start_dose_idx..].to_vec(),
            t0,
            tf,
            dt,
        );
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
}
