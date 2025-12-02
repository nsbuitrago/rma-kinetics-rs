use crate::models::dox::{DoxFields, Model as DoxModel};
use differential_equations::{derive::State as StateTrait, ode::ODE};
use rma_kinetics_derive::Solve;

#[cfg(feature = "py")]
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pymethods};

#[cfg(feature = "py")]
use rma_kinetics_derive::PySolve;

/// Tet-Off model state.
#[derive(StateTrait)]
pub struct State<T> {
    pub brain_rma: T,
    pub plasma_rma: T,
    pub tta: T,
    pub plasma_dox: T,
    pub brain_dox: T,
}

impl State<f64> {
    /// Create a new Tet-Off model state where all concentrations are set to zero.
    pub fn zeros() -> Self {
        Self {
            brain_rma: 0.,
            plasma_rma: 0.,
            tta: 0.,
            brain_dox: 0.,
            plasma_dox: 0.,
        }
    }

    /// Create a new Tet-Off model state given brain RMA, plasma RMA, tTA, brain dox, and plasma dox concentrations.
    pub fn new(brain_rma: f64, plasma_rma: f64, tta: f64, brain_dox: f64, plasma_dox: f64) -> Self {
        Self {
            brain_rma,
            plasma_rma,
            tta,
            brain_dox,
            plasma_dox,
        }
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
    #[pyo3(signature = (brain_rma=0., plasma_rma=0., tta=0., brain_dox=0., plasma_dox=0.))]
    pub fn new(brain_rma: f64, plasma_rma: f64, tta: f64, brain_dox: f64, plasma_dox: f64) -> Self {
        Self {
            inner: State::new(brain_rma, plasma_rma, tta, brain_dox, plasma_dox),
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
    fn get_brain_dox(&self) -> f64 {
        self.inner.brain_dox
    }

    #[getter]
    fn get_plasma_dox(&self) -> f64 {
        self.inner.plasma_dox
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
    fn set_brain_dox(&mut self, value: f64) -> PyResult<()> {
        self.inner.brain_dox = value;
        Ok(())
    }

    #[setter]
    fn set_plasma_dox(&mut self, value: f64) -> PyResult<()> {
        self.inner.plasma_dox = value;
        Ok(())
    }
}

#[cfg_attr(feature = "py", pyclass)]
#[cfg_attr(feature = "py", derive(PySolve))]
#[cfg_attr(feature = "py", py_solve(variant = "TetOff"))]
#[derive(Solve)]
pub struct Model {
    pub rma_prod: f64,
    pub leaky_rma_prod: f64,
    pub rma_bbb_transport: f64,
    pub rma_deg: f64,
    pub tta_prod: f64,
    pub tta_deg: f64,
    pub tta_kd: f64,
    pub tta_cooperativity: f64,
    pub dox_pk_model: DoxModel,
    pub dox_tta_kd: f64,
}

const DEFAULT_RMA_PROD: f64 = 0.2;
const DEFAULT_LEAKY_RMA_PROD: f64 = 0.002;
const DEFAULT_RMA_BBB_TRANSPORT: f64 = 0.6;
const DEFAULT_RMA_DEG: f64 = 0.007;
const DEFAULT_TTA_PROD: f64 = 10.;
const DEFAULT_TTA_DEG: f64 = 1.;
const DEFAULT_TTA_KD: f64 = 10.;
const DEFAULT_TTA_COOPERATIVITY: f64 = 2.;
const DEFAULT_DOX_TTA_KD: f64 = 10.;

#[cfg(feature = "py")]
#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (rma_prod=DEFAULT_RMA_PROD, leaky_rma_prod=DEFAULT_LEAKY_RMA_PROD, rma_bbb_transport=DEFAULT_RMA_BBB_TRANSPORT, rma_deg=DEFAULT_RMA_DEG, tta_prod=DEFAULT_TTA_PROD, tta_deg=DEFAULT_TTA_DEG, tta_kd=DEFAULT_TTA_KD, tta_cooperativity=DEFAULT_TTA_COOPERATIVITY, dox_pk_model=DoxModel::default(), dox_tta_kd=DEFAULT_DOX_TTA_KD))]
    pub fn create(
        rma_prod: f64,
        leaky_rma_prod: f64,
        rma_bbb_transport: f64,
        rma_deg: f64,
        tta_prod: f64,
        tta_deg: f64,
        tta_kd: f64,
        tta_cooperativity: f64,
        dox_pk_model: DoxModel,
        dox_tta_kd: f64,
    ) -> Self {
        Self {
            rma_prod,
            leaky_rma_prod,
            rma_bbb_transport,
            rma_deg,
            tta_prod,
            tta_deg,
            tta_kd,
            tta_cooperativity,
            dox_pk_model,
            dox_tta_kd,
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
    /// Create a new Tet-Off model builder.
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
    }
}

impl ODE<f64, State<f64>> for Model {
    fn diff(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        // tet inducible RMA expression
        let active_tta = 1. / (1. + y.brain_dox / self.dox_tta_kd);
        let tta_hill = (active_tta * y.tta / self.tta_kd).powf(self.tta_cooperativity);
        dydt.brain_rma = (self.leaky_rma_prod + (self.rma_prod * tta_hill)) / (1. + tta_hill)
            - (self.rma_bbb_transport * y.brain_rma);

        let brain_efflux = self.rma_bbb_transport * y.brain_rma;
        dydt.plasma_rma = brain_efflux - (self.rma_deg * y.plasma_rma);

        // constitutive tTA expression
        dydt.tta = self.tta_prod - self.tta_deg * y.tta;

        // dox dynamics
        self.dox_pk_model.diff_with(t, y, dydt);
    }
}

impl Default for Model {
    fn default() -> Self {
        ModelBuilder::default().build()
    }
}

/// Tet-Off model builder.
pub struct ModelBuilder {
    pub rma_prod: f64,
    pub leaky_rma_prod: f64,
    pub rma_bbb_transport: f64,
    pub rma_deg: f64,
    pub tta_prod: f64,
    pub tta_deg: f64,
    pub tta_kd: f64,
    pub tta_cooperativity: f64,
    pub dox_pk_model: DoxModel,
    pub dox_tta_kd: f64,
}

impl Default for ModelBuilder {
    fn default() -> Self {
        Self {
            rma_prod: DEFAULT_RMA_PROD,
            leaky_rma_prod: DEFAULT_LEAKY_RMA_PROD,
            rma_bbb_transport: DEFAULT_RMA_BBB_TRANSPORT,
            rma_deg: DEFAULT_RMA_DEG,
            tta_prod: DEFAULT_TTA_PROD,
            tta_deg: DEFAULT_TTA_DEG,
            tta_kd: DEFAULT_TTA_KD,
            tta_cooperativity: DEFAULT_TTA_COOPERATIVITY,
            dox_pk_model: DoxModel::default(),
            dox_tta_kd: DEFAULT_DOX_TTA_KD,
        }
    }
}

impl ModelBuilder {
    /// Set the RMA production rate (concentration/time)
    pub fn rma_prod(&mut self, rma_prod: f64) -> &mut Self {
        self.rma_prod = rma_prod;
        self
    }

    /// Set the leaky RMA production rate (concentration/time)
    pub fn leaky_rma_prod(&mut self, leaky_rma_prod: f64) -> &mut Self {
        self.leaky_rma_prod = leaky_rma_prod;
        self
    }

    /// Set the RMA BBB transport rate (1/time)
    pub fn rma_bbb_transport(&mut self, rma_bbb_transport: f64) -> &mut Self {
        self.rma_bbb_transport = rma_bbb_transport;
        self
    }

    /// Set the RMA degradation rate (1/time)
    pub fn rma_deg(&mut self, rma_deg: f64) -> &mut Self {
        self.rma_deg = rma_deg;
        self
    }

    /// Set the tTA production rate (concentration/time)
    pub fn tta_prod(&mut self, tta_prod: f64) -> &mut Self {
        self.tta_prod = tta_prod;
        self
    }

    /// Set the tTA degradation rate (1/time)
    pub fn tta_deg(&mut self, tta_deg: f64) -> &mut Self {
        self.tta_deg = tta_deg;
        self
    }

    /// Set the tTA KD (concentration)
    pub fn tta_kd(&mut self, tta_kd: f64) -> &mut Self {
        self.tta_kd = tta_kd;
        self
    }

    /// Set the tTA cooperativity
    pub fn tta_cooperativity(&mut self, tta_cooperativity: usize) -> &mut Self {
        self.tta_cooperativity = tta_cooperativity as f64;
        self
    }

    /// Set the dox PK model
    pub fn dox_pk_model(&mut self, dox_pk_model: DoxModel) -> &mut Self {
        self.dox_pk_model = dox_pk_model;
        self
    }

    /// Set the dox TTA KD (concentration)
    pub fn dox_tta_kd(&mut self, dox_tta_kd: f64) -> &mut Self {
        self.dox_tta_kd = dox_tta_kd;
        self
    }

    /// Build the Tet-Off model
    pub fn build(&self) -> Model {
        Model {
            rma_prod: self.rma_prod,
            leaky_rma_prod: self.leaky_rma_prod,
            rma_bbb_transport: self.rma_bbb_transport,
            rma_deg: self.rma_deg,
            tta_prod: self.tta_prod,
            tta_deg: self.tta_deg,
            tta_kd: self.tta_kd,
            tta_cooperativity: self.tta_cooperativity,
            dox_pk_model: self.dox_pk_model.clone(),
            dox_tta_kd: self.dox_tta_kd,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::dox::AccessPeriod;
    use crate::solve::Solve;
    use differential_equations::methods::ExplicitRungeKutta;

    #[test]
    fn state_creation() {
        let zero_state = State::zeros();
        assert_eq!(zero_state.brain_rma, 0.);
        assert_eq!(zero_state.plasma_rma, 0.);
        assert_eq!(zero_state.tta, 0.);
        assert_eq!(zero_state.brain_dox, 0.);
        assert_eq!(zero_state.plasma_dox, 0.);

        let custom_state = State::new(0., 0., 10., 0., 0.);
        assert_eq!(custom_state.brain_rma, 0.);
        assert_eq!(custom_state.plasma_rma, 0.);
        assert_eq!(custom_state.tta, 10.);
        assert_eq!(custom_state.brain_dox, 0.);
        assert_eq!(custom_state.plasma_dox, 0.);
    }

    #[test]
    fn model_creation() {
        // default model
        let model = Model::default();
        assert_eq!(model.rma_prod, DEFAULT_RMA_PROD);

        // custom model
        let dox_access_period = AccessPeriod::new(40., 0.0..=24.);
        let custom_dox_model = DoxModel::builder()
            .schedule(vec![dox_access_period])
            .build();
        let custom_model = Model::builder()
            .rma_prod(0.5)
            .dox_pk_model(custom_dox_model)
            .build();

        assert_eq!(custom_model.rma_prod, 0.5);
        assert_eq!(custom_model.dox_pk_model.schedule.len(), 1);
        assert_eq!(custom_model.dox_pk_model.schedule[0].dose, 40.);
    }

    #[test]
    fn model_simulation() {
        let default_model = Model::default();
        let mut solver = ExplicitRungeKutta::dopri5();
        let init_state = State::zeros();

        let solution = default_model.solve(0., 24., 1., init_state, &mut solver);
        assert!(solution.is_ok());

        let unwrapped_solution = solution.unwrap();
        assert!(unwrapped_solution.y.last().unwrap().brain_rma > 0.);
        assert!(unwrapped_solution.y.last().unwrap().plasma_rma > 0.);
        assert_eq!(unwrapped_solution.y.last().unwrap().plasma_dox, 0.);

        // custom model with dox administration
        let dox_access_period = AccessPeriod::new(40., 0.0..=24.);
        let custom_dox_model = DoxModel::builder()
            .schedule(vec![dox_access_period])
            .build();
        let custom_model = Model::builder().dox_pk_model(custom_dox_model).build();
        let solution = custom_model.solve(0., 36., 1., init_state, &mut solver);
        assert!(solution.is_ok());

        let unwrapped_solution = solution.unwrap();
        assert!(unwrapped_solution.y.last().unwrap().brain_rma > 0.);
        assert!(unwrapped_solution.y.last().unwrap().brain_dox > 0.);
        assert!(unwrapped_solution.y[1].plasma_dox > 0.);

        assert_eq!(unwrapped_solution.y.len(), 37);
    }
}
