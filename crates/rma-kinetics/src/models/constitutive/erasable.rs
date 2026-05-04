use crate::{
    SolutionAccess, Solve,
    models::erasable::{
        DEFAULT_TEV_CUT_RATE, DEFAULT_TEV_DEG, DEFAULT_TEV_DOSE_NMOL, DEFAULT_TEV_DOSE_TIME,
        DEFAULT_TEV_PLASMA_VD, TevFields,
    },
    pk::{DoseApplyingSolout, validate_unique_dose_times},
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

pub use crate::models::erasable::{TevDose, create_tev_schedule};

/// Constitutive erasable model state.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(StateTrait, Default)]
pub struct State<T> {
    /// Brain RMA concentration.
    pub brain_rma: T,
    /// Plasma RMA concentration.
    pub plasma_rma: T,
    /// TEV amount in plasma compartment (nmol) during integration.
    /// Returned solutions convert this to concentration using `tev_plasma_vd`.
    pub plasma_tev: T,
}

impl State<f64> {
    /// Create a constitutive erasable model state where RMA concentrations and TEV amount
    /// are set to 0.
    pub fn zeros() -> Self {
        Self {
            brain_rma: 0.,
            plasma_rma: 0.,
            plasma_tev: 0.,
        }
    }

    /// Create a new constitutive erasable model state given RMA concentrations and TEV amount (typically nmol) .
    pub fn new(brain_rma: f64, plasma_rma: f64, plasma_tev: f64) -> Self {
        Self {
            brain_rma,
            plasma_rma,
            plasma_tev,
        }
    }
}

impl<T: std::fmt::Display> std::fmt::Display for State<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "brain_rma={:.3}, plasma_rma={:.3}, plasma_tev={:.3}",
            self.brain_rma, self.plasma_rma, self.plasma_tev
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
    #[pyo3(signature = (brain_rma=0., plasma_rma=0., plasma_tev=0.))]
    pub fn new(brain_rma: f64, plasma_rma: f64, plasma_tev: f64) -> Self {
        Self {
            inner: State::new(brain_rma, plasma_rma, plasma_tev),
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

    fn tta(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoTta)
    }

    fn max_tta(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoTta)
    }

    fn brain_dox(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoBrainDox)
    }

    fn max_brain_dox(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoBrainDox)
    }

    fn plasma_dox(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoPlasmaDox)
    }

    fn max_plasma_dox(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoPlasmaDox)
    }

    fn dreadd(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoDreadd)
    }

    fn max_dreadd(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoDreadd)
    }

    fn peritoneal_cno(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoPeritonealCno)
    }

    fn max_peritoneal_cno(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoPeritonealCno)
    }

    fn plasma_cno(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoPlasmaCno)
    }

    fn max_plasma_cno(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoPlasmaCno)
    }

    fn brain_cno(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoBrainCno)
    }

    fn max_brain_cno(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoBrainCno)
    }

    fn plasma_clz(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoPlasmaClz)
    }

    fn max_plasma_clz(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoPlasmaClz)
    }

    fn brain_clz(&self) -> Result<Vec<f64>, SpeciesAccessError> {
        Err(SpeciesAccessError::NoBrainClz)
    }

    fn max_brain_clz(&self) -> Result<(f64, f64), SpeciesAccessError> {
        Err(SpeciesAccessError::NoBrainClz)
    }
}

/// So far we use the same default parameters as constitutive RMA,
/// but define them separately in erasable in case they diverge later.
const DEFAULT_PROD: f64 = 0.2;
const DEFAULT_BBB_TRANSPORT: f64 = 0.6;
const DEFAULT_DEG: f64 = 0.007;

#[cfg_attr(feature = "py", pyclass)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, Clone, Builder)]
#[builder(derive(Debug), build_fn(validate = "Self::validate"))]
pub struct Model {
    /// RMA production rate.
    #[builder(default = "DEFAULT_PROD")]
    pub rma_prod: f64,
    /// RMA blood-brain barrier transport rate.
    #[builder(default = "DEFAULT_BBB_TRANSPORT")]
    pub rma_bbb_transport: f64,
    /// RMA degradation rate.
    #[builder(default = "DEFAULT_DEG")]
    pub rma_deg: f64,
    /// TEV administration schedule in plasma (nmol bolus doses).
    #[builder(default = "vec![TevDose::new(DEFAULT_TEV_DOSE_NMOL, DEFAULT_TEV_DOSE_TIME)]")]
    pub doses: Vec<TevDose>,
    /// TEV plasma volume of distribution used for converting amount to concentration.
    #[builder(default = "DEFAULT_TEV_PLASMA_VD")]
    pub tev_plasma_vd: f64,
    /// TEV degradation rate.
    #[builder(default = "DEFAULT_TEV_DEG")]
    pub tev_deg: f64,
    /// TEV-dependent RMA cutting rate.
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
    fn validate(&self) -> Result<(), String> {
        if let Some(doses) = self.doses.as_deref() {
            validate_unique_dose_times(doses).map_err(|e| e.to_string())
        } else {
            Ok(())
        }
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (doses=vec![TevDose::new(DEFAULT_TEV_DOSE_NMOL, DEFAULT_TEV_DOSE_TIME)], rma_prod=DEFAULT_PROD, rma_bbb_transport=DEFAULT_BBB_TRANSPORT, rma_deg=DEFAULT_DEG, tev_plasma_vd=DEFAULT_TEV_PLASMA_VD, tev_deg=DEFAULT_TEV_DEG, tev_cut_rate=DEFAULT_TEV_CUT_RATE))]
    pub fn create(
        doses: Vec<TevDose>,
        rma_prod: f64,
        rma_bbb_transport: f64,
        rma_deg: f64,
        tev_plasma_vd: f64,
        tev_deg: f64,
        tev_cut_rate: f64,
    ) -> PyResult<Self> {
        validate_unique_dose_times(&doses).map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            rma_prod,
            rma_bbb_transport,
            rma_deg,
            doses,
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
                inner: InnerSolution::Erasable(solution),
            }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to solve: {:?}", e))),
        }
    }

    #[getter]
    fn get_doses(&self) -> Vec<TevDose> {
        self.doses.clone()
    }

    #[setter]
    fn set_doses(&mut self, doses: Vec<TevDose>) -> PyResult<()> {
        self.doses = doses;
        Ok(())
    }
}

impl ODE<f64, State<f64>> for Model {
    /// System of differential equations describing constitutive feRMA expression
    /// in brain tissue and blood-brain barrier transport to plasma.
    fn diff(&self, _t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        let brain_efflux = self.rma_bbb_transport * y.brain_rma;
        let tev_conc = y.plasma_tev / self.tev_plasma_vd;
        let cleaved_rma = self.tev_cut_rate * y.plasma_rma * tev_conc;

        dydt.brain_rma = self.rma_prod - brain_efflux;
        dydt.plasma_rma = brain_efflux - (self.rma_deg * y.plasma_rma) - cleaved_rma;
        dydt.plasma_tev = -(self.tev_deg * y.plasma_tev);
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
            .doses
            .iter()
            .filter_map(|dose| {
                if (dose.time - t0).abs() < 1e-10 {
                    adjusted_init_state.plasma_tev += dose.nmol;
                    None
                } else {
                    Some(dose.clone())
                }
            })
            .collect::<Vec<TevDose>>();

        let mut dosing_solout =
            DoseApplyingSolout::<State<f64>, TevDose>::new(scheduled_updates, t0, tf, dt);

        let problem = ODEProblem::new(self, t0, tf, adjusted_init_state);
        let mut solution = problem.solout(&mut dosing_solout).solve(solver)?;

        // Return TEV as concentration using configured Vd.
        let y = solution
            .y
            .iter()
            .map(|state| State {
                brain_rma: state.brain_rma,
                plasma_rma: state.plasma_rma,
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
    use differential_equations::{methods::ExplicitRungeKutta, status::Status};

    #[test]
    fn tev_dose_creation() {
        let dose = TevDose::new(20., 4.);
        assert_eq!(dose.nmol, 20.);
        assert_eq!(dose.time, 4.);

        let schedule = create_tev_schedule(20., 4., Some(2), Some(24.));
        assert_eq!(schedule.len(), 3);
        assert_eq!(schedule[0].time, 4.);
        assert_eq!(schedule[1].time, 28.);
        assert_eq!(schedule[2].time, 52.);
    }

    #[test]
    fn erasable_model_simulation() -> Result<(), Box<dyn std::error::Error>> {
        let mut solver = ExplicitRungeKutta::dopri5();
        let t0 = 0.;
        let tf = 24.;
        let dt = 1.;
        let init_state = State::zeros();

        let dose = TevDose::new(10., 1.);
        let model = Model::builder()
            .doses(vec![dose.clone()])
            .tev_plasma_vd(2.)
            .build()?;
        let solution = model.solve(t0, tf, dt, init_state, &mut solver)?;

        assert!(matches!(solution.status, Status::Complete));
        assert_eq!(solution.y[1].plasma_tev, dose.nmol / 2.);
        assert!(solution.plasma_tev().is_ok());
        assert!(solution.plasma_cno().is_err());

        Ok(())
    }

    #[test]
    fn expected_ts() -> Result<(), Box<dyn std::error::Error>> {
        let model = Model::builder()
            .doses(vec![TevDose::new(10., 1.)])
            .build()?;
        let dt = 1.;
        let t0 = 0.;
        let tf = 10.;
        let init_state = State::zeros();
        let mut solver = ExplicitRungeKutta::dopri5();

        let solution = model.solve(t0, tf, dt, init_state, &mut solver)?;
        assert!(matches!(solution.status, Status::Complete));
        let expected_len = ((tf - t0) / dt).ceil() as usize + 1;
        assert_eq!(solution.y.len(), expected_len);

        let model = Model::builder()
            .doses(vec![TevDose::new(10., 1.5)])
            .build()?;
        let solution = model.solve(t0, tf, dt, init_state, &mut solver)?;
        assert!(matches!(solution.status, Status::Complete));
        let uneven_expected_len = ((tf - t0) / dt).ceil() as usize + 2;
        assert_eq!(solution.y.len(), uneven_expected_len);
        assert_eq!(solution.t[0], t0);
        assert_eq!(solution.t[2], 1.5);

        Ok(())
    }

    #[test]
    fn t0_dose_is_preapplied() -> Result<(), Box<dyn std::error::Error>> {
        let model = Model::builder()
            .doses(vec![TevDose::new(12., 0.)])
            .tev_plasma_vd(3.)
            .build()?;
        let mut solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(0., 2., 1., State::zeros(), &mut solver)?;

        assert!(solution.y[0].plasma_tev > 0.);
        assert_eq!(solution.y[0].plasma_tev, 12. / 3.);

        Ok(())
    }

    #[test]
    fn erasable_model_rejects_duplicate_nonzero_dose_times() {
        let duplicate_doses = vec![TevDose::new(10., 1.), TevDose::new(20., 1.)];
        let result = Model::builder().doses(duplicate_doses).build();

        assert!(result.is_err());
    }
}
