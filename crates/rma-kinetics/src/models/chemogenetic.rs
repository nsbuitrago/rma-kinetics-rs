use crate::{
    Solve,
    models::{
        cno::{CNOFields, CNOPKAccess, Dose, Model as CNOModel},
        dox::{DoxFields, Model as DoxModel},
    },
    pk::DoseApplyingSolout,
};
use derive_builder::Builder;
use differential_equations::{
    derive::State as StateTrait,
    error::Error,
    ode::{ODE, ODEProblem, OrdinaryNumericalMethod},
    prelude::{Interpolation, Solution},
};

/// Chemogenetic model state.
#[derive(StateTrait)]
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

        Ok(())
    }
}
