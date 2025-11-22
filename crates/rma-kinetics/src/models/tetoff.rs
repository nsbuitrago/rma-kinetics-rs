use crate::models::dox::{DoxFields, Model as DoxModel};
use differential_equations::{derive::State as StateTrait, ode::ODE, prelude::Matrix};
use rma_kinetics_derive::Solve;

#[derive(StateTrait)]
pub struct State<T> {
    pub brain_rma: T,
    pub plasma_rma: T,
    pub tta: T,
    pub plasma_dox: T,
    pub brain_dox: T,
}

impl State<f64> {
    pub fn zeros() -> Self {
        Self {
            brain_rma: 0.,
            plasma_rma: 0.,
            tta: 0.,
            brain_dox: 0.,
            plasma_dox: 0.,
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

impl ODE<f64, State<f64>> for Model {
    fn diff(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        // tet inducible RMA expression
        let active_tta = 1. / (1. + y.brain_dox / self.dox_tta_kd);
        let tta_hill = (active_tta * y.tta / self.tta_kd).powf(self.tta_cooperativity);
        dydt.brain_rma = (self.leaky_rma_prod + (self.rma_prod * tta_hill)) / (1. + tta_hill) - (self.rma_bbb_transport * y.brain_rma);

        let brain_efflux = self.rma_bbb_transport * y.brain_rma;
        dydt.plasma_rma = brain_efflux - (self.rma_deg * y.plasma_rma);

        // constitutive tTA expression
        dydt.tta = self.tta_prod - self.tta_deg * y.tta;

        // dox dynamics
        self.dox_pk_model.diff_with(t, y, dydt);
    }

    // fn jacobian(&self, t: f64, y: &State<f64>, j: &mut Matrix<f64>) {

    //     // self.dox_pk_model.jacobian_with(t, y, j);
    //     let j_dox = self.dox_pk_model.jacobian_with(t, y, j);

    //     j[(0, 0)] = -self.rma_bbb_transport; // df0/d(brain RMA)
    //     j[(0, 1)] = 0.; // df0/d(plasma RMA)
    //     // tTA
    //     let active_tta = 1. / (1. + y.brain_dox / self.dox_tta_kd);
    //     let ta_mod = (active_tta * y.tta / self.tta_kd).powf(self.tta_cooperativity);
    //     j[(0, 2)] = (self.rma_prod * self.tta_cooperativity * active_tta / self.tta_kd).powf(self.tta_cooperativity)
    //         * y.tta.powf(self.tta_cooperativity - 1.)
    //         - (self.leaky_rma_prod * self.tta_cooperativity * active_tta).powf(self.tta_cooperativity)
    //         * y.tta.powf(self.tta_cooperativity - 1.)
    //         / (1. + ta_mod).powf(2.);

    //     // j[(0, 3)] = j_dox[(0, 0)]; // brain dox
    //     // j[(0, 4)] = j_dox[(0, 1)]; // plasma dox

    //     j[(1, 0)] = self.rma_bbb_transport; // df1/d(brain RMA)
    //     j[(1, 1)] = -self.rma_deg; // df1/d(plasma RMA)
    //     j[(1, 2)] = 0.; // df1/d(tTA)
    //     j[(1, 3)] = j_dox[(1, 0)]; // df1/d(brain dox)
    //     j[(1, 4)] = j_dox[(1, 1)]; // df1/d(plasma dox)

    //     j[(2, 0)] = 0.; // df2/d(brain RMA)
    //     j[(2, 1)] = 0.; // df2/d(plasma RMA)
    //     j[(2, 2)] = -self.tta_deg; // df2/d(tTA)
    //     j[(2, 3)] = 0.; // df2/d(brain dox)
    //     j[(2, 4)] = 0.; // df2/d(plasma dox)



    // }
}
