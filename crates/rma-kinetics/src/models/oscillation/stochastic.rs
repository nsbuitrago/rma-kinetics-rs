use derive_builder::Builder;
use differential_equations::sde::SDE;
use rand::{SeedableRng as _, rngs::StdRng};

#[cfg(feature = "py")]
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pymethods};

use rand_distr::{Distribution as _, Normal};
#[cfg(feature = "py")]
use rma_kinetics_derive::StochasticPySolve;

use rma_kinetics_derive::StochasticSolve;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::State;

/// Default oscillating RMA production rate.
const DEFAULT_PROD: f64 = 0.2;
/// Default oscillation frequency.
const DEFAULT_FREQ: f64 = 0.01;
/// Default oscillating RMA blood-brain barrier transport rate.
const DEFAULT_BBB_TRANSPORT: f64 = 0.6;
/// Default oscillating RMA degradation rate.
const DEFAULT_DEG: f64 = 0.007;
const DEFAULT_PROD_STDV: f64 = 0.5;
const DEFAULT_TRANSPORT_STDV: f64 = 0.1;
const DEFAULT_SEED: u64 = 42;

fn rng_from_seed(seed: u64) -> StdRng {
    StdRng::seed_from_u64(seed)
}

#[cfg_attr(feature = "py", pyclass(name = "StochasticModel"))]
#[cfg_attr(feature = "py", derive(StochasticPySolve))]
#[cfg_attr(feature = "py", py_solve(variant = "Oscillation"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(from = "ModelSerde", into = "ModelSerde"))]
#[derive(StochasticSolve, Builder, Clone)]
#[builder(build_fn(private, name = "build_internal"), derive(Debug))]
pub struct StochasticModel {
    #[builder(default = "DEFAULT_PROD")]
    pub prod: f64,
    #[builder(default = "DEFAULT_FREQ")]
    pub freq: f64,
    #[builder(default = "DEFAULT_BBB_TRANSPORT")]
    pub bbb_transport: f64,
    #[builder(default = "DEFAULT_DEG")]
    pub deg: f64,
    /// Gaussian noise standard deviation of protein production and secretion.
    #[builder(default = "DEFAULT_PROD_STDV")]
    pub prod_noise: f64,
    #[builder(default = "DEFAULT_TRANSPORT_STDV")]
    pub transport_noise: f64,
    /// Random seed used to initialize the RNG.
    #[builder(default = "DEFAULT_SEED")]
    pub seed: u64,
    /// Random number generator.
    #[builder(setter(skip), default = "rng_from_seed(DEFAULT_SEED)")]
    pub(crate) rng: StdRng,
}

impl StochasticModel {
    /// Create a new stochastic constitutive expression model.
    pub fn new(
        prod: f64,
        freq: f64,
        bbb_transport: f64,
        deg: f64,
        prod_noise: f64,
        transport_noise: f64,
        seed: u64,
    ) -> Self {
        Self {
            prod,
            freq,
            bbb_transport,
            deg,
            prod_noise,
            transport_noise,
            seed,
            rng: rng_from_seed(seed),
        }
    }

    /// Create a new ModelBuilder for constructing a model instance.
    pub fn builder() -> StochasticModelBuilder {
        StochasticModelBuilder::default()
    }

    /// Set a new random seed and reinitialize the internal RNG.
    pub fn reseed(&mut self, seed: u64) {
        self.seed = seed;
        self.rng = rng_from_seed(seed);
    }
}

impl Default for StochasticModel {
    fn default() -> Self {
        StochasticModelBuilder::default().build().unwrap()
    }
}

#[cfg(feature = "py")]
#[pymethods]
impl StochasticModel {
    /// Create a new stochastic oscillating expression model.
    #[new]
    #[pyo3(signature = (
        prod=DEFAULT_PROD,
        freq=DEFAULT_FREQ,
        bbb_transport=DEFAULT_BBB_TRANSPORT,
        deg=DEFAULT_DEG,
        prod_noise=DEFAULT_PROD_STDV,
        transport_noise=DEFAULT_TRANSPORT_STDV,
        seed=DEFAULT_SEED,
    ))]
    pub fn create(
        prod: f64,
        freq: f64,
        bbb_transport: f64,
        deg: f64,
        prod_noise: f64,
        transport_noise: f64,
        seed: u64,
    ) -> Self {
        Self::new(
            prod,
            freq,
            bbb_transport,
            deg,
            prod_noise,
            transport_noise,
            seed,
        )
    }

    #[pyo3(name = "solve")]
    fn py_solve(
        &mut self,
        t0: f64,
        tf: f64,
        dt: f64,
        init_state: super::PyState,
        solver: crate::solve::PySolver,
    ) -> PyResult<crate::solve::PySolution> {
        let result =
            crate::solve::PyStochasticSolve::solve(self, t0, tf, dt, init_state.inner, solver);
        match result {
            Ok(solution) => Ok(solution),
            Err(e) => Err(PyValueError::new_err(format!("Failed to solve: {:?}", e))),
        }
    }
}

impl StochasticModelBuilder {
    pub fn build(&self) -> Result<StochasticModel, StochasticModelBuilderError> {
        let mut model = self.build_internal()?;
        model.rng = rng_from_seed(model.seed);
        Ok(model)
    }
}

#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
struct ModelSerde {
    prod: f64,
    freq: f64,
    bbb_transport: f64,
    deg: f64,
    prod_noise: f64,
    transport_noise: f64,
    seed: u64,
}

#[cfg(feature = "serde")]
impl From<ModelSerde> for StochasticModel {
    fn from(value: ModelSerde) -> Self {
        StochasticModel::new(
            value.prod,
            value.freq,
            value.bbb_transport,
            value.deg,
            value.prod_noise,
            value.transport_noise,
            value.seed,
        )
    }
}

#[cfg(feature = "serde")]
impl From<StochasticModel> for ModelSerde {
    fn from(value: StochasticModel) -> Self {
        Self {
            prod: value.prod,
            freq: value.freq,
            bbb_transport: value.bbb_transport,
            deg: value.deg,
            prod_noise: value.prod_noise,
            transport_noise: value.transport_noise,
            seed: value.seed,
        }
    }
}

impl SDE<f64, State<f64>> for StochasticModel {
    /// Deterministic part of the SDE.
    fn drift(&self, t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        let production = self.prod * (1.0 + (2.0 * std::f64::consts::PI * self.freq * t).sin());
        let brain_efflux = self.bbb_transport * y.brain_rma;
        dydt.brain_rma = production - brain_efflux;
        dydt.plasma_rma = brain_efflux - (self.deg * y.plasma_rma);
    }

    /// Diffusion term for oscillating RMA expression.
    fn diffusion(&self, _t: f64, y: &State<f64>, dydw: &mut State<f64>) {
        dydw.brain_rma = self.prod_noise * y.brain_rma;
        dydw.plasma_rma = self.transport_noise * y.plasma_rma;
    }

    /// Noise term for oscillating RMA expression.
    fn noise(&mut self, dt: f64, dw: &mut State<f64>) {
        let normal = Normal::new(0.0, dt.sqrt()).unwrap();
        dw.brain_rma = normal.sample(&mut self.rng);
        dw.plasma_rma = normal.sample(&mut self.rng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{SolutionAccess, StochasticSolve};
    use differential_equations::methods::ExplicitRungeKutta;

    const T0: f64 = 0.;
    const TF: f64 = 504.;
    const DT: f64 = 1.;

    #[test]
    fn default_simulation() {
        let mut model = StochasticModel::default();
        let solver = ExplicitRungeKutta::rk4(DT);
        let solution = model.solve(T0, TF, DT, State::zeros(), solver);

        assert!(solution.is_ok());
        let solution = solution.unwrap();

        assert!(solution.plasma_rma().is_ok());
        assert!(solution.plasma_dox().is_err());
        assert!(solution.max_plasma_rma().is_ok());
        assert!(solution.max_tta().is_err());

        // Should have time points from 0 to 504 with dt=1
        assert!(!solution.t.is_empty());
        assert!(!solution.y.is_empty());
        assert_eq!(solution.t.len(), solution.y.len());

        // First time point should be t0
        assert!((solution.t[0] - T0).abs() < 1e-10);
        // Last time point should be tf
        assert!((solution.t[solution.t.len() - 1] - TF).abs() < 1e-10);
    }

    #[test]
    fn deterministic_seed_reproducibility() {
        let mut model_a = StochasticModel::default();
        let mut model_b = StochasticModel::default();
        let solver_a = ExplicitRungeKutta::rk4(DT);
        let solver_b = ExplicitRungeKutta::rk4(DT);

        let solution_a = model_a.solve(T0, TF, DT, State::zeros(), solver_a).unwrap();

        let solution_b = model_b.solve(T0, TF, DT, State::zeros(), solver_b).unwrap();

        // Same seed should produce identical trajectories
        for (a, b) in solution_a.y.iter().zip(solution_b.y.iter()) {
            assert_eq!(a.brain_rma, b.brain_rma);
            assert_eq!(a.plasma_rma, b.plasma_rma);
        }
    }

    #[test]
    fn different_seed_produces_different_trajectory() {
        let mut model_a = StochasticModel::default();
        let mut model_b = StochasticModel::builder().seed(99).build().unwrap();
        let solver_a = ExplicitRungeKutta::rk4(DT);
        let solver_b = ExplicitRungeKutta::rk4(DT);

        let solution_a = model_a.solve(T0, TF, DT, State::zeros(), solver_a).unwrap();

        let solution_b = model_b.solve(T0, TF, DT, State::zeros(), solver_b).unwrap();

        // Different seeds should produce different trajectories (check a late time point)
        let last = solution_a.y.len() - 1;
        assert_ne!(solution_a.y[last].plasma_rma, solution_b.y[last].plasma_rma);
    }

    #[test]
    fn builder_pattern() {
        let result = StochasticModel::builder()
            .prod(0.3)
            .prod_noise(0.1)
            .seed(123)
            .build();

        assert!(result.is_ok());
        let mut model = result.unwrap();

        let solver = ExplicitRungeKutta::rk4(DT);
        let solution = model.solve(T0, TF, DT, State::zeros(), solver);

        assert!(solution.is_ok());
    }
}
