//! Constitutive RMA expression model.
//!
//! The constitutive model is a simple model that describes the expression of a synthetic serum reporter
//! in the brain tissue and blood-brain barrier transport to the plasma.
//!
//! ## Parameters
//!
//! Reporter transcription, translation, and secretion is consolidated into a single term.
//! Transport is assumed to be mainly via Fc receptor mediated reverse-transcytosis and
//! degradation is assumed to be mainly by protein degradation. Degradation by cell division
//! is assumed to be negligible for neuronal cell types.
//!
//! The default parameters are based on the constitutive expression of human-synapsin promoter in CA1 hippocampus.
//! - Production rate: 0.2 nM/hr
//! - Blood-brain barrier transport rate: 0.6 1/hr
//! - Degradation rate: 0.007 1/hr
//!
//! To solve the model over a given period of time, we use the solvers provided by
//! the `differential_equations` dependency. From here, we can use the provided `Solve`
//! trait and use the `solve` method on our model.
//!
//! ```rust
//! use rma_kinetics::{models::constitutive, Solve};
//! use differential_equations::methods::ExplicitRungeKutta;
//!
//! let model = constitutive::Model::default();
//! let init_state = constitutive::State::zeros();
//! let solver = ExplicitRungeKutta::dopri5();
//!
//! let solution = model.solve(0., 100., 1., init_state, solver);
//! assert!(solution.is_ok());
//!
//! let solution = solution.unwrap();
//! println!("{:?}", solution.y);
//! ```

pub mod erasable;
pub mod stochastic;

use nalgebra::{Matrix2, Matrix3x2, SVector};
pub use stochastic::StochasticModel;

use derive_builder::Builder;
use differential_equations::{
    derive::State as StateTrait,
    ivp::IVP,
    ode::{ODE, OrdinaryNumericalMethod},
    prelude::{Interpolation, Matrix, Solution},
};
use rma_kinetics_derive::Solve;

use crate::{
    impl_solution_access_basic_rma,
    inference::{DEFAULT_WEIGHT, Observation},
};

#[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
use crate::solve::ToDataFrame;

#[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
use polars::{error::PolarsError, frame::DataFrame};

#[cfg(feature = "py")]
use pyo3::{PyResult, exceptions::PyValueError, pyclass, pymethods};

#[cfg(feature = "py")]
use rma_kinetics_derive::PySolve;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Constitutive model state.
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(StateTrait)]
pub struct State<T> {
    pub brain_rma: T,
    pub plasma_rma: T,
}

impl State<f64> {
    /// Get a constitutive model state where brain and plasma RMA concentration
    /// are set to 0.
    pub fn zeros() -> Self {
        Self {
            brain_rma: 0.,
            plasma_rma: 0.,
        }
    }

    /// Create a new constitutive model state given brain and plasma RMA concentrations.
    pub fn new(brain_rma: f64, plasma_rma: f64) -> Self {
        Self {
            brain_rma,
            plasma_rma,
        }
    }
}

impl Default for State<f64> {
    /// Default constitutive model state where brain and plasma RMA concentration
    /// are set to 0.
    fn default() -> Self {
        State::zeros()
    }
}

impl<T: std::fmt::Display> std::fmt::Display for State<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "brain_rma={:.3}, plasma_rma={:.3}",
            self.brain_rma, self.plasma_rma
        )
    }
}

impl_solution_access_basic_rma!(Solution<f64, State<f64>>, State<f64>);

#[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
impl ToDataFrame for Solution<f64, State<f64>> {
    fn to_dataframe(self) -> Result<DataFrame, PolarsError> {
        use crate::struct_to_dataframe;

        struct_to_dataframe!(self, [brain_rma, plasma_rma])
    }
}

#[cfg(feature = "py")]
macro_rules! create_interface {
    ($name: ident, $type: ident) => {
        #[derive(Clone)]
        #[pyclass(name = "State")]
        pub struct $name {
            pub inner: State<$type>,
        }
        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (brain_rma=0., plasma_rma=0.))]
            pub fn new(brain_rma: $type, plasma_rma: $type) -> Self {
                Self {
                    inner: State {
                        brain_rma,
                        plasma_rma,
                    },
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
        }
    };
}

#[cfg(feature = "py")]
create_interface!(PyState, f64);

#[cfg(feature = "py")]
impl std::fmt::Display for PyState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// Default constitutive RMA production rate.
const DEFAULT_PROD: f64 = 0.2;
/// Default constitutive RMA blood-brain barrier transport rate.
const DEFAULT_BBB_TRANSPORT: f64 = 0.6;
/// Default constitutive RMA degradation rate.
const DEFAULT_DEG: f64 = 0.007;

/// Constitutive RMA expression model.
///
/// The [`default`](Model::default), [`new`](Model::new) or [`builder`](Model::builder)
/// methods can be used to create a new model instance. See `solve` for more
/// information on integration.
#[cfg_attr(feature = "py", pyclass)]
#[cfg_attr(feature = "py", derive(PySolve))]
#[cfg_attr(feature = "py", py_solve(variant = "Constitutive"))]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Solve, Builder)]
#[builder(derive(Debug))]
pub struct Model {
    /// RMA production rate.
    #[builder(default = "DEFAULT_PROD")]
    pub prod: f64,
    /// RMA blood-brain barrier transport rate.
    #[builder(default = "DEFAULT_BBB_TRANSPORT")]
    pub bbb_transport: f64,
    /// RMA degradation rate.
    #[builder(default = "DEFAULT_DEG")]
    pub deg: f64,
}

#[cfg(feature = "py")]
#[pymethods]
impl Model {
    /// Create a new constitutive expression model given RMA production, blood-brain
    /// barrier transport, and degradation rates.
    #[new]
    #[pyo3(signature = (prod=DEFAULT_PROD, bbb_transport=DEFAULT_BBB_TRANSPORT, deg=DEFAULT_DEG))]
    pub fn create(prod: f64, bbb_transport: f64, deg: f64) -> Self {
        Self {
            prod,
            bbb_transport,
            deg,
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
            Err(e) => Err(PyValueError::new_err(format!("Failed to solve: {:?}", e))),
        }
    }
}

impl Model {
    /// Create a new constitutive expression model given RMA production, blood-brain
    /// barrier transport, and degradation rates.
    pub fn new(prod: f64, bbb_transport: f64, deg: f64) -> Self {
        Self {
            prod,
            bbb_transport,
            deg,
        }
    }

    /// Create a new ModelBuilder for constructing a model instance. This is useful
    /// if you need to update a single rate parameter for example.
    pub fn builder() -> ModelBuilder {
        ModelBuilder::default()
    }

    /// Get parameters of the model as a flat vector.
    pub fn get_parameters(&self) -> SVector<f64, 3> {
        SVector::<f64, 3>::new(self.prod, self.bbb_transport, self.deg)
    }
}

impl Default for Model {
    /// Create a new constitutive model instance with the default parameters
    /// for CA1 hippocampus expression driven by a human-synapsin promoter.
    fn default() -> Self {
        ModelBuilder::default().build().unwrap()
    }
}

impl ODE<f64, State<f64>> for Model {
    /// System of differential equations describing constitutive RMA expression
    /// in the brain tissue and blood-brain barrier transport to the plasma.
    fn diff(&self, _t: f64, y: &State<f64>, dydt: &mut State<f64>) {
        let brain_efflux = self.bbb_transport * y.brain_rma;
        dydt.brain_rma = self.prod - brain_efflux;
        dydt.plasma_rma = brain_efflux - (self.deg * y.plasma_rma);
    }

    fn jacobian(&self, _t: f64, _y: &State<f64>, j: &mut Matrix<f64>) {
        j[(0, 0)] = -self.bbb_transport;
        j[(0, 1)] = 0.;
        j[(1, 0)] = self.bbb_transport;
        j[(1, 1)] = -self.deg;
    }
}

/// Alias for the adjoint state vector where the first 2 elements are lambda
/// (which is the actual adjoint state) and the last 3 elements are µ
/// (the parameter gradients). We call this `AdjointState`. since we
/// use it to represent the combined state during the adjoint solve.
pub type AdjointState = SVector<f64, 5>;

/// The adjoint ODE model for sensitivity analysis and optimization.
pub struct AdjointModel {
    /// A flat vector of model parameters. Order is [prod, bbb_transport, deg].
    parameters: SVector<f64, 3>,
    /// The forward solution of the constitutive model.
    forward_solution: Solution<f64, State<f64>>,
}

impl AdjointModel {
    /// Create a new adjoint problem from the given parameters and forward solution.
    #[inline]
    pub fn new(parameters: SVector<f64, 3>, forward_solution: Solution<f64, State<f64>>) -> Self {
        Self {
            parameters,
            forward_solution,
        }
    }

    /// Simple linear interpolation of the forward solution at time `t`.
    /// For this model we keep is simple and use a relatively dense time grid to
    /// avoid having to use checkpointing and re-solve the forward pass during
    /// the adjoint solve.
    pub fn forward(&self, t: f64) -> State<f64> {
        let times = &self.forward_solution.t;
        let states = &self.forward_solution.y;

        if t <= times[0] {
            return states[0];
        }

        if let Some(time) = times.last() {
            if t >= *time && states.last().is_some() {
                return *states.last().unwrap();
            }
        }

        let upper = times.partition_point(|ti| *ti < t);
        let lower = upper - 1;
        let s = (t - times[lower]) / (times[upper] - times[lower]);

        states[lower] * (1.0 - s) + states[upper] * s
    }

    /// Solve the adjoint ODE given the forward solution and plasma RMA observations.
    pub fn solve<S>(
        &self,
        tf: f64,
        t0: f64,
        init_state: AdjointState,
        observations: &mut [Observation],
        solver_factory: impl Fn() -> S,
    ) -> Result<Solution<f64, AdjointState>, differential_equations::error::Error<f64, AdjointState>>
    where
        S: OrdinaryNumericalMethod<f64, AdjointState> + Interpolation<f64, AdjointState>,
    {
        observations.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        observations.reverse(); // we'll reverse so that we can iterate backwards to the

        let mut current_time = tf;
        let mut current_state = init_state;
        let mut full_solution = Solution::new();

        full_solution.t.push(current_time);
        full_solution.y.push(current_state);

        let mut i = 0;
        while i < observations.len() {
            let obs_time = observations[i].time;

            if obs_time < t0 || obs_time > tf {
                return Err(differential_equations::error::Error::BadInput {
                    msg: "observation time out of bounds".to_string(),
                });
            }

            let mut jump = SVector::<f64, 2>::zeros();

            while i < observations.len() && observations[i].time == obs_time {
                jump += self.observation_jump(&observations[i]);
                i += 1;
            }

            if obs_time < current_time {
                let segment = IVP::ode(self, current_time, obs_time, current_state)
                    .method(solver_factory())
                    .solve()?;

                current_state = *segment.y.last().unwrap();
                append_segment(&mut full_solution, segment);
                current_time = obs_time;
            }

            apply_jump(&mut current_state, jump);
            push_state(&mut full_solution, current_time, current_state);
        }

        if current_time > t0 {
            let segment = IVP::ode(self, current_time, t0, current_state)
                .method(solver_factory())
                .solve()?;

            current_state = *segment.y.last().unwrap();
            append_segment(&mut full_solution, segment);
        }

        if full_solution.t.last().is_some_and(|t| *t == t0) {
            *full_solution.y.last_mut().unwrap() = current_state;
        } else {
            full_solution.t.push(t0);
            full_solution.y.push(current_state);
        }

        Ok(full_solution)
    }

    fn observation_jump(&self, obs: &Observation) -> SVector<f64, 2> {
        let y = self.forward(obs.time);
        let weight = obs.weight.unwrap_or(DEFAULT_WEIGHT);
        let residual = y.plasma_rma - obs.plasma_rma;

        SVector::<f64, 2>::new(0.0, weight * residual)
    }
}

#[inline]
fn apply_jump(state: &mut AdjointState, jump: SVector<f64, 2>) {
    state[0] += jump[0];
    state[1] += jump[1];
}

#[inline]
fn push_state(out: &mut Solution<f64, AdjointState>, t: f64, y: AdjointState) {
    out.t.push(t);
    out.y.push(y);
}

fn append_segment(out: &mut Solution<f64, AdjointState>, segment: Solution<f64, AdjointState>) {
    for (i, (t, y)) in segment.t.into_iter().zip(segment.y.into_iter()).enumerate() {
        let duplicate_boundary = i == 0 && out.t.last().is_some_and(|last| *last == t);
        if !duplicate_boundary {
            out.t.push(t);
            out.y.push(y);
        }
    }
}

impl ODE<f64, AdjointState> for AdjointModel {
    fn diff(&self, t: f64, adjoint_state: &AdjointState, dydt: &mut AdjointState) {
        // interpolate the forward solution at time t
        // Although I think there is another way to do this
        // But either way, we need to get the forward solution at
        // time t
        // Forward can be interpolated, or we use the checkpointing and then run the forward integration to time t.
        let y = self.forward(t);
        let lambda = SVector::<f64, 2>::new(adjoint_state[0], adjoint_state[1]);

        // d/dt RMAb = prod - bbb_transport * RMAb
        // d/dt RMAp = bbb_transport * RMAb - deg * RMAp
        //
        // J_y = [ -bbb_transport, 0    ]
        //       [  bbb_transport, -deg]
        // J_y^T =
        //       [ -bbb_transport,  bbb_transport]
        //       [  0,             -deg]
        let jac_y_t = Matrix2::new(
            -self.parameters[1],
            self.parameters[1],
            0.0,
            -self.parameters[2],
        );

        // Parameter Jacobian
        // [prod, bbb_transport, deg]
        // dRMAb/dprod = 1
        // dRMAb/dbbb_transport = -RMAb
        // dRMAb/ddeg = 0
        //
        // dRMAp/dprod = 0
        // dRMAp/bbb_transport = RMAb
        // dRMAp/ddeg = -RMAp
        //
        // J_p = 2x3 so J_p^T = 3x2
        // J_p = [1, -RMAb,  0   ]
        //       [0,  RMAb, -RMAp]
        // so J_p^T = [1,     0   ]
        //            [-RMAb, RMAb]
        //            [0,    -RMAp]
        //
        let jac_p_t = Matrix3x2::new(1.0, 0.0, -y.brain_rma, y.brain_rma, 0.0, -y.plasma_rma);

        let d_lambda_dt = -jac_y_t * lambda;
        let d_mu_dt = -jac_p_t * lambda;

        dydt[0] = d_lambda_dt[0];
        dydt[1] = d_lambda_dt[1];
        dydt[2] = d_mu_dt[0];
        dydt[3] = d_mu_dt[1];
        dydt[4] = d_mu_dt[2];
    }
}

#[cfg(test)]
mod tests {
    use crate::solve::{SolutionAccess, Solve};

    use super::*;
    use differential_equations::methods::ExplicitRungeKutta;

    const T0: f64 = 0.;
    const TF: f64 = 504.;
    const DT: f64 = 1.;

    #[test]
    fn default_simulation() {
        let model = Model::default();
        let solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(T0, TF, DT, State::default(), solver);

        assert!(solution.is_ok());
        let unwrapped_solution = solution.unwrap();
        assert!(unwrapped_solution.plasma_rma().is_ok());
        assert!(unwrapped_solution.plasma_dox().is_err());
        assert!(unwrapped_solution.max_plasma_rma().is_ok());
        assert!(unwrapped_solution.max_tta().is_err());
    }

    #[test]
    fn custom_rates() {
        let model = Model::new(0.5, 0.7, 0.005);
        let solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(T0, TF, DT, State::default(), solver);

        assert!(solution.is_ok());
    }

    #[test]
    fn builder_pattern() -> Result<(), Box<dyn std::error::Error>> {
        let model = Model::builder().prod(0.5).bbb_transport(0.7).build()?;
        let solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(T0, TF, DT, State::default(), solver);

        assert!(solution.is_ok());
        Ok(())
    }

    #[test]
    fn adjoint_solve() -> Result<(), Box<dyn std::error::Error>> {
        fn interpolate_solution(solution: &Solution<f64, State<f64>>, t: f64) -> State<f64> {
            let times = &solution.t;
            let states = &solution.y;

            if t <= times[0] {
                return states[0];
            }

            if t >= *times.last().unwrap() {
                return *states.last().unwrap();
            }

            let upper = times.partition_point(|ti| *ti < t);
            let lower = upper - 1;
            let s = (t - times[lower]) / (times[upper] - times[lower]);

            states[lower] * (1.0 - s) + states[upper] * s
        }

        let model = Model::default();
        let params = model.get_parameters();
        let forward_solution = model.solve(
            T0,
            TF,
            DT,
            State::zeros(),
            ExplicitRungeKutta::dopri5(),
        )?;

        let adjoint_model = AdjointModel::new(params, forward_solution);
        let init_adjoint_state = AdjointState::zeros();

        let observations_for_fd = [
            Observation {
                time: T0,
                plasma_rma: 0.0,
                weight: None,
            },
            Observation {
                time: TF,
                plasma_rma: 20.0,
                weight: None,
            },
        ];

        let mut observations_for_adjoint = [
            Observation {
                time: T0,
                plasma_rma: 0.0,
                weight: None,
            },
            Observation {
                time: TF,
                plasma_rma: 20.0,
                weight: None,
            },
        ];

        let adjoint_solution = adjoint_model.solve(
            TF,
            T0,
            init_adjoint_state,
            &mut observations_for_adjoint,
            || ExplicitRungeKutta::dopri5(),
        )?;

        let final_adjoint = adjoint_solution.y.last().unwrap();
        let adjoint_gradient = SVector::<f64, 3>::new(
            final_adjoint[2],
            final_adjoint[3],
            final_adjoint[4],
        );

        let loss = |params: SVector<f64, 3>| -> Result<f64, Box<dyn std::error::Error>> {
            let perturbed_model = Model::new(params[0], params[1], params[2]);
            let solution = perturbed_model.solve(
                T0,
                TF,
                DT,
                State::zeros(),
                ExplicitRungeKutta::dopri5(),
            )?;

            let loss = observations_for_fd
                .iter()
                .map(|obs| {
                    let y = interpolate_solution(&solution, obs.time);
                    let weight = obs.weight.unwrap_or(DEFAULT_WEIGHT);
                    let residual = y.plasma_rma - obs.plasma_rma;
                    0.5 * weight * residual * residual
                })
                .sum();

            Ok(loss)
        };

        let mut fd_gradient = SVector::<f64, 3>::zeros();
        for k in 0..3 {
            let step = 1e-6 * params[k].abs().max(1.0);
            let mut p_plus = params;
            let mut p_minus = params;
            p_plus[k] += step;
            p_minus[k] -= step;

            fd_gradient[k] = (loss(p_plus)? - loss(p_minus)?) / (2.0 * step);
        }

        let abs_tol = 1e-4;
        let rel_tol = 1e-3;
        for k in 0..3 {
            let err = (adjoint_gradient[k] - fd_gradient[k]).abs();
            let scale = fd_gradient[k].abs().max(1.0);
            assert!(
                err <= abs_tol + rel_tol * scale,
                "gradient mismatch at parameter {k}: adjoint={}, finite_diff={}, err={}",
                adjoint_gradient[k],
                fd_gradient[k],
                err,
            );
        }

        Ok(())
    }

    #[cfg(any(feature = "polars-native", feature = "polars-wasm"))]
    #[test]
    fn dataframe_conversion() -> Result<(), PolarsError> {
        let model = Model::default();
        let solver = ExplicitRungeKutta::dopri5();
        let solution = model.solve(T0, TF, DT, State::default(), solver);

        assert!(solution.is_ok());
        let unwrapped_solution = solution.unwrap();
        let dataframe = unwrapped_solution.to_dataframe()?;

        assert_eq!(dataframe.shape(), (505, 3));
        assert_eq!(
            dataframe.get_column_names(),
            &["time", "brain_rma", "plasma_rma"]
        );
        Ok(())
    }
}
