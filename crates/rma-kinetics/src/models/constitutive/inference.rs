//! Inference helpers for the constitutive model.
//!
//! Public inference helpers use log-rate parameters in the order:
//! `[log_prod, log_bbb_transport, log_deg]`.

use differential_equations::{
    error::Error as OdeError, methods::ExplicitRungeKutta, prelude::Solution,
};
use nalgebra::SVector;
use thiserror::Error;

#[cfg(feature = "py")]
use std::sync::Mutex;

#[cfg(feature = "py")]
use numpy::{AllowTypeChange, PyArray1, PyArrayLike1};

#[cfg(feature = "py")]
use pyo3::{Bound, PyResult, Python, exceptions::PyValueError, pyclass, pymethods};

use crate::{
    inference::Cotangent,
    models::constitutive::{AdjointModel, AdjointState, Model, State},
    solve::Solve,
};

#[cfg(feature = "py")]
use crate::models::constitutive::PyState;

/// Forward solve result that can be reused for prediction and VJP computation.
#[derive(Clone)]
pub struct ConstitutiveForwardResult {
    pub log_params: [f64; 3],
    pub raw_params: SVector<f64, 3>,
    pub predictions: Vec<f64>,
    pub solution: Solution<f64, State<f64>>,
}

/// Errors returned by constitutive inference helpers.
#[derive(Debug, Error)]
pub enum InferenceError {
    #[error("log_params must contain finite values")]
    NonFiniteLogParams,
    #[error("initial state must contain finite values")]
    NonFiniteInitialState,
    #[error("observation times must contain finite values")]
    NonFiniteObsTimes,
    #[error("cotangents must contain finite values")]
    NonFiniteCotangents,
    #[error("obs_times and cotangent must have the same length")]
    LengthMismatch,
    #[error("dt must be positive and finite")]
    InvalidDt,
    #[error("tf must be greater than or equal to t0 and both must be finite")]
    InvalidTimeBounds,
    #[error("observation time out of bounds")]
    ObservationTimeOutOfBounds,
    #[error("n_mice must be greater than zero")]
    InvalidMouseCount,
    #[error("log_prod_mouse length does not match n_mice")]
    LogProdMouseLengthMismatch,
    #[error("mouse_id and obs_times must have the same length")]
    MouseIdLengthMismatch,
    #[error("mouse_id out of bounds")]
    MouseIdOutOfBounds,
    #[error("forward solve failed: {0:?}")]
    ForwardSolve(#[from] OdeError<f64, State<f64>>),
    #[error("adjoint solve failed: {0:?}")]
    AdjointSolve(#[from] OdeError<f64, AdjointState>),
}

/// Solve the constitutive model with log-rate parameters and return plasma RMA
/// predictions at `obs_times`, preserving the input order and duplicates.
pub fn predict(
    log_params: [f64; 3],
    init_state: State<f64>,
    obs_times: &[f64],
    t0: f64,
    tf: f64,
    dt: f64,
) -> Result<Vec<f64>, InferenceError> {
    Ok(solve_forward(log_params, init_state, obs_times, t0, tf, dt)?.predictions)
}

/// Solve the constitutive model once, then reuse that forward solution for an
/// adjoint vector-Jacobian product.
///
/// Returns plasma RMA predictions in the same order as `obs_times` and the VJP
/// gradient with respect to log-rate parameters.
pub fn predict_and_vjp(
    log_params: [f64; 3],
    init_state: State<f64>,
    obs_times: &[f64],
    cotangent: &[f64],
    t0: f64,
    tf: f64,
    dt: f64,
) -> Result<(Vec<f64>, [f64; 3]), InferenceError> {
    if obs_times.len() != cotangent.len() {
        return Err(InferenceError::LengthMismatch);
    }
    validate_cotangents(cotangent)?;

    let forward = solve_forward(log_params, init_state, obs_times, t0, tf, dt)?;
    let predictions = forward.predictions.clone();
    let gradient = vjp_from_forward(forward, obs_times, cotangent, t0, tf)?;

    Ok((predictions, gradient))
}

/// Forward solve result for population inference.
#[derive(Clone)]
pub struct PopulationForwardResult {
    pub log_prod_mouse: Vec<f64>,
    pub log_bbb: f64,
    pub log_deg: f64,
    pub predictions: Vec<f64>,
    pub per_mouse_forward: Vec<Option<ConstitutiveForwardResult>>,
}

/// Solve one constitutive trajectory per mouse and return plasma RMA predictions
/// in the original observation order.
pub fn population_predict(
    log_prod_mouse: &[f64],
    log_bbb: f64,
    log_deg: f64,
    init_state: State<f64>,
    mouse_id: &[usize],
    obs_times: &[f64],
    n_mice: usize,
    t0: f64,
    tf: f64,
    dt: f64,
) -> Result<Vec<f64>, InferenceError> {
    Ok(solve_population_forward(
        log_prod_mouse,
        log_bbb,
        log_deg,
        init_state,
        mouse_id,
        obs_times,
        n_mice,
        t0,
        tf,
        dt,
    )?
    .predictions)
}

/// Solve one constitutive trajectory per mouse, then reuse those forward
/// solutions for an adjoint vector-Jacobian product.
pub fn population_predict_and_vjp(
    log_prod_mouse: &[f64],
    log_bbb: f64,
    log_deg: f64,
    init_state: State<f64>,
    mouse_id: &[usize],
    obs_times: &[f64],
    cotangent: &[f64],
    n_mice: usize,
    t0: f64,
    tf: f64,
    dt: f64,
) -> Result<(Vec<f64>, Vec<f64>, f64, f64), InferenceError> {
    if obs_times.len() != cotangent.len() {
        return Err(InferenceError::LengthMismatch);
    }
    validate_cotangents(cotangent)?;

    let forward = solve_population_forward(
        log_prod_mouse,
        log_bbb,
        log_deg,
        init_state,
        mouse_id,
        obs_times,
        n_mice,
        t0,
        tf,
        dt,
    )?;
    let predictions = forward.predictions.clone();
    let (grad_prod, grad_bbb, grad_deg) =
        population_vjp_from_forward(forward, mouse_id, obs_times, cotangent, n_mice, t0, tf)?;

    Ok((predictions, grad_prod, grad_bbb, grad_deg))
}

/// Solve the forward model with log-rate parameters.
///
/// Predictions preserve `obs_times` order and duplicates.
pub fn solve_forward(
    log_params: [f64; 3],
    init_state: State<f64>,
    obs_times: &[f64],
    t0: f64,
    tf: f64,
    dt: f64,
) -> Result<ConstitutiveForwardResult, InferenceError> {
    validate_inputs(log_params, init_state, obs_times, t0, tf, dt)?;

    let raw_params = SVector::<f64, 3>::new(
        log_params[0].exp(),
        log_params[1].exp(),
        log_params[2].exp(),
    );

    if !raw_params.iter().all(|v| v.is_finite()) {
        return Err(InferenceError::NonFiniteLogParams);
    }

    let model = Model::new(raw_params[0], raw_params[1], raw_params[2]);
    let solution = model.solve(t0, tf, dt, init_state, ExplicitRungeKutta::dopri5())?;
    let predictions = obs_times
        .iter()
        .map(|&time| interpolate_plasma_rma(&solution, time))
        .collect();

    Ok(ConstitutiveForwardResult {
        log_params,
        raw_params,
        predictions,
        solution,
    })
}

pub fn solve_population_forward(
    log_prod_mouse: &[f64],
    log_bbb: f64,
    log_deg: f64,
    init_state: State<f64>,
    mouse_id: &[usize],
    obs_times: &[f64],
    n_mice: usize,
    t0: f64,
    tf: f64,
    dt: f64,
) -> Result<PopulationForwardResult, InferenceError> {
    validate_population_inputs(
        log_prod_mouse,
        log_bbb,
        log_deg,
        init_state,
        mouse_id,
        obs_times,
        n_mice,
        t0,
        tf,
        dt,
    )?;

    let obs_by_mouse = observations_by_mouse(mouse_id, n_mice)?;
    let mut predictions = vec![0.0; obs_times.len()];
    let mut per_mouse_forward = vec![None; n_mice];

    for (mouse, obs_indices) in obs_by_mouse.iter().enumerate() {
        if obs_indices.is_empty() {
            continue;
        }

        let mouse_obs_times = obs_indices
            .iter()
            .map(|&idx| obs_times[idx])
            .collect::<Vec<_>>();
        let log_params = [log_prod_mouse[mouse], log_bbb, log_deg];
        let forward = solve_forward(log_params, init_state, &mouse_obs_times, t0, tf, dt)?;

        for (&obs_idx, &prediction) in obs_indices.iter().zip(forward.predictions.iter()) {
            predictions[obs_idx] = prediction;
        }
        per_mouse_forward[mouse] = Some(forward);
    }

    Ok(PopulationForwardResult {
        log_prod_mouse: log_prod_mouse.to_vec(),
        log_bbb,
        log_deg,
        predictions,
        per_mouse_forward,
    })
}

fn population_vjp_from_forward(
    forward: PopulationForwardResult,
    mouse_id: &[usize],
    obs_times: &[f64],
    cotangent: &[f64],
    n_mice: usize,
    t0: f64,
    tf: f64,
) -> Result<(Vec<f64>, f64, f64), InferenceError> {
    if obs_times.len() != cotangent.len() {
        return Err(InferenceError::LengthMismatch);
    }
    validate_cotangents(cotangent)?;
    let obs_by_mouse = observations_by_mouse(mouse_id, n_mice)?;

    let mut grad_log_prod_mouse = vec![0.0; n_mice];
    let mut grad_log_bbb = 0.0;
    let mut grad_log_deg = 0.0;

    for (mouse, obs_indices) in obs_by_mouse.iter().enumerate() {
        let Some(mouse_forward) = forward.per_mouse_forward[mouse].clone() else {
            continue;
        };

        let mouse_obs_times = obs_indices
            .iter()
            .map(|&idx| obs_times[idx])
            .collect::<Vec<_>>();
        let mouse_cotangent = obs_indices
            .iter()
            .map(|&idx| cotangent[idx])
            .collect::<Vec<_>>();
        let gradient = vjp_from_forward(mouse_forward, &mouse_obs_times, &mouse_cotangent, t0, tf)?;
        grad_log_prod_mouse[mouse] += gradient[0];
        grad_log_bbb += gradient[1];
        grad_log_deg += gradient[2];
    }

    Ok((grad_log_prod_mouse, grad_log_bbb, grad_log_deg))
}

fn vjp_from_forward(
    forward: ConstitutiveForwardResult,
    obs_times: &[f64],
    cotangent: &[f64],
    t0: f64,
    tf: f64,
) -> Result<[f64; 3], InferenceError> {
    if obs_times.len() != cotangent.len() {
        return Err(InferenceError::LengthMismatch);
    }
    validate_cotangents(cotangent)?;

    let mut cotangents = obs_times
        .iter()
        .zip(cotangent.iter())
        .map(|(&time, &value)| Cotangent { time, value })
        .collect::<Vec<_>>();

    let raw_params = forward.raw_params;
    let adjoint_model = AdjointModel::new(raw_params, forward.solution);
    let grad_raw =
        adjoint_model.solve_vjp(tf, t0, AdjointState::zeros(), &mut cotangents, || {
            ExplicitRungeKutta::dopri5()
        })?;

    let grad_log = grad_raw.component_mul(&raw_params);
    Ok([grad_log[0], grad_log[1], grad_log[2]])
}

fn validate_inputs(
    log_params: [f64; 3],
    init_state: State<f64>,
    obs_times: &[f64],
    t0: f64,
    tf: f64,
    dt: f64,
) -> Result<(), InferenceError> {
    if !log_params.iter().all(|v| v.is_finite()) {
        return Err(InferenceError::NonFiniteLogParams);
    }
    if !init_state.brain_rma.is_finite() || !init_state.plasma_rma.is_finite() {
        return Err(InferenceError::NonFiniteInitialState);
    }
    if !obs_times.iter().all(|v| v.is_finite()) {
        return Err(InferenceError::NonFiniteObsTimes);
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(InferenceError::InvalidDt);
    }
    if !t0.is_finite() || !tf.is_finite() || tf < t0 {
        return Err(InferenceError::InvalidTimeBounds);
    }
    if obs_times.iter().any(|&time| time < t0 || time > tf) {
        return Err(InferenceError::ObservationTimeOutOfBounds);
    }

    Ok(())
}

fn validate_population_inputs(
    log_prod_mouse: &[f64],
    log_bbb: f64,
    log_deg: f64,
    init_state: State<f64>,
    mouse_id: &[usize],
    obs_times: &[f64],
    n_mice: usize,
    t0: f64,
    tf: f64,
    dt: f64,
) -> Result<(), InferenceError> {
    if n_mice == 0 {
        return Err(InferenceError::InvalidMouseCount);
    }
    if log_prod_mouse.len() != n_mice {
        return Err(InferenceError::LogProdMouseLengthMismatch);
    }
    if mouse_id.len() != obs_times.len() {
        return Err(InferenceError::MouseIdLengthMismatch);
    }
    if mouse_id.iter().any(|&id| id >= n_mice) {
        return Err(InferenceError::MouseIdOutOfBounds);
    }
    if !log_prod_mouse.iter().all(|v| v.is_finite()) || !log_bbb.is_finite() || !log_deg.is_finite()
    {
        return Err(InferenceError::NonFiniteLogParams);
    }
    validate_inputs(
        [log_prod_mouse[0], log_bbb, log_deg],
        init_state,
        obs_times,
        t0,
        tf,
        dt,
    )
}

fn observations_by_mouse(
    mouse_id: &[usize],
    n_mice: usize,
) -> Result<Vec<Vec<usize>>, InferenceError> {
    let mut obs_by_mouse = vec![Vec::new(); n_mice];
    for (obs_idx, &mouse) in mouse_id.iter().enumerate() {
        if mouse >= n_mice {
            return Err(InferenceError::MouseIdOutOfBounds);
        }
        obs_by_mouse[mouse].push(obs_idx);
    }
    Ok(obs_by_mouse)
}

fn validate_cotangents(cotangent: &[f64]) -> Result<(), InferenceError> {
    if !cotangent.iter().all(|v| v.is_finite()) {
        return Err(InferenceError::NonFiniteCotangents);
    }
    Ok(())
}

#[cfg(feature = "py")]
#[derive(Clone)]
struct CachedForward {
    log_params: [f64; 3],
    forward: ConstitutiveForwardResult,
}

#[cfg(feature = "py")]
#[derive(Clone)]
struct PopulationCachedForward {
    log_prod_mouse: Vec<f64>,
    log_bbb: f64,
    log_deg: f64,
    forward: PopulationForwardResult,
}

#[cfg(feature = "py")]
#[pyclass(name = "InferenceSolver")]
pub struct PyInferenceSolver {
    obs_time: Vec<f64>,
    init_state: State<f64>,
    t0: f64,
    tf: f64,
    dt: f64,
    cached_forward: Mutex<Option<CachedForward>>,
}

#[cfg(feature = "py")]
#[pymethods]
impl PyInferenceSolver {
    #[new]
    #[pyo3(signature = (obs_time, *, init_state=None, t0=0.0, tf=None, dt=0.25))]
    fn new(
        obs_time: PyArrayLike1<'_, f64, AllowTypeChange>,
        init_state: Option<PyState>,
        t0: f64,
        tf: Option<f64>,
        dt: f64,
    ) -> PyResult<Self> {
        let obs_time = obs_time.as_array().iter().copied().collect::<Vec<f64>>();
        let tf = match tf {
            Some(value) => value,
            None => obs_time.iter().copied().reduce(f64::max).ok_or_else(|| {
                PyValueError::new_err("obs_time must not be empty when tf is None")
            })?,
        };
        let init_state = init_state
            .map(|state| state.inner)
            .unwrap_or_else(State::zeros);

        validate_inputs([0.0, 0.0, 0.0], init_state, &obs_time, t0, tf, dt)
            .map_err(py_inference_error)?;

        Ok(Self {
            obs_time,
            init_state,
            t0,
            tf,
            dt,
            cached_forward: Mutex::new(None),
        })
    }

    #[getter]
    fn get_n_obs(&self) -> usize {
        self.obs_time.len()
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        log_params: PyArrayLike1<'_, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let log_params = py_log_params(log_params)?;
        let obs_time = self.obs_time.clone();
        let init_state = self.init_state;
        let t0 = self.t0;
        let tf = self.tf;
        let dt = self.dt;

        let forward = py
            .detach(move || solve_forward(log_params, init_state, &obs_time, t0, tf, dt))
            .map_err(py_inference_error)?;
        let predictions = forward.predictions.clone();

        {
            let mut cache = self
                .cached_forward
                .lock()
                .map_err(|_| PyValueError::new_err("inference solver cache lock poisoned"))?;
            *cache = Some(CachedForward {
                log_params,
                forward,
            });
        }

        Ok(PyArray1::from_vec(py, predictions))
    }

    fn predict_and_vjp<'py>(
        &self,
        py: Python<'py>,
        log_params: PyArrayLike1<'_, f64, AllowTypeChange>,
        cotangent: PyArrayLike1<'_, f64, AllowTypeChange>,
    ) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
        let log_params = py_log_params(log_params)?;
        let cotangent = cotangent.as_array().iter().copied().collect::<Vec<f64>>();
        if cotangent.len() != self.obs_time.len() {
            return Err(PyValueError::new_err(format!(
                "cotangent length {} does not match n_obs {}",
                cotangent.len(),
                self.obs_time.len()
            )));
        }
        validate_cotangents(&cotangent).map_err(py_inference_error)?;

        let cached_forward = {
            let cache = self
                .cached_forward
                .lock()
                .map_err(|_| PyValueError::new_err("inference solver cache lock poisoned"))?;
            cache
                .as_ref()
                .filter(|cached| same_log_params(cached.log_params, log_params))
                .map(|cached| cached.forward.clone())
        };

        let obs_time = self.obs_time.clone();
        let init_state = self.init_state;
        let t0 = self.t0;
        let tf = self.tf;
        let dt = self.dt;

        let (forward, gradient) = py
            .detach(move || {
                let forward = match cached_forward {
                    Some(forward) => forward,
                    None => solve_forward(log_params, init_state, &obs_time, t0, tf, dt)?,
                };
                let predictions_forward = forward.clone();
                let gradient = vjp_from_forward(forward, &obs_time, &cotangent, t0, tf)?;
                Ok::<_, InferenceError>((predictions_forward, gradient))
            })
            .map_err(py_inference_error)?;

        {
            let mut cache = self
                .cached_forward
                .lock()
                .map_err(|_| PyValueError::new_err("inference solver cache lock poisoned"))?;
            *cache = Some(CachedForward {
                log_params,
                forward: forward.clone(),
            });
        }

        Ok((
            PyArray1::from_vec(py, forward.predictions),
            PyArray1::from_vec(py, gradient.to_vec()),
        ))
    }

    fn clear_cache(&self) -> PyResult<()> {
        let mut cache = self
            .cached_forward
            .lock()
            .map_err(|_| PyValueError::new_err("inference solver cache lock poisoned"))?;
        *cache = None;
        Ok(())
    }
}

#[cfg(feature = "py")]
#[pyclass(name = "PopulationInferenceSolver")]
pub struct PyPopulationInferenceSolver {
    mouse_id: Vec<usize>,
    obs_time: Vec<f64>,
    n_mice: usize,
    init_state: State<f64>,
    t0: f64,
    tf: f64,
    dt: f64,
    cached_forward: Mutex<Option<PopulationCachedForward>>,
}

#[cfg(feature = "py")]
#[pymethods]
impl PyPopulationInferenceSolver {
    #[new]
    #[pyo3(signature = (mouse_id, obs_time, n_mice, *, init_state=None, t0=0.0, tf=None, dt=0.25))]
    fn new(
        mouse_id: PyArrayLike1<'_, i64>,
        obs_time: PyArrayLike1<'_, f64, AllowTypeChange>,
        n_mice: usize,
        init_state: Option<PyState>,
        t0: f64,
        tf: Option<f64>,
        dt: f64,
    ) -> PyResult<Self> {
        let raw_mouse_id = mouse_id.as_array().iter().copied().collect::<Vec<i64>>();
        let mouse_id = py_mouse_id(raw_mouse_id)?;
        let obs_time = obs_time.as_array().iter().copied().collect::<Vec<f64>>();
        let tf = match tf {
            Some(value) => value,
            None => obs_time.iter().copied().reduce(f64::max).ok_or_else(|| {
                PyValueError::new_err("obs_time must not be empty when tf is None")
            })?,
        };
        let init_state = init_state
            .map(|state| state.inner)
            .unwrap_or_else(State::zeros);

        validate_population_inputs(
            &vec![0.0; n_mice],
            0.0,
            0.0,
            init_state,
            &mouse_id,
            &obs_time,
            n_mice,
            t0,
            tf,
            dt,
        )
        .map_err(py_inference_error)?;

        Ok(Self {
            mouse_id,
            obs_time,
            n_mice,
            init_state,
            t0,
            tf,
            dt,
            cached_forward: Mutex::new(None),
        })
    }

    #[getter]
    fn get_n_obs(&self) -> usize {
        self.obs_time.len()
    }

    #[getter]
    fn get_n_mice(&self) -> usize {
        self.n_mice
    }

    fn predict<'py>(
        &self,
        py: Python<'py>,
        log_prod_mouse: PyArrayLike1<'_, f64, AllowTypeChange>,
        log_bbb: f64,
        log_deg: f64,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let log_prod_mouse = py_log_prod_mouse(log_prod_mouse, self.n_mice)?;
        if !log_bbb.is_finite() || !log_deg.is_finite() {
            return Err(py_inference_error(InferenceError::NonFiniteLogParams));
        }

        let mouse_id = self.mouse_id.clone();
        let obs_time = self.obs_time.clone();
        let n_mice = self.n_mice;
        let init_state = self.init_state;
        let t0 = self.t0;
        let tf = self.tf;
        let dt = self.dt;

        let forward = py
            .detach(move || {
                solve_population_forward(
                    &log_prod_mouse,
                    log_bbb,
                    log_deg,
                    init_state,
                    &mouse_id,
                    &obs_time,
                    n_mice,
                    t0,
                    tf,
                    dt,
                )
            })
            .map_err(py_inference_error)?;
        let predictions = forward.predictions.clone();

        {
            let mut cache = self
                .cached_forward
                .lock()
                .map_err(|_| PyValueError::new_err("inference solver cache lock poisoned"))?;
            *cache = Some(PopulationCachedForward {
                log_prod_mouse: forward.log_prod_mouse.clone(),
                log_bbb,
                log_deg,
                forward,
            });
        }

        Ok(PyArray1::from_vec(py, predictions))
    }

    fn predict_and_vjp<'py>(
        &self,
        py: Python<'py>,
        log_prod_mouse: PyArrayLike1<'_, f64, AllowTypeChange>,
        log_bbb: f64,
        log_deg: f64,
        cotangent: PyArrayLike1<'_, f64, AllowTypeChange>,
    ) -> PyResult<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        f64,
        f64,
    )> {
        let log_prod_mouse = py_log_prod_mouse(log_prod_mouse, self.n_mice)?;
        if !log_bbb.is_finite() || !log_deg.is_finite() {
            return Err(py_inference_error(InferenceError::NonFiniteLogParams));
        }
        let cotangent = cotangent.as_array().iter().copied().collect::<Vec<f64>>();
        if cotangent.len() != self.obs_time.len() {
            return Err(PyValueError::new_err(format!(
                "cotangent length {} does not match n_obs {}",
                cotangent.len(),
                self.obs_time.len()
            )));
        }
        validate_cotangents(&cotangent).map_err(py_inference_error)?;

        let cached_forward = {
            let cache = self
                .cached_forward
                .lock()
                .map_err(|_| PyValueError::new_err("inference solver cache lock poisoned"))?;
            cache
                .as_ref()
                .filter(|cached| {
                    same_population_log_params(
                        &cached.log_prod_mouse,
                        cached.log_bbb,
                        cached.log_deg,
                        &log_prod_mouse,
                        log_bbb,
                        log_deg,
                    )
                })
                .map(|cached| cached.forward.clone())
        };

        let mouse_id = self.mouse_id.clone();
        let obs_time = self.obs_time.clone();
        let n_mice = self.n_mice;
        let init_state = self.init_state;
        let t0 = self.t0;
        let tf = self.tf;
        let dt = self.dt;

        let (forward, grad_prod, grad_bbb, grad_deg) = py
            .detach(move || {
                let forward = match cached_forward {
                    Some(forward) => forward,
                    None => solve_population_forward(
                        &log_prod_mouse,
                        log_bbb,
                        log_deg,
                        init_state,
                        &mouse_id,
                        &obs_time,
                        n_mice,
                        t0,
                        tf,
                        dt,
                    )?,
                };
                let predictions_forward = forward.clone();
                let (grad_prod, grad_bbb, grad_deg) = population_vjp_from_forward(
                    forward, &mouse_id, &obs_time, &cotangent, n_mice, t0, tf,
                )?;
                Ok::<_, InferenceError>((predictions_forward, grad_prod, grad_bbb, grad_deg))
            })
            .map_err(py_inference_error)?;

        {
            let mut cache = self
                .cached_forward
                .lock()
                .map_err(|_| PyValueError::new_err("inference solver cache lock poisoned"))?;
            *cache = Some(PopulationCachedForward {
                log_prod_mouse: forward.log_prod_mouse.clone(),
                log_bbb,
                log_deg,
                forward: forward.clone(),
            });
        }

        Ok((
            PyArray1::from_vec(py, forward.predictions),
            PyArray1::from_vec(py, grad_prod),
            grad_bbb,
            grad_deg,
        ))
    }

    fn clear_cache(&self) -> PyResult<()> {
        let mut cache = self
            .cached_forward
            .lock()
            .map_err(|_| PyValueError::new_err("inference solver cache lock poisoned"))?;
        *cache = None;
        Ok(())
    }
}

#[cfg(feature = "py")]
fn same_log_params(a: [f64; 3], b: [f64; 3]) -> bool {
    a.iter()
        .zip(b.iter())
        .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[cfg(feature = "py")]
fn same_population_log_params(
    a_prod: &[f64],
    a_bbb: f64,
    a_deg: f64,
    b_prod: &[f64],
    b_bbb: f64,
    b_deg: f64,
) -> bool {
    a_bbb.to_bits() == b_bbb.to_bits()
        && a_deg.to_bits() == b_deg.to_bits()
        && a_prod.len() == b_prod.len()
        && a_prod
            .iter()
            .zip(b_prod.iter())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

#[cfg(feature = "py")]
fn py_mouse_id(raw_mouse_id: Vec<i64>) -> PyResult<Vec<usize>> {
    raw_mouse_id
        .into_iter()
        .map(|id| {
            usize::try_from(id).map_err(|_| PyValueError::new_err("mouse_id must be nonnegative"))
        })
        .collect()
}

#[cfg(feature = "py")]
fn py_log_prod_mouse(
    log_prod_mouse: PyArrayLike1<'_, f64, AllowTypeChange>,
    n_mice: usize,
) -> PyResult<Vec<f64>> {
    let log_prod_mouse = log_prod_mouse
        .as_array()
        .iter()
        .copied()
        .collect::<Vec<f64>>();
    if log_prod_mouse.len() != n_mice {
        return Err(PyValueError::new_err(format!(
            "log_prod_mouse length {} does not match n_mice {}",
            log_prod_mouse.len(),
            n_mice
        )));
    }
    if !log_prod_mouse.iter().all(|v| v.is_finite()) {
        return Err(py_inference_error(InferenceError::NonFiniteLogParams));
    }
    Ok(log_prod_mouse)
}

#[cfg(feature = "py")]
fn py_log_params(log_params: PyArrayLike1<'_, f64, AllowTypeChange>) -> PyResult<[f64; 3]> {
    let log_params = log_params.as_array().iter().copied().collect::<Vec<f64>>();
    if log_params.len() != 3 {
        return Err(PyValueError::new_err(format!(
            "log_params length {} does not match expected length 3",
            log_params.len()
        )));
    }
    Ok([log_params[0], log_params[1], log_params[2]])
}

#[cfg(feature = "py")]
fn py_inference_error(error: InferenceError) -> pyo3::PyErr {
    PyValueError::new_err(error.to_string())
}

fn interpolate_plasma_rma(solution: &Solution<f64, State<f64>>, time: f64) -> f64 {
    let times = &solution.t;
    let states = &solution.y;

    if time <= times[0] {
        return states[0].plasma_rma;
    }

    if time >= *times.last().unwrap() {
        return states.last().unwrap().plasma_rma;
    }

    let upper = times.partition_point(|ti| *ti < time);
    let lower = upper - 1;
    let s = (time - times[lower]) / (times[upper] - times[lower]);

    states[lower].plasma_rma * (1.0 - s) + states[upper].plasma_rma * s
}

#[cfg(test)]
mod tests {
    use super::*;

    const T0: f64 = 0.0;
    const TF: f64 = 24.0;
    const DT: f64 = 0.25;

    #[test]
    fn preserve_obs_order_and_duplicates() -> Result<(), InferenceError> {
        let log_params = [0.2_f64.ln(), 0.6_f64.ln(), 0.007_f64.ln()];
        let obs_times = [12.0, 1.0, 12.0, 6.0];

        let predictions = predict(log_params, State::zeros(), &obs_times, T0, TF, DT)?;

        assert_eq!(predictions.len(), obs_times.len());
        assert_eq!(predictions[0], predictions[2]);
        Ok(())
    }

    #[test]
    fn zero_cotangent_and_gradient() -> Result<(), InferenceError> {
        let log_params = [0.2_f64.ln(), 0.6_f64.ln(), 0.007_f64.ln()];
        let obs_times = [1.0, 6.0, 12.0, 24.0];
        let cotangent = [0.0; 4];

        let (_predictions, gradient) = predict_and_vjp(
            log_params,
            State::zeros(),
            &obs_times,
            &cotangent,
            T0,
            TF,
            DT,
        )?;

        for value in gradient {
            assert!(
                value.abs() < 1e-12,
                "expected zero gradient, got {gradient:?}"
            );
        }

        Ok(())
    }

    #[test]
    fn vjp_matches_finite_difference() -> Result<(), InferenceError> {
        let log_params = [0.2_f64.ln(), 0.6_f64.ln(), 0.007_f64.ln()];
        let obs_times = [1.0, 6.0, 12.0, 24.0];
        let cotangent = [0.25, -0.5, 0.75, 1.25];

        let (_predictions, gradient) = predict_and_vjp(
            log_params,
            State::zeros(),
            &obs_times,
            &cotangent,
            T0,
            TF,
            DT,
        )?;

        let scalar = |params: [f64; 3]| -> Result<f64, InferenceError> {
            let predictions = predict(params, State::zeros(), &obs_times, T0, TF, DT)?;
            Ok(predictions
                .iter()
                .zip(cotangent.iter())
                .map(|(prediction, cotangent)| prediction * cotangent)
                .sum())
        };

        for k in 0..3 {
            let step = 1e-6;
            let mut plus = log_params;
            let mut minus = log_params;
            plus[k] += step;
            minus[k] -= step;
            let fd = (scalar(plus)? - scalar(minus)?) / (2.0 * step);
            let err = (gradient[k] - fd).abs();
            let scale = fd.abs().max(1.0);
            assert!(
                err <= 1e-4 + 1e-3 * scale,
                "VJP mismatch at log parameter {k}: adjoint={}, finite_diff={fd}, err={err}",
                gradient[k],
            );
        }

        Ok(())
    }

    #[test]
    fn pop_preserves_obs_order_duplicates_and_empty_mice() -> Result<(), InferenceError> {
        let log_prod_mouse = [0.2_f64.ln(), 0.4_f64.ln(), 0.8_f64.ln()];
        let log_bbb = 0.6_f64.ln();
        let log_deg = 0.007_f64.ln();
        let mouse_id = [1, 0, 1, 0];
        let obs_times = [12.0, 1.0, 12.0, 6.0];

        let predictions = population_predict(
            &log_prod_mouse,
            log_bbb,
            log_deg,
            State::zeros(),
            &mouse_id,
            &obs_times,
            3,
            T0,
            TF,
            DT,
        )?;

        assert_eq!(predictions.len(), obs_times.len());
        assert_eq!(predictions[0], predictions[2]);
        Ok(())
    }

    #[test]
    fn pop_zero_cotangent_and_gradient() -> Result<(), InferenceError> {
        let log_prod_mouse = [0.2_f64.ln(), 0.4_f64.ln(), 0.8_f64.ln()];
        let mouse_id = [0, 0, 1, 1];
        let obs_times = [1.0, 6.0, 12.0, 24.0];
        let cotangent = [0.0; 4];

        let (_predictions, grad_prod, grad_bbb, grad_deg) = population_predict_and_vjp(
            &log_prod_mouse,
            0.6_f64.ln(),
            0.007_f64.ln(),
            State::zeros(),
            &mouse_id,
            &obs_times,
            &cotangent,
            3,
            T0,
            TF,
            DT,
        )?;

        for value in grad_prod {
            assert!(value.abs() < 1e-12);
        }
        assert!(grad_bbb.abs() < 1e-12);
        assert!(grad_deg.abs() < 1e-12);
        Ok(())
    }

    #[test]
    fn pop_vjp_matches_finite_difference() -> Result<(), InferenceError> {
        let log_prod_mouse = [0.2_f64.ln(), 0.4_f64.ln()];
        let log_bbb = 0.6_f64.ln();
        let log_deg = 0.007_f64.ln();
        let mouse_id = [0, 0, 1, 1, 0];
        let obs_times = [1.0, 6.0, 1.0, 12.0, 24.0];
        let cotangent = [0.25, -0.5, 0.75, 1.25, -0.4];

        let (_predictions, grad_prod, grad_bbb, grad_deg) = population_predict_and_vjp(
            &log_prod_mouse,
            log_bbb,
            log_deg,
            State::zeros(),
            &mouse_id,
            &obs_times,
            &cotangent,
            2,
            T0,
            TF,
            DT,
        )?;

        let scalar = |prod: &[f64], bbb: f64, deg: f64| -> Result<f64, InferenceError> {
            let predictions = population_predict(
                prod,
                bbb,
                deg,
                State::zeros(),
                &mouse_id,
                &obs_times,
                2,
                T0,
                TF,
                DT,
            )?;
            Ok(predictions
                .iter()
                .zip(cotangent.iter())
                .map(|(prediction, cotangent)| prediction * cotangent)
                .sum())
        };

        let step = 1e-6;
        for mouse in 0..2 {
            let mut plus = log_prod_mouse;
            let mut minus = log_prod_mouse;
            plus[mouse] += step;
            minus[mouse] -= step;
            let fd = (scalar(&plus, log_bbb, log_deg)? - scalar(&minus, log_bbb, log_deg)?)
                / (2.0 * step);
            let err = (grad_prod[mouse] - fd).abs();
            let scale = fd.abs().max(1.0);
            assert!(
                err <= 1e-4 + 1e-3 * scale,
                "population prod VJP mismatch at mouse {mouse}: adjoint={}, finite_diff={fd}, err={err}",
                grad_prod[mouse]
            );
        }

        for (name, adjoint, plus_args, minus_args) in [
            (
                "bbb",
                grad_bbb,
                (log_bbb + step, log_deg),
                (log_bbb - step, log_deg),
            ),
            (
                "deg",
                grad_deg,
                (log_bbb, log_deg + step),
                (log_bbb, log_deg - step),
            ),
        ] {
            let fd = (scalar(&log_prod_mouse, plus_args.0, plus_args.1)?
                - scalar(&log_prod_mouse, minus_args.0, minus_args.1)?)
                / (2.0 * step);
            let err = (adjoint - fd).abs();
            let scale = fd.abs().max(1.0);
            assert!(
                err <= 1e-4 + 1e-3 * scale,
                "population {name} VJP mismatch: adjoint={adjoint}, finite_diff={fd}, err={err}",
            );
        }

        Ok(())
    }

    #[test]
    fn pop_global_gradient_accumulates_over_mice() -> Result<(), InferenceError> {
        let log_prod_mouse = [0.2_f64.ln(), 0.4_f64.ln()];
        let log_bbb = 0.6_f64.ln();
        let log_deg = 0.007_f64.ln();
        let mouse_id = [0, 0, 1, 1];
        let obs_times = [1.0, 6.0, 1.0, 12.0];
        let cotangent = [0.25, -0.5, 0.75, 1.25];

        let (_predictions, _grad_prod, grad_bbb, grad_deg) = population_predict_and_vjp(
            &log_prod_mouse,
            log_bbb,
            log_deg,
            State::zeros(),
            &mouse_id,
            &obs_times,
            &cotangent,
            2,
            T0,
            TF,
            DT,
        )?;

        let (_p0, g0) = predict_and_vjp(
            [log_prod_mouse[0], log_bbb, log_deg],
            State::zeros(),
            &[1.0, 6.0],
            &[0.25, -0.5],
            T0,
            TF,
            DT,
        )?;
        let (_p1, g1) = predict_and_vjp(
            [log_prod_mouse[1], log_bbb, log_deg],
            State::zeros(),
            &[1.0, 12.0],
            &[0.75, 1.25],
            T0,
            TF,
            DT,
        )?;

        assert!((grad_bbb - (g0[1] + g1[1])).abs() < 1e-10);
        assert!((grad_deg - (g0[2] + g1[2])).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn pop_sum_duplicate_cotangent_times_per_mouse() -> Result<(), InferenceError> {
        let log_prod_mouse = [0.2_f64.ln(), 0.4_f64.ln()];
        let log_bbb = 0.6_f64.ln();
        let log_deg = 0.007_f64.ln();

        let duplicated_mouse_id = [0, 0, 1, 1];
        let duplicated_times = [1.0, 1.0, 6.0, 6.0];
        let duplicated_cotangent = [0.25, -0.75, 1.5, -0.5];
        let (_predictions, duplicated_prod, duplicated_bbb, duplicated_deg) =
            population_predict_and_vjp(
                &log_prod_mouse,
                log_bbb,
                log_deg,
                State::zeros(),
                &duplicated_mouse_id,
                &duplicated_times,
                &duplicated_cotangent,
                2,
                T0,
                TF,
                DT,
            )?;

        let summed_mouse_id = [0, 1];
        let summed_times = [1.0, 6.0];
        let summed_cotangent = [-0.5, 1.0];
        let (_predictions, summed_prod, summed_bbb, summed_deg) = population_predict_and_vjp(
            &log_prod_mouse,
            log_bbb,
            log_deg,
            State::zeros(),
            &summed_mouse_id,
            &summed_times,
            &summed_cotangent,
            2,
            T0,
            TF,
            DT,
        )?;

        for mouse in 0..2 {
            assert!((duplicated_prod[mouse] - summed_prod[mouse]).abs() < 1e-10);
        }
        assert!((duplicated_bbb - summed_bbb).abs() < 1e-10);
        assert!((duplicated_deg - summed_deg).abs() < 1e-10);
        Ok(())
    }

    #[test]
    fn sum_duplicate_cotangent_times() -> Result<(), InferenceError> {
        let log_params = [0.2_f64.ln(), 0.6_f64.ln(), 0.007_f64.ln()];

        let duplicated_times = [1.0, 1.0, 6.0, 12.0];
        let duplicated_cotangent = [0.25, -0.75, 1.5, -0.5];
        let (_predictions, duplicated_gradient) = predict_and_vjp(
            log_params,
            State::zeros(),
            &duplicated_times,
            &duplicated_cotangent,
            T0,
            TF,
            DT,
        )?;

        let summed_times = [1.0, 6.0, 12.0];
        let summed_cotangent = [-0.5, 1.5, -0.5];
        let (_predictions, summed_gradient) = predict_and_vjp(
            log_params,
            State::zeros(),
            &summed_times,
            &summed_cotangent,
            T0,
            TF,
            DT,
        )?;

        for k in 0..3 {
            let err = (duplicated_gradient[k] - summed_gradient[k]).abs();
            assert!(
                err < 1e-10,
                "duplicate-time VJP mismatch at parameter {k}: duplicated={}, summed={}, err={err}",
                duplicated_gradient[k],
                summed_gradient[k],
            );
        }

        Ok(())
    }
}
