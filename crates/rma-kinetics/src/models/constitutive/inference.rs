//! Inference helpers for the constitutive model.
//!
//! Public inference helpers use log-rate parameters in the order:
//! `[log_prod, log_bbb_transport, log_deg]`.

use differential_equations::{
    error::Error as OdeError, methods::ExplicitRungeKutta, prelude::Solution,
};
use nalgebra::SVector;
use thiserror::Error;

use crate::{
    inference::Cotangent,
    models::constitutive::{AdjointModel, AdjointState, Model, State},
    solve::Solve,
};

/// Forward solve result that can be reused for prediction and VJP computation.
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
    let mut cotangents = obs_times
        .iter()
        .zip(cotangent.iter())
        .map(|(&time, &value)| Cotangent { time, value })
        .collect::<Vec<_>>();

    let adjoint_model = AdjointModel::new(forward.raw_params, forward.solution);
    let grad_raw =
        adjoint_model.solve_vjp(tf, t0, AdjointState::zeros(), &mut cotangents, || {
            ExplicitRungeKutta::dopri5()
        })?;

    let grad_log = grad_raw.component_mul(&forward.raw_params);

    Ok((forward.predictions, [grad_log[0], grad_log[1], grad_log[2]]))
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

fn validate_cotangents(cotangent: &[f64]) -> Result<(), InferenceError> {
    if !cotangent.iter().all(|v| v.is_finite()) {
        return Err(InferenceError::NonFiniteCotangents);
    }
    Ok(())
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
    fn predict_preserves_observation_order_and_duplicates() -> Result<(), InferenceError> {
        let log_params = [0.2_f64.ln(), 0.6_f64.ln(), 0.007_f64.ln()];
        let obs_times = [12.0, 1.0, 12.0, 6.0];

        let predictions = predict(log_params, State::zeros(), &obs_times, T0, TF, DT)?;

        assert_eq!(predictions.len(), obs_times.len());
        assert_eq!(predictions[0], predictions[2]);
        Ok(())
    }

    #[test]
    fn zero_cotangent_returns_zero_gradient() -> Result<(), InferenceError> {
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
    fn vjp_matches_finite_difference_log_params() -> Result<(), InferenceError> {
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
    fn duplicate_cotangent_times_are_summed_for_vjp() -> Result<(), InferenceError> {
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
