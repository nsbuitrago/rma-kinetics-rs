//! Pharmacokinetic model utilities.
//!
//! - DoseApplyingSolout: a custom solout implementation for `differential_equations` to apply
//!   scheduled state updates and handle evenly-spaced output points.
//! - Error: an error type for pharmacokinetic model errors.

use differential_equations::{
    prelude::{ControlFlag, Interpolation, Solution},
    solout::Solout,
    traits::State as StateTrait,
};
use thiserror::Error as ErrorTrait;

/// PK Errors
#[derive(ErrorTrait, Debug)]
pub enum Error {
    #[error("Bioavailability must be between 0 and 1")]
    InvalidBioavailability(f64),
    #[error("Dose times must be unique; found duplicate times at {0} and {1}")]
    DuplicateDoseTimes(f64, f64),
}

/// Check if two f64 values are close.
/// Here, arbitrarily used 1e-10 for some small precision tolerance.
macro_rules! is_close {
    ($a:expr, $b:expr) => {
        ($a - $b).abs() < 1e-10
    };
}

/// Trait for dose schedules used by [`validate_unique_dose_times`].
pub trait ScheduledDose {
    fn time(&self) -> f64;
    fn amount(&self) -> f64;
}

/// Trait for scheduled state updates used by [`DoseApplyingSolout`].
pub trait ScheduledStateUpdate<S> {
    fn time(&self) -> f64;
    fn apply(&self, state: &mut S);
}

/// Validate that scheduled doses do not contain duplicate administration times.
/// Times are considered duplicates if they differ by less than `1e-10`.
pub fn validate_unique_dose_times<D: ScheduledDose>(doses: &[D]) -> Result<(), Error> {
    for (i, dose_i) in doses.iter().enumerate() {
        let time_i = dose_i.time();

        for dose_j in doses.iter().skip(i + 1) {
            let time_j = dose_j.time();

            if is_close!(time_i, time_j) {
                return Err(Error::DuplicateDoseTimes(time_i, time_j));
            }
        }
    }

    Ok(())
}

/// Custom solout for applying scheduled state updates and evenly-spaced output points.
pub struct DoseApplyingSolout<S: StateTrait<f64> + Clone, U: ScheduledStateUpdate<S> + Clone> {
    updates: Vec<U>,
    t0: f64,
    tf: f64,
    dt: f64,
    next_update_index: usize,
    last_output_time: Option<f64>,
    direction: f64,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: StateTrait<f64> + Clone, U: ScheduledStateUpdate<S> + Clone> DoseApplyingSolout<S, U> {
    pub fn new(mut updates: Vec<U>, t0: f64, tf: f64, dt: f64) -> Self {
        let direction = (tf - t0).signum();
        updates.sort_by(|a, b| a.time().total_cmp(&b.time()));
        Self {
            updates,
            t0,
            tf,
            dt,
            next_update_index: 0,
            last_output_time: None,
            direction,
            _phantom: std::marker::PhantomData,
        }
    }

    fn get_pending_updates<I: Interpolation<f64, S>>(
        &mut self,
        t_curr: f64,
        t_prev: f64,
        y_curr: &S,
        interpolator: &mut I,
    ) -> Option<(f64, S)> {
        if self.next_update_index >= self.updates.len() {
            return None;
        }

        let update_time = self.updates[self.next_update_index].time();

        if !(update_time > t_prev && update_time <= t_curr) {
            return None;
        }

        let mut y_update = if is_close!(update_time, t_curr) {
            *y_curr
        } else {
            // Safe because update_time is known to be in (t_prev, t_curr].
            interpolator.interpolate(update_time).unwrap()
        };

        while self.next_update_index < self.updates.len() {
            let update = &self.updates[self.next_update_index];
            if !is_close!(update.time(), update_time) {
                break;
            }
            update.apply(&mut y_update);
            self.next_update_index += 1;
        }

        Some((update_time, y_update))
    }
}

impl<S: Clone + StateTrait<f64>, U: ScheduledStateUpdate<S> + Clone> Solout<f64, S>
    for DoseApplyingSolout<S, U>
{
    fn solout<I>(
        &mut self,
        t_curr: f64,
        t_prev: f64,
        y_curr: &S,
        y_prev: &S,
        interpolator: &mut I,
        solution: &mut Solution<f64, S>,
    ) -> ControlFlag<f64, S>
    where
        I: Interpolation<f64, S>,
    {
        let pending_update = self.get_pending_updates(t_curr, t_prev, y_curr, interpolator);
        let next_output_time = match self.last_output_time {
            Some(t) => t + self.dt,
            None => t_prev,
        };

        if let Some((update_time, updated_state)) = pending_update {
            if is_close!(update_time, next_output_time) && next_output_time <= t_curr {
                solution.push(next_output_time, updated_state);
                self.last_output_time = Some(next_output_time);
                return ControlFlag::ModifyState(update_time, updated_state);
            }

            let update_already_output = match self.last_output_time {
                Some(last_t) => is_close!(update_time, last_t) || update_time < last_t,
                None => is_close!(update_time, t_prev),
            };

            if !update_already_output && update_time <= t_curr {
                solution.push(update_time, updated_state);
            }

            return ControlFlag::ModifyState(update_time, updated_state);
        }

        // From EvenSolout to handle evenly-spaced output points
        // Modified to use f64 directly for T which is safe given our type constraints on other
        // structs.
        let offset = self.t0 % self.dt;
        let tol = self.dt.abs() * 1e-12 + f64::EPSILON * 10.0;

        let start_t = match self.last_output_time {
            Some(t) => t + self.dt * self.direction,
            None => {
                if (t_prev - self.t0).abs() < f64::EPSILON {
                    solution.push(self.t0, *y_prev);
                    self.last_output_time = Some(self.t0);
                    self.t0 + self.dt * self.direction
                } else {
                    let rem = (t_prev - offset) % self.dt;

                    if self.direction > 0.0 {
                        if rem.abs() < f64::EPSILON {
                            t_prev
                        } else {
                            t_prev + (self.dt - rem)
                        }
                    } else if rem.abs() < f64::EPSILON {
                        t_prev
                    } else {
                        t_prev - rem
                    }
                }
            }
        };

        let mut ti = start_t;

        while (self.direction > 0.0 && ti <= t_curr) || (self.direction < 0.0 && ti >= t_curr) {
            if (self.direction > 0.0 && ti >= t_prev && ti <= t_curr)
                || (self.direction < 0.0 && ti <= t_prev && ti >= t_curr)
            {
                if self
                    .last_output_time
                    .map(|t_last| (ti - t_last).abs() <= tol)
                    .unwrap_or(false)
                {
                    // Skip near-duplicate point.
                } else {
                    let yi = interpolator.interpolate(ti).unwrap();
                    solution.push(ti, yi);
                    self.last_output_time = Some(ti);
                }
            }

            ti += self.dt * self.direction;
        }

        if t_curr == self.tf {
            match self.last_output_time {
                Some(t_last) => {
                    if (t_last - self.tf).abs() <= tol {
                        let _ = solution.pop();
                        solution.push(self.tf, *y_curr);
                        self.last_output_time = Some(self.tf);
                    } else if t_last != self.tf {
                        solution.push(self.tf, *y_curr);
                        self.last_output_time = Some(self.tf);
                    }
                }
                None => {
                    solution.push(self.tf, *y_curr);
                    self.last_output_time = Some(self.tf);
                }
            }
        }

        ControlFlag::Continue
    }
}
