//! Pharmacokinetic model utilities.
//!
//! - DoseApplyingSolout: a custom solout implementation for `differential_equations` to apply drug doses and handle evenly-spaced output points.
//! - Error: an error type for pharmacokinetic model errors.

use crate::models::cno::{CNOFields, Dose};
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
}

/// Check if two f64 values are close.
/// Here, arbitrarily used 1e-10 for some small precision tolerance.
macro_rules! is_close {
    ($a:expr, $b:expr) => {
        ($a - $b).abs() < 1e-10
    };
}

/// Custom solout for applying CNO doses and evenly-spaced output points.
/// Generic over state types that implement CNOFields.
pub struct DoseApplyingSolout<S: CNOFields + StateTrait<f64> + Clone> {
    doses: Vec<Dose>,
    t0: f64,
    tf: f64,
    dt: f64,
    next_dose_index: usize,
    last_output_time: Option<f64>,
    direction: f64,
    _phantom: std::marker::PhantomData<S>,
}

impl<S: CNOFields + StateTrait<f64> + Clone> DoseApplyingSolout<S> {
    pub fn new(doses: Vec<Dose>, t0: f64, tf: f64, dt: f64) -> Self {
        let direction = (tf - t0).signum();
        Self {
            doses,
            t0,
            tf,
            dt,
            next_dose_index: 0,
            last_output_time: None,
            direction,
            _phantom: std::marker::PhantomData,
        }
    }

    fn get_pending_dose<I: Interpolation<f64, S>>(
        &mut self,
        t_curr: f64,
        t_prev: f64,
        y_curr: &S,
        interpolator: &mut I,
    ) -> Option<(f64, S)> {
        if self.next_dose_index >= self.doses.len() {
            return None;
        }

        let dose = &self.doses[self.next_dose_index];

        if dose.time > t_prev && dose.time <= t_curr {
            let mut y_dose = if is_close!(dose.time, t_curr) {
                y_curr.clone()
            } else {
                // this is safe to unwrap since we checked that dose.time is in
                // the range [t_prev, t_curr], so we can never get a OutofBounds error.
                interpolator.interpolate(dose.time).unwrap()
            };

            // apply the dose to the peritoneal compartment
            *y_dose.peritoneal_cno_mut() += dose.nmol;
            self.next_dose_index += 1;
            Some((dose.time, y_dose))
        } else {
            None
        }
    }
}

impl<S: CNOFields + Clone + StateTrait<f64>> Solout<f64, S> for DoseApplyingSolout<S> {
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
        let pending_dose = self.get_pending_dose(t_curr, t_prev, y_curr, interpolator);
        let next_output_time = match self.last_output_time {
            Some(t) => t + self.dt,
            None => t_prev,
        };

        // Check if there's a pending dose that matches the output time
        if let Some((dose_time, dosed_state)) = pending_dose {
            // If dose time matches the output time, use the dosed state for output
            if is_close!(dose_time, next_output_time) && next_output_time <= t_curr {
                solution.push(next_output_time, dosed_state.clone());
                self.last_output_time = Some(next_output_time);
                return ControlFlag::ModifyState(dose_time, dosed_state);
            }

            // also output state if the dose time is between the evenly-spaced output points
            let dose_already_output = match self.last_output_time {
                Some(last_t) => is_close!(dose_time, last_t) || dose_time < last_t,
                None => is_close!(dose_time, t_prev),
            };

            if !dose_already_output && dose_time <= t_curr {
                solution.push(dose_time, dosed_state.clone());
            }

            return ControlFlag::ModifyState(dose_time, dosed_state);
        }
        // From EvenSolout to handle evenly-spaced output points
        // Modified to use f64 directly for T which is safe given our type constraints on other structs.
        // Determine the alignment offset (remainder when divided by dt)
        let offset = self.t0 % self.dt;

        // Tolerance for comparing time points to avoid near-duplicates from FP error
        // Scales with dt and also includes a small absolute epsilon
        let tol = self.dt.abs() * 1e-12 + f64::EPSILON * 10.0;

        // Start from the last output point if available, otherwise from t_prev
        let start_t = match self.last_output_time {
            Some(t) => t + self.dt * self.direction,
            None => {
                // First time through, we need to include t0
                if (t_prev - self.t0).abs() < f64::EPSILON {
                    solution.push(self.t0, y_prev.clone());
                    self.last_output_time = Some(self.t0);
                    self.t0 + self.dt * self.direction
                } else {
                    // Find the next aligned point after t_prev
                    let rem = (t_prev - offset) % self.dt;

                    if self.direction > 0.0 {
                        // For forward integration
                        if rem.abs() < f64::EPSILON {
                            t_prev
                        } else {
                            t_prev + (self.dt - rem)
                        }
                    } else {
                        // For backward integration
                        if rem.abs() < f64::EPSILON {
                            t_prev
                        } else {
                            t_prev - rem
                        }
                    }
                }
            }
        };

        let mut ti = start_t;

        // Interpolate between steps
        while (self.direction > 0.0 && ti <= t_curr) || (self.direction < 0.0 && ti >= t_curr) {
            // Only output if the point falls within the current step
            if (self.direction > 0.0 && ti >= t_prev && ti <= t_curr)
                || (self.direction < 0.0 && ti <= t_prev && ti >= t_curr)
            {
                // Skip if this ti is essentially the same as the last output
                if self
                    .last_output_time
                    .map(|t_last| (ti - t_last).abs() <= tol)
                    .unwrap_or(false)
                {
                    // Do nothing; advance to next ti
                } else {
                    let yi = interpolator.interpolate(ti).unwrap();
                    solution.push(ti, yi);
                    self.last_output_time = Some(ti);
                }
            }

            // Move to the next point
            ti += self.dt * self.direction;
        }

        // Include final point if this step reaches tf and we haven't added it yet
        if t_curr == self.tf {
            match self.last_output_time {
                Some(t_last) => {
                    if (t_last - self.tf).abs() <= tol {
                        // Replace the near-duplicate last point with the exact final time
                        let _ = solution.pop();
                        solution.push(self.tf, y_curr.clone());
                        self.last_output_time = Some(self.tf);
                    } else if t_last != self.tf {
                        solution.push(self.tf, y_curr.clone());
                        self.last_output_time = Some(self.tf);
                    }
                }
                None => {
                    solution.push(self.tf, y_curr.clone());
                    self.last_output_time = Some(self.tf);
                }
            }
        }

        // Continue the integration
        ControlFlag::Continue
    }
}
