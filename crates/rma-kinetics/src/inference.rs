pub const DEFAULT_WEIGHT: f64 = 1.0;

/// Weighted plasma RMA observation at a given timepoint.
pub struct Observation {
    pub time: f64,
    pub plasma_rma: f64,
    pub weight: Option<f64>,
}

#[derive(Clone, Copy, Debug)]
pub struct Cotangent {
    pub time: f64,
    pub value: f64,
}
