use thiserror::Error as ErrorTrait;

#[derive(ErrorTrait, Debug)]
pub enum Error {
    #[error("Bioavailability must be between 0 and 1")]
    InvalidBioavailability(f64),
}

