# Rust

## Installation

The Rust crate can be installed with `cargo`. For example, to get the latest
version,

```bash
cargo add rma-kinetics
```

You can find all available versions on [crates.io]().

## Getting Started

This crate provides model implementations for

- constitutive expression
- dox gated expression in the TetOff system
- neuronal activity induced expressed
- oscillating gene expression input

Models can be created from their respective modules. For example,

```rust
use rma_kinetics::models::constitutive;

fn main() {
    let default_model = constitutive::Model();

    // create a new model with custom parameters
    let custom_model = constitutive::Model::new(0.4, 0.5, 0.006);

    // or use the builder pattern...
    // here we just update the production rate as an example
    let new_model = constitutive::Model::builder().prod_rate(0.3)

    // ...
}
```

To solve the models over a given period of time, we use the solvers provided by
the `differential_equations` dependency. From here, we can use the provided `Solve`
trait and use the `solve` method on our model.

```rust
use rma_kinetics::{models::constitutive, solve::Solve};
use differential_equations::methods::ExplicitRungeKutta;

fn main() {
    let default_model = constitutive::Model();
    let mut solver = ExplicitRungeKutta::Dopri5();

    let solution = default_model.solve(
        0., // start time (hr)
        100., // stop time (hr)
        1., // create solution with even steps every 1hr
        constitutive::State::zeros(), // use an initial vector of 0s
        &mut solver
    );

    // ...unpack the result and work with solution
}
```

By default the `solve` method returns a `Solution` struct with evenly spaced steps
between the start and stop time. If you need more control over the solving, you
can use the `differential_equations` crate `ODEProblem` directly. For example,

```rust
use rma_kinetics::{models::constitutive, solve::Solve};
use differential_equations::{methods::ExplicitRungeKutta, ode::ODEProblem};

fn main() {
    let default_model = constitutive::Model();
    let mut solver = ExplicitRungeKutta::Dopri5();

    let problem = differential_equations::ode::ODEProblem::new(
        default_model,
        1.,
        100.,
        State::zeros());

    let solution = problem.solve(solver);
    // ...unpack the result and work with solution
}
```

(based on chemogenetic activation of
synthetic receptors and the robust activity marking system).
