# RMA Kinetics

RMA Kinetics is a library of synthetic serum marker kinetic models.
Specifically, we focus on a class of synthetic markers called Released Markers of
Activity or RMAs. 

The core library is written in Rust and high level Python bindings are available
for more ergonomic analysis with existing tools such as numpy or scipy. For the
Rust library documentation, please see the [docs.rs page](https://docs.rs/rma-kinetics/latest/rma_kinetics/).

If you're interested in a graphical interface for running the models, please see
our [web](./app/web.md) and [desktop](./app/desktop.md) application docs.

## Installation

=== ":fontawesome-brands-python: Python"
      
    ```bash
    git clone https://github.com/nsbuitrago/rma-kinetics-rs
    cd rma-kinetics-rs
    uv venv
    uvx maturin develop --uv --features py -m crates/rma-kinetics/Cargo.toml
    ```

=== ":fontawesome-brands-rust: Rust"
    ``` shell
    cargo add rma-kinetics

    # Or Cargo.toml
    [dependencies]
    rma-kinetics = "<version>" 
    ```

Please see the [GitHub README.md](https://github.com/nsbuitrago/rma-kinetics-rs)
for more details on building the library from source.

## Usage

### Running a model

There are three core model modules available, including [`constitutive`](./models/constitutive.md), [`tetoff`](./models/tetoff.md), and [`chemogenetic`](./models/chemogenetic.md).
Each module at least contains a corresponding `Model` and `State` object.

For example, to run the default constitutive model over the time span 0-100 hours,
we create a new `Model` and `State` object and solve over the desired time span.
The default parameters of the constitutive model are akin to RMA expression in the CA1
region of the hippocampus under a human-synapsin promoter.

=== ":fontawesome-brands-python: Python" 

    ```python
    from rma_kinetics.models.constitutive import Model, State
    from rma_kinetics.solvers import Dopri5

    model = Model()
    init_state = State()
    solver = Dopri5()
    t0 = 0; tf = 100; dt = 1

    solution = model.solve(t0, tf, dt, init_state, solver)
    print(f"Final plasma RMA concentration: {solution.plasma_rma[-1]}")
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    use rma_kinetics::models::constitutive::{Model, State};
    use rma_kinetics::Solve;
    use differential_equations::methods::ExplicitRungeKutta;

    let model = Model::default();
    let init_state = State::default();
    let mut solver = ExplicitRungeKutta::dopri5();
    let t0 = 0.;
    let tf = 100.;
    let dt = 1.;

    let solution = model.solve(t0, tf, dt, init_state, &mut solver);
    ```

Please see the [Models](./models/index.md) section for more details on a specific model.

### Accessing solutions

Specific arrays can be accessed directly from the returned `Solution` object
for further analysis. In Python, the `solve` method returns a `Solution`
class, while in Rust we return a `Solution` struct from the `differential_equations` crate.

Below are some examples of accessing specific species from `Solution`.

=== ":fontawesome-brands-python: Python" 

    ```python
    plasma_rma = solution.plasma_rma # access the numpy array for plasma RMA
    brain_rma = solution.brain_rma # etc.
    ```

=== ":fontawesome-brands-rust: Rust"

    ```rust
    let plasma_rma = solution.y
        .iter()
        .map(|state| state.plasma_rma).collect::<Vec<f64>>();

    let brain_rma = solution.y
        .iter()
        .map(|state| state.brain_rma).collect::<Vec<f64>>();
    ```
