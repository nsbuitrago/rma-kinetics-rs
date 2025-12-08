# Solvers

The Python API provides a few different ODE solvers for numerical integration.
The simple models (constitutive, Tet-Off, and Oscillation) can usually be solved
with the `Dopri5` solver. We recommend using the `Kvaerno3` solver for the
`chemogenetic` model since it inlucdes RMA, dox, and CNO dynamics which are usually
on different scales.

## Available solvers:

::: rma_kinetics.solvers.Dopri5
    options:
      show_source: false
      members:
        - __init__

::: rma_kinetics.solvers.Kvaerno3
    options:
      show_source: false
      members:
        - __init__

The Rust library uses the [`differential_equations`](https://github.com/Ryan-D-Gast/differential-equations)
crate directly. You can use any solver provided by this crate for more direct control.
