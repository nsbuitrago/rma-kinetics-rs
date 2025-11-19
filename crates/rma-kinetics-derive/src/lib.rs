use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, parse_macro_input};

#[cfg(feature = "py")]
use syn::{Expr, Ident, Lit, Meta, Token, punctuated::Punctuated};

#[proc_macro_derive(Solve)]
pub fn solve_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = input.ident;

    // Assume the state type is `State<f64>` and is defined in the same module.
    let state_type = quote! { State<f64> };

    let expanded = quote! {
        impl crate::solve::Solve for #name {
            type State = #state_type;

            fn solve<S>(
                &self,
                t0: f64,
                tf: f64,
                dt: f64,
                init_state: Self::State,
                solver: &mut S,
            ) -> Result<differential_equations::prelude::Solution<f64, Self::State>, differential_equations::error::Error<f64, Self::State>>
            where
                S: differential_equations::ode::OrdinaryNumericalMethod<f64, Self::State> + differential_equations::interpolate::Interpolation<f64, Self::State>
            {
                let problem = differential_equations::ode::ODEProblem::new(self, t0, tf, init_state);
                problem.even(dt).solve(solver)
            }
        }
    };

    TokenStream::from(expanded)
}

#[cfg(feature = "py")]
#[proc_macro_derive(PySolve, attributes(py_solve))]
pub fn py_solve_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;

    let mut variant_name = None;
    for attr in &input.attrs {
        if attr.path().is_ident("py_solve") {
            let list: Punctuated<Meta, Token![,]> =
                attr.parse_args_with(Punctuated::parse_terminated).unwrap();
            for meta in list {
                if let Meta::NameValue(name_value) = meta {
                    if name_value.path.is_ident("variant") {
                        if let Expr::Lit(expr_lit) = name_value.value {
                            if let Lit::Str(lit_str) = expr_lit.lit {
                                variant_name = Some(Ident::new(&lit_str.value(), lit_str.span()));
                            }
                        }
                    }
                }
            }
        }
    }

    let variant_ident = match variant_name {
        Some(name) => name,
        None => {
            return TokenStream::from(quote! {
                compile_error!("Expected a `#[py_solve(variant = \"...\")]` attribute");
            });
        }
    };

    // Assume the state type is `State<f64>` and is defined in the same module.
    let state_type = quote! { State<f64> };

    let expanded = quote! {
        impl crate::solve::PySolve for #name {
            type State = #state_type;

            fn solve(
                &self,
                t0: f64,
                tf: f64,
                dt: f64,
                init_state: Self::State,
                solver: crate::solve::PySolver,
            ) -> Result<crate::solve::PySolution, differential_equations::error::Error<f64, Self::State>>
            {
                let mut internal_solver = match solver.solver_type.as_str() {
                    "dopri5" => differential_equations::methods::ExplicitRungeKutta::dopri5()
                        .rtol(solver.rtol)
                        .atol(solver.atol)
                        .h0(solver.dt0)
                        .h_min(solver.min_dt)
                        .h_max(solver.max_dt)
                        .max_steps(solver.max_steps)
                        .max_rejects(solver.max_rejected_steps)
                        .safety_factor(solver.safety_factor)
                        .min_scale(solver.min_scale)
                        .max_scale(solver.max_scale),
                    _ => panic!("Solver not supported"),
                };

                let problem = differential_equations::ode::ODEProblem::new(self, t0, tf, init_state);
                let solution = problem.even(dt).solve(&mut internal_solver)?;

                Ok(crate::solve::PySolution {
                    inner: crate::solve::InnerSolution::#variant_ident(solution),
                })
            }
        }
    };

    TokenStream::from(expanded)
}
